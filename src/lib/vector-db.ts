/**
 * VectorDB — Key-Value Vector Database
 *
 * Decoupled from embedding providers. Users pass pre-computed vectors
 * as number arrays (or Float32Array) with string keys.
 *
 * Supports:
 * - set/get/setMany/getMany for key-value CRUD
 * - query for similarity search (dot product on normalized vectors)
 * - flush to persist, close to flush+release, clear to wipe
 * - Streaming export/import using Web Streams API
 * - Last-write-wins semantics for duplicate keys (append-only storage)
 */

import { normalize, searchAll } from "./compute";
import { VectorCapacityExceededError } from "./errors";
import { decodeLexicon, encodeLexicon } from "./lexicon";
import { MemoryManager } from "./memory-manager";
import type { ResultItem } from "./result-set";
import { iterableResults, topKResults } from "./result-set";
import { getSimdWasmBinary } from "./simd-binary";
import type { StorageProvider } from "./storage";
import { OPFSStorageProvider } from "./storage";
import type { OpenOptions, OpenOptionsInternal, QueryOptions, SetOptions, VectorInput } from "./types";
import { instantiateWasm, type WasmExports } from "./wasm-compute";

const VECTORS_FILE = "vectors.bin";
const KEYS_FILE = "keys.bin";

/** Binary export format magic bytes: "EGDB" */
const EXPORT_MAGIC = 0x42444745; // "EGDB" in little-endian
const EXPORT_VERSION = 1;
const EXPORT_HEADER_BYTES = 24;

/** Chunk size for streaming export/import (64KB, matches WASM page size) */
const STREAM_CHUNK_SIZE = 65536;

export class VectorDB {
  private readonly memoryManager: MemoryManager;
  private readonly storage: StorageProvider;
  private readonly dimensions: number;
  private readonly shouldNormalize: boolean;
  private wasmExports: WasmExports | null;

  /** Maps key to its slot index in the vector array */
  private keyToSlot: Map<string, number>;

  /** Maps slot index back to its key */
  private slotToKey: string[];

  /** Whether this instance has been closed */
  private closed = false;

  private constructor(
    memoryManager: MemoryManager,
    storage: StorageProvider,
    dimensions: number,
    shouldNormalize: boolean,
    wasmExports: WasmExports | null,
    keyToSlot: Map<string, number>,
    slotToKey: string[],
  ) {
    this.memoryManager = memoryManager;
    this.storage = storage;
    this.dimensions = dimensions;
    this.shouldNormalize = shouldNormalize;
    this.wasmExports = wasmExports;
    this.keyToSlot = keyToSlot;
    this.slotToKey = slotToKey;
  }

  /**
   * Opens a VectorDB instance.
   * Loads existing data from storage into WASM memory.
   */
  static async open(options: OpenOptions): Promise<VectorDB>;
  static async open(options: OpenOptionsInternal): Promise<VectorDB>;
  static async open(options: OpenOptionsInternal): Promise<VectorDB> {
    const name = options.name ?? "default";
    const storage = options.storage ?? new OPFSStorageProvider(name);
    const shouldNormalize = options.normalize !== false;

    // Load existing data from storage
    const [vectorBytes, keysBytes] = await Promise.all([storage.readAll(VECTORS_FILE), storage.readAll(KEYS_FILE)]);

    // Decode stored keys
    const keys = keysBytes.byteLength > 0 ? decodeLexicon(keysBytes) : [];
    const vectorCount = vectorBytes.byteLength / (options.dimensions * 4);

    // Build key-to-slot mapping.
    // flush() always writes deduplicated state, so keys are unique on load.
    const keyToSlot = new Map<string, number>();
    const slotToKey: string[] = [];

    for (let i = 0; i < keys.length; i++) {
      keyToSlot.set(keys[i], i);
      slotToKey[i] = keys[i];
    }

    // Initialize memory manager
    const mm = new MemoryManager(options.dimensions, vectorCount);

    if (vectorBytes.byteLength > 0) {
      mm.loadVectorBytes(vectorBytes, vectorCount);
    }

    // Try to instantiate WASM SIMD module
    let wasmExports: WasmExports | null = null;
    const wasmBinary = options.wasmBinary !== undefined ? options.wasmBinary : getSimdWasmBinary();
    if (wasmBinary !== null) {
      try {
        wasmExports = await instantiateWasm(wasmBinary, mm.memory);
      } catch {
        // Fall back to JS compute
      }
    }

    return new VectorDB(mm, storage, options.dimensions, shouldNormalize, wasmExports, keyToSlot, slotToKey);
  }

  /** Total number of key-value pairs in the database */
  get size(): number {
    return this.keyToSlot.size;
  }

  /**
   * Set a key-value pair. If the key already exists, its vector is overwritten (last-write-wins).
   * The value is a number[] or Float32Array of length equal to the configured dimensions.
   */
  set(key: string, value: VectorInput, options?: SetOptions): void {
    this.assertOpen();

    if (value.length !== this.dimensions) {
      throw new Error(`Vector dimension mismatch: expected ${this.dimensions}, got ${value.length}`);
    }

    // Convert to Float32Array (also clones to avoid mutating caller's array)
    const vec = new Float32Array(value);

    // Normalize if needed
    const doNormalize = options?.normalize ?? this.shouldNormalize;
    if (doNormalize) {
      this.normalizeVector(vec);
    }

    const existingSlot = this.keyToSlot.get(key);
    if (existingSlot !== undefined) {
      // Overwrite existing slot
      this.memoryManager.writeVector(existingSlot, vec);
    } else {
      // Append new entry
      const newTotal = this.memoryManager.vectorCount + 1;
      if (newTotal > this.memoryManager.maxVectors) {
        throw new VectorCapacityExceededError(this.memoryManager.maxVectors);
      }
      this.memoryManager.ensureCapacity(1);
      const slotIndex = this.memoryManager.vectorCount;
      this.memoryManager.appendVectors([vec]);
      this.keyToSlot.set(key, slotIndex);
      this.slotToKey[slotIndex] = key;
    }
  }

  /**
   * Get the stored vector for a key. Returns undefined if the key does not exist.
   * Returns a copy of the stored vector as a plain number array.
   */
  get(key: string): number[] | undefined {
    this.assertOpen();

    const slot = this.keyToSlot.get(key);
    if (slot === undefined) return undefined;

    // Return a plain array copy so callers can't corrupt WASM memory
    return Array.from(this.memoryManager.readVector(slot));
  }

  /**
   * Set multiple key-value pairs at once. Last-write-wins applies within the batch.
   */
  setMany(entries: [string, VectorInput][]): void {
    for (const [key, value] of entries) {
      this.set(key, value);
    }
  }

  /**
   * Get vectors for multiple keys. Returns undefined for keys that don't exist.
   */
  getMany(keys: string[]): (number[] | undefined)[] {
    return keys.map((key) => this.get(key));
  }

  /**
   * Search for the most similar vectors to the given query vector.
   *
   * Default: returns a plain ResultItem[] sorted by descending similarity.
   * With `{ iterable: true }`: returns a lazy Iterable<ResultItem> where keys
   * are resolved only as each item is consumed.
   *
   * Similarity is the dot product of query and stored vectors. With
   * normalization (default), this equals cosine similarity: 1 = identical,
   * -1 = opposite.
   */
  query(value: VectorInput, options: QueryOptions & { iterable: true }): Iterable<ResultItem>;
  query(value: VectorInput, options?: QueryOptions): ResultItem[];
  query(value: VectorInput, options?: QueryOptions): ResultItem[] | Iterable<ResultItem> {
    this.assertOpen();

    const k = options?.topK ?? Infinity;
    const minSimilarity = options?.minSimilarity;
    const iterable = options && "iterable" in options && options.iterable;

    if (this.size === 0) {
      return [];
    }

    if (value.length !== this.dimensions) {
      throw new Error(`Query vector dimension mismatch: expected ${this.dimensions}, got ${value.length}`);
    }

    // Convert to Float32Array and optionally normalize the query vector
    const queryVec = new Float32Array(value);
    const doNormalize = options?.normalize ?? this.shouldNormalize;
    if (doNormalize) {
      this.normalizeVector(queryVec);
    }

    // Write query to WASM memory
    this.memoryManager.writeQuery(queryVec);

    // Ensure memory has space for scores buffer
    this.memoryManager.ensureCapacity(0);

    // Total vectors in memory
    const totalVectors = this.memoryManager.vectorCount;

    // Execute search
    const scoresOffset = this.memoryManager.scoresOffset;
    if (this.wasmExports) {
      this.wasmExports.search_all(
        this.memoryManager.queryOffset,
        this.memoryManager.dbOffset,
        scoresOffset,
        totalVectors,
        this.dimensions,
      );
    } else {
      const queryView = new Float32Array(
        this.memoryManager.memory.buffer,
        this.memoryManager.queryOffset,
        this.dimensions,
      );
      const dbView = new Float32Array(
        this.memoryManager.memory.buffer,
        this.memoryManager.dbOffset,
        totalVectors * this.dimensions,
      );
      const scoresView = new Float32Array(this.memoryManager.memory.buffer, scoresOffset, totalVectors);
      searchAll(queryView, dbView, scoresView, totalVectors, this.dimensions);
    }

    // Read scores (make a copy so the buffer can be reused)
    const scores = new Float32Array(this.memoryManager.readScores());

    // Resolve key from slot index
    const slotToKey = this.slotToKey;
    const resolveKey = (slotIndex: number): string => {
      return slotToKey[slotIndex];
    };

    if (iterable) {
      return iterableResults(scores, resolveKey, k, minSimilarity);
    }
    return topKResults(scores, resolveKey, k, minSimilarity);
  }

  /**
   * Persist the current in-memory state to storage.
   */
  async flush(): Promise<void> {
    this.assertOpen();

    const totalVectors = this.memoryManager.vectorCount;

    // Serialize vectors from WASM memory
    const vectorBytes = new Uint8Array(totalVectors * this.dimensions * 4);
    if (totalVectors > 0) {
      const src = new Uint8Array(
        this.memoryManager.memory.buffer,
        this.memoryManager.dbOffset,
        totalVectors * this.dimensions * 4,
      );
      vectorBytes.set(src);
    }

    // Serialize keys using lexicon format
    const keysBytes = encodeLexicon(this.slotToKey);

    await Promise.all([this.storage.write(VECTORS_FILE, vectorBytes), this.storage.write(KEYS_FILE, keysBytes)]);
  }

  /**
   * Flush data to storage and release the instance.
   * The instance cannot be used after close.
   */
  async close(): Promise<void> {
    if (this.closed) return;
    await this.flush();
    this.closed = true;
  }

  /**
   * Clear all data from the database and storage.
   */
  async clear(): Promise<void> {
    this.assertOpen();

    this.keyToSlot.clear();
    this.slotToKey.length = 0;
    this.memoryManager.reset();

    await this.storage.destroy();
  }

  /**
   * Export the entire database as a ReadableStream of binary chunks.
   *
   * Format: [Header 24 bytes][Vector data][Keys data]
   * Header: magic(4) + version(4) + dimensions(4) + vectorCount(4) + vectorDataLen(4) + keysDataLen(4)
   *
   * Vectors are streamed in 64KB chunks from WASM memory to avoid large
   * heap allocations.
   */
  async export(): Promise<ReadableStream<Uint8Array>> {
    this.assertOpen();

    const totalVectors = this.memoryManager.vectorCount;
    const vectorDataLen = totalVectors * this.dimensions * 4;

    // Encode keys (typically much smaller than vectors)
    const keysBytes = encodeLexicon(this.slotToKey);
    const keysDataLen = keysBytes.byteLength;

    // Build header
    const header = new ArrayBuffer(EXPORT_HEADER_BYTES);
    const headerView = new DataView(header);
    headerView.setUint32(0, EXPORT_MAGIC, true);
    headerView.setUint32(4, EXPORT_VERSION, true);
    headerView.setUint32(8, this.dimensions, true);
    headerView.setUint32(12, totalVectors, true);
    headerView.setUint32(16, vectorDataLen, true);
    headerView.setUint32(20, keysDataLen, true);

    const mm = this.memoryManager;
    let phase: "header" | "vectors" | "keys" | "done" = "header";
    let vectorOffset = 0;

    return new ReadableStream<Uint8Array>({
      pull(controller) {
        switch (phase) {
          case "header":
            controller.enqueue(new Uint8Array(header));
            phase = vectorDataLen > 0 ? "vectors" : keysDataLen > 0 ? "keys" : "done";
            if (phase === "done") controller.close();
            break;

          case "vectors": {
            const remaining = vectorDataLen - vectorOffset;
            const chunkSize = Math.min(remaining, STREAM_CHUNK_SIZE);
            const chunk = new Uint8Array(chunkSize);
            chunk.set(new Uint8Array(mm.memory.buffer, mm.dbOffset + vectorOffset, chunkSize));
            controller.enqueue(chunk);
            vectorOffset += chunkSize;
            if (vectorOffset >= vectorDataLen) {
              phase = keysDataLen > 0 ? "keys" : "done";
              if (phase === "done") controller.close();
            }
            break;
          }

          case "keys":
            controller.enqueue(keysBytes);
            phase = "done";
            controller.close();
            break;
        }
      },
    });
  }

  /**
   * Import data from a ReadableStream, replacing all existing data.
   * Performs a dimension check against the configured dimensions.
   *
   * Vectors are streamed directly into WASM memory in chunks to avoid
   * large heap allocations.
   */
  async import(stream: ReadableStream<Uint8Array>): Promise<void> {
    this.assertOpen();

    const sr = new StreamReader(stream);

    // Read and validate header
    const headerBytes = await sr.readExact(EXPORT_HEADER_BYTES);
    const headerView = new DataView(headerBytes.buffer, headerBytes.byteOffset, headerBytes.byteLength);

    const magic = headerView.getUint32(0, true);
    if (magic !== EXPORT_MAGIC) {
      throw new Error("Invalid import data: unrecognized format");
    }

    const version = headerView.getUint32(4, true);
    if (version !== EXPORT_VERSION) {
      throw new Error(`Unsupported import version: expected ${EXPORT_VERSION}, got ${version}`);
    }

    const dimensions = headerView.getUint32(8, true);
    if (dimensions !== this.dimensions) {
      throw new Error(`Import dimension mismatch: expected ${this.dimensions}, got ${dimensions}`);
    }

    const vectorCount = headerView.getUint32(12, true);
    const vectorDataLen = headerView.getUint32(16, true);
    const keysDataLen = headerView.getUint32(20, true);

    // Clear existing state
    this.keyToSlot.clear();
    this.slotToKey.length = 0;
    this.memoryManager.reset();

    // Stream vectors directly into WASM memory in chunks
    if (vectorCount > 0) {
      this.memoryManager.ensureCapacity(vectorCount);

      let offset = 0;
      await sr.readChunked(vectorDataLen, (chunk) => {
        new Uint8Array(this.memoryManager.memory.buffer, this.memoryManager.dbOffset + offset, chunk.byteLength).set(
          chunk,
        );
        offset += chunk.byteLength;
      });

      this.memoryManager.setVectorCount(vectorCount);
    }

    // Read and decode keys (typically much smaller than vectors)
    if (keysDataLen > 0) {
      const keysBytes = await sr.readExact(keysDataLen);
      const keys = decodeLexicon(keysBytes);
      for (let i = 0; i < keys.length; i++) {
        this.keyToSlot.set(keys[i], i);
        this.slotToKey[i] = keys[i];
      }
    }

    sr.release();
  }

  /**
   * Normalize a vector using WASM (if available) or JS fallback.
   */
  private normalizeVector(vec: Float32Array): void {
    if (this.wasmExports) {
      const ptr = this.memoryManager.queryOffset;
      new Float32Array(this.memoryManager.memory.buffer, ptr, vec.length).set(vec);
      this.wasmExports.normalize(ptr, vec.length);
      const normalized = new Float32Array(this.memoryManager.memory.buffer, ptr, vec.length);
      vec.set(normalized);
    } else {
      normalize(vec);
    }
  }

  private assertOpen(): void {
    if (this.closed) {
      throw new Error("VectorDB instance has been closed");
    }
  }
}

/**
 * Helper for reading exact byte counts from a ReadableStream.
 * Handles chunk boundaries transparently.
 */
class StreamReader {
  private reader: ReadableStreamDefaultReader<Uint8Array>;
  private buffer = new Uint8Array(0);

  constructor(stream: ReadableStream<Uint8Array>) {
    this.reader = stream.getReader();
  }

  /** Read exactly `n` bytes, accumulating from stream chunks as needed. */
  async readExact(n: number): Promise<Uint8Array> {
    if (n === 0) return new Uint8Array(0);

    // Fast path: buffer already has enough data
    if (this.buffer.byteLength >= n) {
      const result = this.buffer.subarray(0, n);
      this.buffer = this.buffer.subarray(n);
      return result;
    }

    // Collect chunks until we have enough bytes
    const chunks: Uint8Array[] = [];
    let total = this.buffer.byteLength;
    if (total > 0) {
      chunks.push(this.buffer);
      this.buffer = new Uint8Array(0);
    }

    while (total < n) {
      const { done, value } = await this.reader.read();
      if (done) throw new Error("Invalid import data: unexpected end of stream");
      chunks.push(value);
      total += value.byteLength;
    }

    // Combine into a single buffer
    const combined = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.byteLength;
    }

    if (total > n) {
      this.buffer = combined.subarray(n);
      return combined.subarray(0, n);
    }
    return combined;
  }

  /**
   * Read `totalBytes` from the stream in chunks, calling `onChunk` for each.
   * Avoids allocating a single large buffer for the full data.
   */
  async readChunked(totalBytes: number, onChunk: (chunk: Uint8Array) => void): Promise<void> {
    let remaining = totalBytes;

    // Drain buffered leftover first
    if (this.buffer.byteLength > 0) {
      const take = Math.min(this.buffer.byteLength, remaining);
      onChunk(this.buffer.subarray(0, take));
      remaining -= take;
      this.buffer = this.buffer.subarray(take);
    }

    while (remaining > 0) {
      const { done, value } = await this.reader.read();
      if (done) throw new Error("Invalid import data: unexpected end of stream");

      if (value.byteLength <= remaining) {
        onChunk(value);
        remaining -= value.byteLength;
      } else {
        onChunk(value.subarray(0, remaining));
        this.buffer = value.slice(remaining);
        remaining = 0;
      }
    }
  }

  /** Release the stream reader lock. */
  release(): void {
    this.reader.releaseLock();
  }
}
