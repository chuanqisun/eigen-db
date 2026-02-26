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
 * - Last-write-wins semantics for duplicate keys (append-only storage)
 */

import { normalize, searchAll } from "./compute";
import { VectorCapacityExceededError } from "./errors";
import { decodeLexicon, encodeLexicon } from "./lexicon";
import { MemoryManager } from "./memory-manager";
import { ResultSet } from "./result-set";
import { getSimdWasmBinary } from "./simd-binary";
import type { StorageProvider } from "./storage";
import { OPFSStorageProvider } from "./storage";
import type { OpenOptions, OpenOptionsInternal, QueryOptions, SetOptions, VectorInput } from "./types";
import { instantiateWasm, type WasmExports } from "./wasm-compute";

const VECTORS_FILE = "vectors.bin";
const KEYS_FILE = "keys.bin";

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
   * Returns a ResultSet sorted by descending similarity score.
   */
  query(value: VectorInput, options?: QueryOptions): ResultSet {
    this.assertOpen();

    const k = options?.topK ?? this.size;

    if (this.size === 0) {
      return ResultSet.fromScores(new Float32Array(0), () => "", 0);
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

    return ResultSet.fromScores(scores, resolveKey, k);
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
