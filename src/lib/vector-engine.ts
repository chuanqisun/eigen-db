/**
 * THE CORE ENGINE
 *
 * Orchestrates storage, memory, compute, and lexicon layers to provide
 * a high-level vector search API.
 */

import { normalize, searchAll } from "./compute";
import { VectorCapacityExceededError } from "./errors";
import { encodeLexicon, buildLexiconIndex, decodeLexiconAtOffset } from "./lexicon";
import { MemoryManager } from "./memory-manager";
import { ResultSet } from "./result-set";
import { getSimdWasmBinary } from "./simd-binary";
import type { StorageProvider } from "./storage";
import { OPFSStorageProvider } from "./storage";
import type { EmbeddingFunction, EngineConfig } from "./types";
import { instantiateWasm, type WasmExports } from "./wasm-compute";

const VECTORS_FILE = "vectors.bin";
const LEXICON_FILE = "lexicon.bin";

export interface VectorEngineOptions extends EngineConfig {
  /** Override the storage provider (default: OPFS). Useful for testing. */
  storage?: StorageProvider;
  /** Pre-compiled WASM binary. If not provided, uses the embedded SIMD binary. Set to null to force JS-only mode. */
  wasmBinary?: Uint8Array | null;
}

export class VectorEngine {
  private readonly memoryManager: MemoryManager;
  private readonly storage: StorageProvider;
  private readonly embedder: EmbeddingFunction;
  private readonly dimensions: number;
  private lexiconData: Uint8Array;
  private lexiconIndex: Uint32Array;
  private wasmExports: WasmExports | null;

  private constructor(
    memoryManager: MemoryManager,
    storage: StorageProvider,
    embedder: EmbeddingFunction,
    dimensions: number,
    lexiconData: Uint8Array,
    lexiconIndex: Uint32Array,
    wasmExports: WasmExports | null,
  ) {
    this.memoryManager = memoryManager;
    this.storage = storage;
    this.embedder = embedder;
    this.dimensions = dimensions;
    this.lexiconData = lexiconData;
    this.lexiconIndex = lexiconIndex;
    this.wasmExports = wasmExports;
  }

  /**
   * Initializes the engine.
   * Reads OPFS vectors.bin directly into WASM memory (Zero-copy).
   */
  static async open(config: EngineConfig): Promise<VectorEngine>;
  static async open(config: VectorEngineOptions): Promise<VectorEngine>;
  static async open(config: VectorEngineOptions): Promise<VectorEngine> {
    const storage = config.storage ?? new OPFSStorageProvider(config.name);

    // Load existing data from storage
    const [vectorBytes, lexiconBytes] = await Promise.all([
      storage.readAll(VECTORS_FILE),
      storage.readAll(LEXICON_FILE),
    ]);

    const vectorCount = vectorBytes.byteLength / (config.dimensions * 4);

    // Initialize memory manager with space for existing vectors
    const mm = new MemoryManager(config.dimensions, vectorCount);

    // Zero-copy load: write vector bytes directly into WASM memory
    if (vectorBytes.byteLength > 0) {
      mm.loadVectorBytes(vectorBytes, vectorCount);
    }

    // Build lexicon index for O(1) text lookups
    const lexiconIndex = buildLexiconIndex(lexiconBytes);

    // Try to instantiate WASM SIMD module
    let wasmExports: WasmExports | null = null;
    const wasmBinary = config.wasmBinary !== undefined ? config.wasmBinary : getSimdWasmBinary();
    if (wasmBinary !== null) {
      try {
        wasmExports = await instantiateWasm(wasmBinary, mm.memory);
      } catch {
        // Fall back to JS compute
      }
    }

    return new VectorEngine(
      mm,
      storage,
      config.embedder,
      config.dimensions,
      lexiconBytes,
      lexiconIndex,
      wasmExports,
    );
  }

  /** Total records in the database */
  get size(): number {
    return this.memoryManager.vectorCount;
  }

  /**
   * Fetches embeddings, normalizes them via WASM, appends to OPFS,
   * and updates the WASM memory buffer.
   */
  async add(text: string | string[]): Promise<void> {
    const texts = Array.isArray(text) ? text : [text];
    if (texts.length === 0) return;

    // Check capacity before embedding (to fail fast)
    const newTotal = this.size + texts.length;
    if (newTotal > this.memoryManager.maxVectors) {
      throw new VectorCapacityExceededError(this.memoryManager.maxVectors);
    }

    // Get embeddings from the user-provided function
    const embeddings = await this.embedder(texts);

    // Validate embeddings
    for (const emb of embeddings) {
      if (emb.length !== this.dimensions) {
        throw new Error(
          `Embedding dimension mismatch: expected ${this.dimensions}, got ${emb.length}`,
        );
      }
    }

    // Normalize each vector in-place
    for (const emb of embeddings) {
      this.normalizeVector(emb);
    }

    // Ensure WASM memory is large enough
    this.memoryManager.ensureCapacity(embeddings.length);

    // Append to WASM memory
    this.memoryManager.appendVectors(embeddings);

    // Persist to storage
    const vectorBytes = new Uint8Array(embeddings.length * this.dimensions * 4);
    let offset = 0;
    for (const emb of embeddings) {
      vectorBytes.set(new Uint8Array(emb.buffer, emb.byteOffset, emb.byteLength), offset);
      offset += emb.byteLength;
    }

    const lexiconBytes = encodeLexicon(texts);

    await Promise.all([
      this.storage.append(VECTORS_FILE, vectorBytes),
      this.storage.append(LEXICON_FILE, lexiconBytes),
    ]);

    // Update lexicon data and index
    const newLexiconData = new Uint8Array(this.lexiconData.byteLength + lexiconBytes.byteLength);
    newLexiconData.set(this.lexiconData, 0);
    newLexiconData.set(lexiconBytes, this.lexiconData.byteLength);
    this.lexiconData = newLexiconData;
    this.lexiconIndex = buildLexiconIndex(this.lexiconData);
  }

  /**
   * Embeds the query, executes dot-products across the entire DB,
   * sorts the results via JS TypedArrays, and returns a lazy ResultSet.
   */
  async search(query: string, topK?: number): Promise<ResultSet> {
    const k = topK ?? this.size;

    if (this.size === 0) {
      return ResultSet.fromScores(new Float32Array(0), () => "", 0);
    }

    // Embed the query
    const [queryVector] = await this.embedder([query]);
    if (queryVector.length !== this.dimensions) {
      throw new Error(
        `Query embedding dimension mismatch: expected ${this.dimensions}, got ${queryVector.length}`,
      );
    }

    // Normalize query
    this.normalizeVector(queryVector);

    // Write query to WASM memory
    this.memoryManager.writeQuery(queryVector);

    // Ensure memory has space for scores buffer
    this.memoryManager.ensureCapacity(0);

    // Execute search
    const scoresOffset = this.memoryManager.scoresOffset;
    if (this.wasmExports) {
      this.wasmExports.search_all(
        this.memoryManager.queryOffset,
        this.memoryManager.dbOffset,
        scoresOffset,
        this.size,
        this.dimensions,
      );
    } else {
      // JS fallback
      const queryView = new Float32Array(
        this.memoryManager.memory.buffer,
        this.memoryManager.queryOffset,
        this.dimensions,
      );
      const dbView = new Float32Array(
        this.memoryManager.memory.buffer,
        this.memoryManager.dbOffset,
        this.size * this.dimensions,
      );
      const scoresView = new Float32Array(
        this.memoryManager.memory.buffer,
        scoresOffset,
        this.size,
      );
      searchAll(queryView, dbView, scoresView, this.size, this.dimensions);
    }

    // Read scores (make a copy so the buffer can be reused)
    const scores = new Float32Array(this.memoryManager.readScores());

    // Create lazy text resolver using lexicon index
    const lexiconData = this.lexiconData;
    const lexiconIndex = this.lexiconIndex;
    const resolveText = (dbIndex: number): string => {
      return decodeLexiconAtOffset(lexiconData, lexiconIndex[dbIndex]);
    };

    return ResultSet.fromScores(scores, resolveText, k);
  }

  /**
   * Normalize a vector using WASM (if available) or JS fallback.
   */
  private normalizeVector(vec: Float32Array): void {
    if (this.wasmExports) {
      // For WASM normalize, we temporarily write to a scratch area in memory
      // Use the query buffer as scratch space
      const ptr = this.memoryManager.queryOffset;
      new Float32Array(this.memoryManager.memory.buffer, ptr, vec.length).set(vec);
      this.wasmExports.normalize(ptr, vec.length);
      const normalized = new Float32Array(this.memoryManager.memory.buffer, ptr, vec.length);
      vec.set(normalized);
    } else {
      normalize(vec);
    }
  }
}
