/**
 * Memory Manager for WASM shared memory.
 *
 * Memory Layout:
 * [ 0x00000 ] -> Query Vector Buffer (Fixed, dimensions * 4 bytes, aligned to 64KB page)
 * [ DB_OFFSET ] -> Vector Database (Grows dynamically)
 * [ Dynamic ] -> Scores Buffer (Mapped after DB during search)
 */

/** WASM page size is 64KB */
const PAGE_SIZE = 65536;

/** Maximum WASM memory: ~4GB (65536 pages of 64KB each) */
const MAX_PAGES = 65536;

export class MemoryManager {
  readonly memory: WebAssembly.Memory;
  readonly dimensions: number;
  readonly queryOffset: number;
  readonly dbOffset: number;
  private _vectorCount: number;

  constructor(dimensions: number, initialVectorCount: number = 0) {
    this.dimensions = dimensions;

    // Query buffer: dimensions * 4 bytes, aligned to page boundary
    this.queryOffset = 0;
    const queryBytes = dimensions * 4;
    this.dbOffset = Math.ceil(queryBytes / PAGE_SIZE) * PAGE_SIZE;

    // Calculate initial memory needed
    const dbBytes = initialVectorCount * dimensions * 4;
    const totalBytes = this.dbOffset + dbBytes;
    const initialPages = Math.max(1, Math.ceil(totalBytes / PAGE_SIZE));

    this.memory = new WebAssembly.Memory({ initial: initialPages });
    this._vectorCount = initialVectorCount;
  }

  /** Current number of vectors stored */
  get vectorCount(): number {
    return this._vectorCount;
  }

  /** Byte offset where the scores buffer starts (right after DB) */
  get scoresOffset(): number {
    return this.dbOffset + this._vectorCount * this.dimensions * 4;
  }

  /** Total bytes needed for scores buffer */
  get scoresBytes(): number {
    return this._vectorCount * 4;
  }

  /**
   * Maximum vectors that can be stored given the 4GB WASM memory limit.
   * Accounts for query buffer, DB space, and scores buffer.
   */
  get maxVectors(): number {
    const availableBytes = MAX_PAGES * PAGE_SIZE - this.dbOffset;
    // Each vector needs: dimensions * 4 bytes (DB) + 4 bytes (score)
    const bytesPerVector = this.dimensions * 4 + 4;
    return Math.floor(availableBytes / bytesPerVector);
  }

  /**
   * Ensures memory is large enough for the current DB + scores buffer.
   * Calls memory.grow() if needed.
   */
  ensureCapacity(additionalVectors: number): void {
    const newTotal = this._vectorCount + additionalVectors;
    const requiredBytes =
      this.dbOffset + newTotal * this.dimensions * 4 + newTotal * 4; // DB + scores
    const currentBytes = this.memory.buffer.byteLength;

    if (requiredBytes > currentBytes) {
      const pagesNeeded = Math.ceil((requiredBytes - currentBytes) / PAGE_SIZE);
      const currentPages = currentBytes / PAGE_SIZE;
      if (currentPages + pagesNeeded > MAX_PAGES) {
        throw new Error("WASM memory limit exceeded");
      }
      this.memory.grow(pagesNeeded);
    }
  }

  /**
   * Write a query vector into the query buffer region.
   */
  writeQuery(vector: Float32Array): void {
    new Float32Array(this.memory.buffer, this.queryOffset, this.dimensions).set(vector);
  }

  /**
   * Append vectors to the database region.
   * Returns the byte offset where the new vectors were written.
   */
  appendVectors(vectors: Float32Array[]): number {
    const startOffset = this.dbOffset + this._vectorCount * this.dimensions * 4;
    let offset = startOffset;
    for (const vec of vectors) {
      new Float32Array(this.memory.buffer, offset, this.dimensions).set(vec);
      offset += this.dimensions * 4;
    }
    this._vectorCount += vectors.length;
    return startOffset;
  }

  /**
   * Load raw vector bytes directly into the database region.
   * Used for bulk loading from OPFS.
   */
  loadVectorBytes(data: Uint8Array, vectorCount: number): void {
    new Uint8Array(this.memory.buffer, this.dbOffset, data.byteLength).set(data);
    this._vectorCount = vectorCount;
  }

  /**
   * Read the scores buffer as a Float32Array view.
   */
  readScores(): Float32Array {
    return new Float32Array(this.memory.buffer, this.scoresOffset, this._vectorCount);
  }

  /**
   * Read the DB region for a specific vector index.
   */
  readVector(index: number): Float32Array {
    const offset = this.dbOffset + index * this.dimensions * 4;
    return new Float32Array(this.memory.buffer, offset, this.dimensions);
  }
}
