/**
 * Options for opening a VectorDB instance.
 */
export interface OpenOptions {
  /** Vector dimensions (e.g., 1536 for OpenAI text-embedding-3-small) */
  dimensions: number;
  /** Whether to normalize vectors on set/query (default: true). */
  normalize?: boolean;
  /** Storage provider for persistence. Defaults to InMemoryStorageProvider. Use OPFSStorageProvider for browser persistence. */
  storage?: import("./storage").StorageProvider;
}

/**
 * Extended options including internal overrides for testing/advanced use.
 */
export interface OpenOptionsInternal extends OpenOptions {
  /** Pre-compiled WASM binary. If not provided, uses the embedded SIMD binary. Set to null to force JS-only mode. */
  wasmBinary?: Uint8Array | null;
}

/**
 * Accepted vector input types. Users can pass a plain number[] or a Float32Array.
 * Internally converted to Float32Array for WASM operations.
 */
export type VectorInput = number[] | Float32Array;

/**
 * Options for set/setMany operations.
 */
export interface SetOptions {
  /** Override normalization for this call. */
  normalize?: boolean;
}

/**
 * Options for query operations.
 * Returns a plain ResultItem[] array by default.
 */
export interface QueryOptions {
  /** Maximum number of results to return. Defaults to Infinity (all results). */
  limit?: number;
  /** Result ordering. "ascend" sorts by ascending similarity, "descend" sorts by descending similarity. Defaults to "descend". */
  order?: "ascend" | "descend";
  /** Minimum similarity threshold (inclusive). Results with similarity < minSimilarity are excluded. */
  minSimilarity?: number;
  /** Maximum similarity threshold (inclusive). Results with similarity > maxSimilarity are excluded. */
  maxSimilarity?: number;
  /** Override normalization for this call. */
  normalize?: boolean;
  /** When true, returns an Iterable<ResultItem> instead of ResultItem[]. */
  iterable?: boolean;
}
