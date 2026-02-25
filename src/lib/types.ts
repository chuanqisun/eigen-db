/**
 * Options for opening a VectorDB instance.
 */
export interface OpenOptions {
  /** Name of the OPFS directory. Defaults to "default". */
  name?: string;
  /** Vector dimensions (e.g., 1536 for OpenAI text-embedding-3-small) */
  dimensions: number;
  /** Whether to normalize vectors on set/query (default: true). */
  normalize?: boolean;
}

/**
 * Extended options including internal overrides for testing/advanced use.
 */
export interface OpenOptionsInternal extends OpenOptions {
  /** Override the storage provider (default: OPFS). Useful for testing. */
  storage?: import("./storage").StorageProvider;
  /** Pre-compiled WASM binary. If not provided, uses the embedded SIMD binary. Set to null to force JS-only mode. */
  wasmBinary?: Uint8Array | null;
}

/**
 * Options for set/setMany operations.
 */
export interface SetOptions {
  /** Override normalization for this call. */
  normalize?: boolean;
}

/**
 * Options for query operations.
 */
export interface QueryOptions {
  /** Maximum number of results to return. Defaults to all. */
  topK?: number;
  /** Override normalization for this call. */
  normalize?: boolean;
}
