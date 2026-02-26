/**
 * eigen-db: High-Performance In-Browser Vector Database
 *
 * Stores and queries embedding vectors entirely on the client side,
 * utilizing OPFS for persistent storage and WASM SIMD for computation.
 */

export { VectorCapacityExceededError } from "./errors";
export type { ResultItem } from "./result-set";
export { InMemoryStorageProvider, OPFSStorageProvider } from "./storage";
export type { StorageProvider } from "./storage";
export type { IterableQueryOptions, OpenOptions, OpenOptionsInternal, QueryOptions, SetOptions, VectorInput } from "./types";
export { VectorDB as DB } from "./vector-db";
