/**
 * web-vector-base: In-Browser Vector Compute Engine
 *
 * High-performance vector search entirely on the client side,
 * utilizing OPFS for persistent storage and WASM SIMD for computation.
 */

export { VectorCapacityExceededError } from "./errors";
export { ResultSet } from "./result-set";
export type { ResultItem } from "./result-set";
export { InMemoryStorageProvider, OPFSStorageProvider } from "./storage";
export type { StorageProvider } from "./storage";
export type { OpenOptions, OpenOptionsInternal, QueryOptions, SetOptions, VectorInput } from "./types";
export { VectorDB as DB } from "./vector-db";
