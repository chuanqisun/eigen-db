/**
 * web-vector-base: In-Browser Vector Compute Engine
 *
 * High-performance vector search entirely on the client side,
 * utilizing OPFS for persistent storage and WASM SIMD for computation.
 */

export { VectorDB } from "./vector-db";
export { ResultSet } from "./result-set";
export type { ResultItem } from "./result-set";
export { VectorCapacityExceededError } from "./errors";
export type { OpenOptions, OpenOptionsInternal, SetOptions, QueryOptions } from "./types";
export { InMemoryStorageProvider, OPFSStorageProvider } from "./storage";
export type { StorageProvider } from "./storage";
