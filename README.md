# Eigen DB

High-performance vector database for the web.

`eigen-db` stores and queries embedding vectors in-browser, using:

- OPFS (Origin Private File System) for persistence
- WASM SIMD for fast compute when available
- JavaScript fallback when WASM SIMD is unavailable

## Install

```bash
npm install eigen-db
```

## Guide: Set up and query

### 1) Open a database

```ts
import { DB } from "eigen-db";

const db = await DB.open({
  name: "my-index", // optional, defaults to "default"
  dimensions: 1536, // required
  normalize: true, // optional, defaults to true
});
```

### 2) Insert vectors

```ts
db.set("doc:1", embedding1);
db.set("doc:2", embedding2);

db.setMany([
  ["doc:3", embedding3],
  ["doc:4", embedding4],
]);
```

Notes:

- Each vector must be a `number[]` (or `Float32Array`) with exactly `dimensions` elements.
- Duplicate keys use last-write-wins semantics.

### 3) Query nearest vectors

```ts
const queryVector = embeddingQuery;

// Returns a plain array of { key, score } sorted by similarity
const results = db.query(queryVector, { topK: 10 });

for (const { key, score } of results) {
  console.log(key, score);
}
```

For lazy iteration (useful for pagination or early stopping):

```ts
const results = db.query(queryVector, { topK: 100, iterable: true });

// Iterate and break early — keys are resolved on demand
for (const { key, score } of results) {
  if (score < 0.5) break;
  console.log(key, score);
}

// Or spread into an array when you need all results
const all = [...results];
```

### 4) Persist and lifecycle

```ts
await db.flush(); // persist current state
await db.close(); // flush + mark closed
```

To delete all vectors and storage:

```ts
await db.clear();
```

## Full API Reference

## Exports

```ts
export { DB };
export type { ResultItem };
export { VectorCapacityExceededError };
export type { OpenOptions, OpenOptionsInternal, SetOptions, QueryOptions, IterableQueryOptions, VectorInput };
export { InMemoryStorageProvider, OPFSStorageProvider };
export type { StorageProvider };
```

### `DB`

#### `DB.open(options)`

```ts
static open(options: OpenOptions): Promise<DB>
static open(options: OpenOptionsInternal): Promise<DB>
```

Opens (or creates) a database instance and loads persisted data.

#### Properties

- `size: number` — current number of key-vector pairs

#### Methods

- `set(key: string, value: VectorInput, options?: SetOptions): void`
  - Inserts or overwrites a vector.
  - Throws on dimension mismatch.
- `get(key: string): number[] | undefined`
  - Returns a copy of the stored vector.
- `setMany(entries: [string, VectorInput][]): void`
  - Batch insert/update.
- `getMany(keys: string[]): (number[] | undefined)[]`
  - Batch lookup.
- `query(value: VectorInput, options?: QueryOptions): ResultItem[]`
  - Returns similarity-ranked results as a plain array.
  - Throws on dimension mismatch.
- `query(value: VectorInput, options: IterableQueryOptions): Iterable<ResultItem>`
  - With `{ iterable: true }`, returns a lazy iterable. Keys are resolved
    only as each item is consumed, enabling early stopping and pagination.
  - Throws on dimension mismatch.
- `flush(): Promise<void>`
  - Persists in-memory state to storage.
- `close(): Promise<void>`
  - Flushes and closes the instance.
  - Subsequent operations throw.
- `clear(): Promise<void>`
  - Clears in-memory state and destroys storage for this DB.

### `ResultItem`

```ts
interface ResultItem {
  key: string;
  score: number;
}
```

### Option types

#### `OpenOptions`

```ts
interface OpenOptions {
  name?: string; // OPFS directory name, default: "default"
  dimensions: number; // vector size
  normalize?: boolean; // default: true
}
```

#### `OpenOptionsInternal`

Advanced/testing override options.

```ts
interface OpenOptionsInternal extends OpenOptions {
  storage?: StorageProvider;
  wasmBinary?: Uint8Array | null;
}
```

- `storage`: provide custom storage implementation (for example, tests)
- `wasmBinary`:
  - `Uint8Array`: use provided precompiled WASM
  - `null`: force JavaScript-only compute
  - omitted: use embedded SIMD binary

#### `SetOptions`

```ts
interface SetOptions {
  normalize?: boolean;
}
```

#### `QueryOptions`

```ts
interface QueryOptions {
  topK?: number; // default: all vectors
  normalize?: boolean;
}
```

#### `IterableQueryOptions`

```ts
interface IterableQueryOptions extends QueryOptions {
  iterable: true; // returns Iterable<ResultItem> instead of ResultItem[]
}
```

### Storage

#### `StorageProvider`

```ts
interface StorageProvider {
  readAll(fileName: string): Promise<Uint8Array>;
  append(fileName: string, data: Uint8Array): Promise<void>;
  write(fileName: string, data: Uint8Array): Promise<void>;
  destroy(): Promise<void>;
}
```

#### `OPFSStorageProvider`

Browser persistence provider backed by OPFS.

```ts
new OPFSStorageProvider(dirName: string)
```

#### `InMemoryStorageProvider`

Non-persistent in-memory provider, useful for tests or ephemeral sessions.

```ts
new InMemoryStorageProvider();
```

### Errors

#### `VectorCapacityExceededError`

Thrown when memory growth would exceed WASM 32-bit memory limits for the configured dimension size.

## Practical notes

- Similarity is dot product; with normalization enabled (default), this behaves like cosine similarity.
- Querying an empty database returns an empty array (`[]`).
- `flush()` writes deduplicated state, and reopen preserves key-to-slot mapping.
