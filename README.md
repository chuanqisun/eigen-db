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

const results = db.query(queryVector, { topK: 10 });

for (let i = 0; i < results.length; i++) {
  const { key, score } = results.get(i);
  console.log(i, key, score);
}

// Or paginate for UI rendering:
const page0 = results.getPage(0, 10);
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
export { ResultSet };
export type { ResultItem };
export { VectorCapacityExceededError };
export type { OpenOptions, OpenOptionsInternal, SetOptions, QueryOptions, VectorInput };
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
- `query(value: VectorInput, options?: QueryOptions): ResultSet`
  - Returns similarity-ranked results.
  - Throws on dimension mismatch.
- `flush(): Promise<void>`
  - Persists in-memory state to storage.
- `close(): Promise<void>`
  - Flushes and closes the instance.
  - Subsequent operations throw.
- `clear(): Promise<void>`
  - Clears in-memory state and destroys storage for this DB.

### `ResultSet`

Represents a lazily resolved, score-sorted search result collection.

#### Properties

- `length: number` — number of results available (bounded by `topK`)

#### Methods

- `get(rank: number): ResultItem`
  - Returns the item at rank (`0` is best match).
  - Throws `RangeError` when out of bounds.
- `getPage(page: number, pageSize: number): ResultItem[]`
  - Convenience pagination helper.

#### Static

- `fromScores(scores, resolveKey, topK): ResultSet`
  - Constructs a sorted lazy result set from raw scores.

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
- Querying an empty database returns a `ResultSet` with `length === 0`.
- `flush()` writes deduplicated state, and reopen preserves key-to-slot mapping.
