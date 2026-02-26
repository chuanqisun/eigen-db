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

// Returns a plain array of { key, similarity } sorted by descending similarity
const results = db.query(queryVector, { topK: 10 });

for (const { key, similarity } of results) {
  console.log(key, similarity);
}
```

For lazy iteration (useful for pagination or early stopping):

```ts
const results = db.query(queryVector, { topK: 100, iterable: true });

// Iterate and break early — keys are resolved on demand
for (const { key, similarity } of results) {
  if (similarity < 0.5) break;
  console.log(key, similarity);
}

// Or spread into an array when you need all results
const all = [...results];
```

Use `minSimilarity` to automatically cut off results below a threshold:

```ts
// Only return results with similarity ≥ 0.7 (inclusive)
const results = db.query(queryVector, { minSimilarity: 0.7 });

// Works with iterable mode too — iteration stops early at the threshold
const results = db.query(queryVector, { minSimilarity: 0.7, iterable: true });
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

### 5) Export and import

Export the entire database as a streaming binary file:

```ts
const stream = await db.export(); // ReadableStream<Uint8Array>

// In a browser — download as a file
const response = new Response(stream);
const blob = await response.blob();
const url = URL.createObjectURL(blob);
const a = document.createElement("a");
a.href = url;
a.download = "database.bin";
a.click();
```

Import from a stream, replacing all existing data:

```ts
// From a File (e.g., <input type="file">)
await db.import(file.stream());

// From a fetch response
const res = await fetch("/path/to/database.bin");
await db.import(res.body!);
```

Notes:

- `import()` replaces all existing data in the target database.
- A dimension check is performed on import: the stream must contain data exported from a database with the same `dimensions` setting.
- Both methods use the Web Streams API to avoid large heap allocations — vectors are streamed in 64KB chunks.

## Similarity metric

Similarity is the dot product of the query and stored vectors.

- **With normalization enabled** (the default): vectors are L2-normalized before storage and query, so the dot product equals cosine similarity. Similarity ranges from **1** (identical) to **-1** (opposite), with **0** indicating orthogonal vectors.
- **With normalization disabled** (`normalize: false`): the dot product is computed on raw vectors. The range depends on the magnitude of your vectors. Use this mode when your vectors are already normalized or when you want raw dot-product semantics.

**When to normalize:**

| Scenario | Normalize? | Notes |
| --- | --- | --- |
| Using embeddings from OpenAI, Cohere, etc. | `true` (default) | Embeddings may not be unit-length; normalization ensures cosine similarity. |
| Vectors are already unit-length | Either | Setting `false` avoids redundant work. |
| You need raw dot-product semantics | `false` | Similarity will be the raw dot product; range depends on vector magnitudes. |

## Full API Reference

## Exports

```ts
export { DB };
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
- `query(value: VectorInput, options?: QueryOptions): ResultItem[]`
  - Returns results sorted by descending similarity as a plain array.
  - Throws on dimension mismatch.
- `query(value: VectorInput, options: QueryOptions & { iterable: true }): Iterable<ResultItem>`
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
- `export(): Promise<ReadableStream<Uint8Array>>`
  - Exports the entire database as a streaming binary. Vectors are streamed in 64KB chunks.
- `import(stream: ReadableStream<Uint8Array>): Promise<void>`
  - Imports data from a stream, replacing all existing data.
  - Throws on dimension mismatch between the stream data and the database.

### `ResultItem`

```ts
interface ResultItem {
  key: string;
  similarity: number;
}
```

- `similarity` — The dot product of query and stored vectors. With normalization (default), this is cosine similarity: 1 = identical, -1 = opposite.

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
  topK?: number; // default: Infinity (all results)
  minSimilarity?: number; // inclusive lower bound on similarity; results below this are excluded
  normalize?: boolean;
  iterable?: boolean; // when true, returns Iterable<ResultItem> instead of ResultItem[]
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

## Benchmark results

WASM SIMD vs pure JavaScript performance on 1536-dimensional vectors (OpenAI embedding size), measured with `vitest bench` (Node.js):

| Operation | JS (ops/s) | WASM SIMD (ops/s) | Speedup |
| --- | --- | --- | --- |
| normalize (1536 dims) | 223,117 | 2,226,734 | **~10×** |
| searchAll (100 vectors × 1536 dims) | 3,429 | 77,130 | **~22×** |
| searchAll (1,000 vectors × 1536 dims) | 344 | 8,009 | **~23×** |
| searchAll (10,000 vectors × 1536 dims) | 34 | 398 | **~12×** |

The WASM SIMD layer uses 2-vector outer loop unrolling (halving query memory reads) and 4× inner loop unrolling with multiple independent accumulators.

### Running benchmarks

**Node.js** (via vitest):

```bash
npm run bench
```

**Browser**: start the dev server and navigate to the benchmark page:

```bash
npm run dev
# Open http://localhost:5173/bench.html
```

## Practical notes

- Similarity is the dot product of query and stored vectors; with normalization enabled (default), this behaves like cosine similarity (1 = identical, -1 = opposite).
- `topK` defaults to `Infinity`, returning all stored vectors sorted by similarity. Use `minSimilarity` to limit results by proximity.
- Querying an empty database returns an empty array (`[]`).
- `flush()` writes deduplicated state, and reopen preserves key-to-slot mapping.

## Related

- Just need cosine similarity? Try [fast-theta](https://github.com/chuanqisun/fast-theta).
