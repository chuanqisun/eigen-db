# Implementation Specification: In-Browser Vector Compute Engine

## 1. Overview & Design Principles

This specification outlines the architecture for a high-performance, in-browser vector search engine. It is designed to handle hundreds of thousands of high-dimensional vectors entirely on the client side, utilizing the Origin Private File System (OPFS) for persistent storage and hand-optimized WebAssembly (WASM) SIMD for computation.

The architecture is driven by five core principles:

1. **Inversion of Control (Embeddings):** The engine is agnostic to the embedding model. Users provide a function that maps `string[] -> Promise<Float32Array[]>`.
2. **Zero-Copy Storage:** Vector data is stored as a contiguous binary blob in OPFS. On initialization, it is read directly into `WebAssembly.Memory` without passing through standard JavaScript arrays.
3. **Algorithmic Simplification (Manual WASM):** By pre-normalizing vectors upon insertion, Cosine Similarity is reduced to a simple Dot Product. The math is executed via hand-written WebAssembly Text (WAT) using 128-bit SIMD instructions, bypassing heavy Rust/C++ toolchains.
4. **Append-Only Immutability:** Vectors cannot be updated or deleted. This eliminates memory fragmentation, complex indexing, and garbage collection overhead.
5. **Lazy Evaluation:** To support massive queries (e.g., returning the entire dataset sorted by relevance), results are returned as a lazy view over TypedArrays, preventing JavaScript heap crashes.

---

## 2. System Architecture

The system is divided into three primary layers:

### A. Storage Layer (OPFS)

Data is split into two append-only files:

- **`vectors.bin`**: A raw, flat binary file of `Float32Array` data.
- **`lexicon.bin`**: A length-prefixed UTF-8 encoded file containing the original text strings.

### B. Compute Layer (WASM SIMD)

A manually crafted `.wat` module containing minimal logic:

- `normalize(ptr, dimensions)`: Normalizes a vector in-place before it is saved.
- `search_all(query_ptr, db_ptr, scores_ptr, db_size, dimensions)`: A highly optimized SIMD loop that calculates the dot product of the query against the entire database, writing `f32` scores directly to memory.

### C. Memory Layer (Dynamic Shared Memory)

JavaScript instantiates a `WebAssembly.Memory` object and imports it into the WASM module. JS acts as the memory manager, calculating required bytes and calling `memory.grow()` before operations.

**Memory Layout:**

```text
[ 0x00000 ] -> Query Vector Buffer (Fixed, e.g., 64KB)
[ 0x10000 ] -> Vector Database (Grows dynamically as OPFS loads/appends)
[ Dynamic ] -> Scores Buffer (Temporarily mapped at the end of the DB during search)
```

---

## 3. High-Level API Specification

```typescript
/**
 * Contract for external embedding providers (OpenAI, HuggingFace, local WebGPU, etc.)
 */
export type EmbeddingFunction = (texts: string[]) => Promise<Float32Array[]>;

export interface EngineConfig {
  /** Name of the OPFS directory */
  name: string;
  /** Vector dimensions (e.g., 1536 for OpenAI text-embedding-3-small) */
  dimensions: number;
  /** User-provided embedding function */
  embedder: EmbeddingFunction;
}

/**
 * Thrown when the database exceeds the 4GB WebAssembly 32-bit memory limit,
 * or the browser's available RAM.
 */
export class VectorCapacityExceededError extends Error {
  constructor(maxVectors: number) {
    super(`Capacity exceeded. Max vectors for this dimension size is ~${maxVectors}.`);
  }
}

/**
 * LAZY RESULT SET
 * Holds pointers to sorted TypedArrays. Prevents JS heap overflow when K is massive.
 * Strings are only instantiated from the Lexicon when explicitly requested.
 */
export class ResultSet {
  /** Total number of results matched (up to K) */
  readonly length: number;

  /** Fetch a single result by its rank (0 is best match) */
  get(rank: number): { text: string; score: number };

  /** Helper for UI pagination. Instantiates strings only for the requested page. */
  getPage(page: number, pageSize: number): { text: string; score: number }[];
}

/**
 * THE CORE ENGINE
 */
export class VectorEngine {
  /**
   * Initializes the engine.
   * Reads OPFS `vectors.bin` directly into WASM memory (Zero-copy).
   */
  static async open(config: EngineConfig): Promise<VectorEngine>;

  /**
   * Fetches embeddings, normalizes them via WASM, appends to OPFS,
   * and updates the WASM memory buffer.
   */
  async add(text: string | string[]): Promise<void>;

  /**
   * Embeds the query, executes WASM SIMD dot-products across the entire DB,
   * sorts the results via JS TypedArrays, and returns a lazy ResultSet.
   */
  async search(query: string, topK?: number): Promise<ResultSet>;

  /** Total records in the database */
  get size(): number;
}
```

---

## 4. Ideal API Usage

The following example demonstrates how a developer integrates the engine into a web application, highlighting the batching and lazy-loading features.

### Initialization & Setup

```typescript
import { VectorEngine } from "./vector-engine";

// 1. Define the embedder (e.g., OpenAI)
const openAIEmbedder = async (texts: string[]) => {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: { Authorization: `Bearer ${API_KEY}`, "Content-Type": "application/json" },
    body: JSON.stringify({ input: texts, model: "text-embedding-3-small" }),
  });
  const json = await res.json();
  return json.data.map((d: any) => new Float32Array(d.embedding));
};

// 2. Open the database (Loads instantly from OPFS)
const db = await VectorEngine.open({
  name: "browser-knowledge-base",
  dimensions: 1536,
  embedder: openAIEmbedder,
});

console.log(`Engine ready. Loaded ${db.size} records.`);
```

### Appending Data

```typescript
try {
  // Batching is highly recommended to minimize network round-trips
  await db.add([
    "WebAssembly SIMD provides near-native performance in the browser.",
    "Origin Private File System allows high-performance local storage.",
    "Cosine similarity measures the angle between two vectors.",
  ]);
} catch (e) {
  if (e instanceof VectorCapacityExceededError) {
    console.error("Database is full! Cannot add more records.");
  }
}
```

### Querying & Lazy Pagination

Even if the database contains 500,000 records and the user requests `topK = 500000`, the search executes in milliseconds and consumes almost zero extra memory.

```typescript
// Search the entire database (topK = db.size)
const results = await db.search("How fast is WASM?", db.size);

console.log(`Found ${results.length} total matches.`);

// UI Pagination: Only instantiate strings for the first 10 results
const page1 = results.getPage(0, 10);

console.table(page1);
/*
  | (index) | text                                              | score    |
  |---------|---------------------------------------------------|----------|
  | 0       | "WebAssembly SIMD provides near-native..."        | 0.892    |
  | 1       | "Origin Private File System allows..."            | 0.412    |
  | 2       | "Cosine similarity measures the angle..."         | 0.201    |
*/

// Later, when the user clicks "Next Page" in the UI:
const page2 = results.getPage(1, 10);
```

## 5. Summary of Performance Characteristics

- **Initialization:** $O(1)$ parsing overhead. Disk I/O bound (typically < 50ms for 100MB).
- **Search Compute:** $O(N \times D)$ where $N$ is dataset size and $D$ is dimensions. Executed at SIMD speeds (4 floats per CPU cycle).
- **Search Sort:** $O(N \log N)$ executed via highly optimized V8 TypedArray sorting.
- **Memory Footprint:** Strictly bounded. $N \times D \times 4$ bytes for vectors, plus a negligible $N \times 8$ bytes for search sorting. Max capacity is safely enforced at the ~4GB WASM limit.
