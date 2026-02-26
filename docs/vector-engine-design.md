Here is a design for your in-browser vector compute engine.

### Part 1: Design Principles & Architecture

To meet your requirements, the API is designed around four core principles:

1. **Inversion of Control for Embeddings:** The engine knows nothing about _how_ text becomes a vector. It only requires a contract: "Give me an array of strings, I will give you a Promise of a `Float32Array`." This allows users to plug in OpenAI, HuggingFace (via API), or even local WebGPU models (like Transformers.js).
2. **Batch-First Design:** While the user thinks in single texts, network-based embedding APIs are slow if called one by one. The API accepts both single strings and arrays, but internally optimizes for batching.
3. **Flat Binary Storage (OPFS):** To achieve near-instant initialization, the OPFS storage is split conceptually into two files (or two segments of one file):
   - **The Vector Block:** A pure, contiguous binary blob of `Float32Array` data. On system init, this is loaded directly into memory via `await file.arrayBuffer()`. No parsing required.
   - **The Lexicon (Keys):** A binary-encoded list of strings (length-prefixed UTF-8). Because it's append-only, we just stream new bytes to the end of both files.
4. **Immutability & Append-Only:** By removing `update` and `delete`, we eliminate the need for complex memory management, fragmentation handling, or re-indexing. The index is just an ever-growing array.

---

### Part 2: The API Definition

Here is the TypeScript definition of the API. It is designed to be highly readable and self-documenting.

```typescript
/**
 * The contract for the external embedding provider.
 * Takes an array of texts and returns a flat array of Float32Arrays.
 */
export type EmbeddingFunction = (texts: string[]) => Promise<Float32Array[]>;

export interface EngineConfig {
  /** The name of the database folder in OPFS */
  name: string;
  /** The dimension size of the vectors (e.g., 1536 for OpenAI) */
  dimensions: number;
  /** The user-provided function to convert text to vectors */
  embedder: EmbeddingFunction;
}

export interface SearchResult {
  text: string;
  score: number; // Cosine similarity score (1.0 is perfect match)
}

export class VectorEngine {
  /**
   * Initializes the engine, loading existing binary data from OPFS into memory.
   * If the DB doesn't exist, it creates the necessary OPFS files.
   */
  static async open(config: EngineConfig): Promise<VectorEngine>;

  /**
   * Appends new text(s) to the database.
   * Automatically fetches embeddings and streams them to the OPFS binary files.
   * Skips texts that already exist in the database.
   */
  async add(text: string | string[]): Promise<void>;

  /**
   * Searches the database for the closest matches.
   * If the query text is not in the DB, it will call the embedder first.
   *
   * @param query The text to search for
   * @param topK The maximum number of results to return (defaults to 10)
   */
  async search(query: string, topK?: number): Promise<SearchResult[]>;

  /**
   * Returns the total number of items currently in the database.
   */
  get size(): number;
}
```

---

### Part 3: Usage Examples

Here is what it feels like for a developer to use this API in their browser application.

#### 1. Initialization & Plugging in the Embedder

The user sets up the engine and provides their preferred embedding service (e.g., OpenAI). Notice how the user is forced to provide the `dimensions` upfront—this is the secret to making the binary OPFS storage lightning fast, as we know exactly how many bytes each record takes.

```typescript
import { VectorEngine } from "./vector-engine";

// 1. Define the external embedding function
const openAIEmbedder = async (texts: string[]) => {
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer YOUR_API_KEY`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      input: texts,
      model: "text-embedding-3-small",
    }),
  });

  const data = await response.json();
  // Convert standard arrays to Float32Arrays for binary efficiency
  return data.data.map((item: any) => new Float32Array(item.embedding));
};

// 2. Open the database (loads instantly from OPFS if it exists)
const db = await VectorEngine.open({
  name: "my-knowledge-base",
  dimensions: 1536, // OpenAI text-embedding-3-small dimensions
  embedder: openAIEmbedder,
});

console.log(`Engine ready. Loaded ${db.size} records from OPFS.`);
```

#### 2. Appending Data

Because the engine is append-only, the user just throws text at it. The engine handles the batching, calls the embedder, and appends the raw bytes to the OPFS files.

```typescript
// Add a single item
await db.add("The quick brown fox jumps over the lazy dog.");

// Add multiple items (highly recommended to save network overhead)
await db.add([
  "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
  "E=mc^2 is the mass-energy equivalence formula.",
  "Water boils at 100 degrees Celsius at sea level.",
]);

console.log(`New size: ${db.size}`);
```

#### 3. Querying the Engine

The user queries using plain text. They don't need to know if the word exists in the DB or not. The engine abstracts the math and the fallback logic.

```typescript
// Querying with a word that MIGHT NOT be in the database
// The engine will embed "physics", calculate cosine similarity against all records,
// and return the top K results.
const results = await db.search("physics", 2);

console.table(results);
/*
  Output:
  | (index) | text                                             | score      |
  |---------|--------------------------------------------------|------------|
  | 0       | "E=mc^2 is the mass-energy equivalence formula." | 0.89231    |
  | 1       | "Water boils at 100 degrees Celsius..."          | 0.71024    |
*/

// Querying with a word that IS in the database
// The engine recognizes "Photosynthesis...", skips the network call,
// uses the cached vector, and performs the search.
const exactResults = await db.search("Photosynthesis is the process used by plants to convert light energy into chemical energy.", 1);

console.log(exactResults[0].score); // 1.000000 (Perfect match)
```

### Why this design works well for the Browser:

1. **Zero-Copy Reads:** Because you enforce `Float32Array` at the boundary, when the app starts, you can read the OPFS vector file directly into a WebAssembly memory buffer or a WebGL/WebGPU buffer without parsing JSON or iterating through arrays.
2. **Extensibility:** If you later want to add metadata (like tags or URLs), you can easily extend the `add` method to accept `{ text: string, metadata: any }` and store the metadata in a separate OPFS JSON/Binary file, without breaking the core vector math pipeline.
3. **Memory Efficiency:** By keeping the vectors in a single flat `Float32Array` in memory, the garbage collector isn't overwhelmed by thousands of tiny array objects.
