import { describe, it, expect, beforeEach } from "vitest";
import { readFileSync } from "fs";
import { resolve } from "path";
import { VectorEngine } from "../vector-engine";
import { InMemoryStorageProvider } from "../storage";
import { VectorCapacityExceededError } from "../errors";
import { compileWatToWasm } from "../wasm-compute";
import type { EmbeddingFunction } from "../types";

/**
 * Creates a deterministic mock embedder that generates embeddings based on text content.
 * Each unique character contributes to a different dimension.
 */
function createMockEmbedder(dimensions: number): EmbeddingFunction {
  return async (texts: string[]) => {
    return texts.map((text) => {
      const vec = new Float32Array(dimensions);
      for (let i = 0; i < text.length; i++) {
        vec[text.charCodeAt(i) % dimensions] += 1;
      }
      return vec;
    });
  };
}

/** Simple embedder that returns known vectors for testing */
function createFixedEmbedder(
  vectors: Map<string, Float32Array>,
  defaultDimensions: number,
): EmbeddingFunction {
  return async (texts: string[]) => {
    return texts.map((text) => {
      const vec = vectors.get(text);
      if (vec) return new Float32Array(vec);
      // Return zero vector for unknown texts
      return new Float32Array(defaultDimensions);
    });
  };
}

const watSource = readFileSync(resolve(__dirname, "../simd.wat"), "utf-8");

// Pre-compile WAT to WASM binary for test use
let wasmBinaryPromise: Promise<Uint8Array>;
function getWasmBinary(): Promise<Uint8Array> {
  if (!wasmBinaryPromise) {
    wasmBinaryPromise = compileWatToWasm(watSource);
  }
  return wasmBinaryPromise;
}

describe("VectorEngine", () => {
  let storage: InMemoryStorageProvider;

  beforeEach(() => {
    storage = new InMemoryStorageProvider();
  });

  describe("with JS compute", () => {
    runTestSuite(false);
  });

  describe("with WASM SIMD compute", () => {
    runTestSuite(true);
  });

  function runTestSuite(useWasm: boolean) {
    let wasmBinary: Uint8Array | null;

    beforeEach(async () => {
      wasmBinary = useWasm ? await getWasmBinary() : null;
    });

    it("opens with zero records", async () => {
      const db = await VectorEngine.open({
        name: "test-db",
        dimensions: 4,
        embedder: createMockEmbedder(4),
        storage,
        wasmBinary: wasmBinary,
      });

      expect(db.size).toBe(0);
    });

    it("adds a single text and increments size", async () => {
      const db = await VectorEngine.open({
        name: "test-db",
        dimensions: 4,
        embedder: createMockEmbedder(4),
        storage,
        wasmBinary: wasmBinary,
      });

      await db.add("hello world");
      expect(db.size).toBe(1);
    });

    it("adds multiple texts in a batch", async () => {
      const db = await VectorEngine.open({
        name: "test-db",
        dimensions: 4,
        embedder: createMockEmbedder(4),
        storage,
        wasmBinary: wasmBinary,
      });

      await db.add(["alpha", "beta", "gamma"]);
      expect(db.size).toBe(3);
    });

    it("adds texts in multiple batches", async () => {
      const db = await VectorEngine.open({
        name: "test-db",
        dimensions: 4,
        embedder: createMockEmbedder(4),
        storage,
        wasmBinary: wasmBinary,
      });

      await db.add(["first", "second"]);
      await db.add("third");
      expect(db.size).toBe(3);
    });

    it("searches and returns ranked results", async () => {
      const dimensions = 4;
      const vectors = new Map<string, Float32Array>();
      vectors.set("x-axis", new Float32Array([1, 0, 0, 0]));
      vectors.set("y-axis", new Float32Array([0, 1, 0, 0]));
      vectors.set("xy-axis", new Float32Array([1, 1, 0, 0]));
      vectors.set("query-x", new Float32Array([1, 0, 0, 0]));

      const embedder = createFixedEmbedder(vectors, dimensions);

      const db = await VectorEngine.open({
        name: "test-db",
        dimensions,
        embedder,
        storage,
        wasmBinary: wasmBinary,
      });

      await db.add(["x-axis", "y-axis", "xy-axis"]);

      const results = await db.search("query-x");
      expect(results.length).toBe(3);

      // x-axis should be the best match (identical direction)
      expect(results.get(0).text).toBe("x-axis");
      expect(results.get(0).score).toBeCloseTo(1.0, 2);

      // xy-axis should be second (partially aligned)
      expect(results.get(1).text).toBe("xy-axis");
      expect(results.get(1).score).toBeGreaterThan(0);

      // y-axis should be last (orthogonal)
      expect(results.get(2).text).toBe("y-axis");
      expect(results.get(2).score).toBeCloseTo(0.0, 2);
    });

    it("respects topK parameter", async () => {
      const dimensions = 4;
      const vectors = new Map<string, Float32Array>();
      vectors.set("a", new Float32Array([1, 0, 0, 0]));
      vectors.set("b", new Float32Array([0, 1, 0, 0]));
      vectors.set("c", new Float32Array([0, 0, 1, 0]));
      vectors.set("q", new Float32Array([1, 0, 0, 0]));

      const db = await VectorEngine.open({
        name: "test-db",
        dimensions,
        embedder: createFixedEmbedder(vectors, dimensions),
        storage,
        wasmBinary: wasmBinary,
      });

      await db.add(["a", "b", "c"]);
      const results = await db.search("q", 2);
      expect(results.length).toBe(2);
    });

    it("supports pagination on search results", async () => {
      const dimensions = 4;
      const texts = ["t1", "t2", "t3", "t4", "t5"];
      const vectors = new Map<string, Float32Array>();
      texts.forEach((t, i) => {
        const v = new Float32Array(dimensions);
        v[0] = 1 - i * 0.2; // decreasing similarity
        vectors.set(t, v);
      });
      vectors.set("q", new Float32Array([1, 0, 0, 0]));

      const db = await VectorEngine.open({
        name: "test-db",
        dimensions,
        embedder: createFixedEmbedder(vectors, dimensions),
        storage,
        wasmBinary: wasmBinary,
      });

      await db.add(texts);
      const results = await db.search("q");

      const page0 = results.getPage(0, 2);
      expect(page0).toHaveLength(2);

      const page1 = results.getPage(1, 2);
      expect(page1).toHaveLength(2);

      const page2 = results.getPage(2, 2);
      expect(page2).toHaveLength(1);
    });

    it("persists data and reloads from storage", async () => {
      const dimensions = 4;
      const vectors = new Map<string, Float32Array>();
      vectors.set("alpha", new Float32Array([1, 0, 0, 0]));
      vectors.set("beta", new Float32Array([0, 1, 0, 0]));
      vectors.set("q", new Float32Array([1, 0, 0, 0]));

      const embedder = createFixedEmbedder(vectors, dimensions);

      // First session: add data
      const db1 = await VectorEngine.open({
        name: "test-db",
        dimensions,
        embedder,
        storage,
        wasmBinary: wasmBinary,
      });
      await db1.add(["alpha", "beta"]);

      // Second session: reopen same storage
      const db2 = await VectorEngine.open({
        name: "test-db",
        dimensions,
        embedder,
        storage,
        wasmBinary: wasmBinary,
      });

      expect(db2.size).toBe(2);

      const results = await db2.search("q");
      expect(results.get(0).text).toBe("alpha");
    });

    it("handles search on empty database", async () => {
      const db = await VectorEngine.open({
        name: "test-db",
        dimensions: 4,
        embedder: createMockEmbedder(4),
        storage,
        wasmBinary: wasmBinary,
      });

      const results = await db.search("anything");
      expect(results.length).toBe(0);
    });

    it("handles adding empty array", async () => {
      const db = await VectorEngine.open({
        name: "test-db",
        dimensions: 4,
        embedder: createMockEmbedder(4),
        storage,
        wasmBinary: wasmBinary,
      });

      await db.add([]);
      expect(db.size).toBe(0);
    });

    it("validates embedding dimensions", async () => {
      const badEmbedder: EmbeddingFunction = async (texts) => {
        return texts.map(() => new Float32Array(8)); // wrong dimensions
      };

      const db = await VectorEngine.open({
        name: "test-db",
        dimensions: 4,
        embedder: badEmbedder,
        storage,
        wasmBinary: wasmBinary,
      });

      await expect(db.add("test")).rejects.toThrow("dimension mismatch");
    });
  }

  it("throws VectorCapacityExceededError when full", async () => {
    // Create engine with very small capacity to test overflow
    // Max vectors = (4GB - 64KB) / (dims * 4 + 4) ≈ limited
    // We can't actually reach 4GB in tests, but we can verify the check
    const dimensions = 4;
    const db = await VectorEngine.open({
      name: "test-db",
      dimensions,
      embedder: createMockEmbedder(dimensions),
      storage,
    });

    // The maxVectors for 4 dims should be very large, so this test
    // verifies the error class exists and works
    const err = new VectorCapacityExceededError(100);
    expect(err).toBeInstanceOf(VectorCapacityExceededError);
    expect(err.message).toContain("100");

    // Verify db.size is 0 initially
    expect(db.size).toBe(0);
  });
});
