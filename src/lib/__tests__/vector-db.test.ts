import { readFileSync } from "fs";
import { resolve } from "path";
import { beforeEach, describe, expect, it } from "vitest";
import { VectorCapacityExceededError } from "../errors";
import { InMemoryStorageProvider } from "../storage";
import { VectorDB } from "../vector-db";
import { compileWatToWasm } from "./compile-wat-helper";

const watSource = readFileSync(resolve(__dirname, "../simd.wat"), "utf-8");

let wasmBinaryPromise: Promise<Uint8Array>;
function getWasmBinary(): Promise<Uint8Array> {
  if (!wasmBinaryPromise) {
    wasmBinaryPromise = compileWatToWasm(watSource);
  }
  return wasmBinaryPromise;
}

/** Collect a ReadableStream into a single Uint8Array. */
async function streamToBytes(stream: ReadableStream<Uint8Array>): Promise<Uint8Array> {
  const chunks: Uint8Array[] = [];
  const reader = stream.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }
  const totalSize = chunks.reduce((sum, c) => sum + c.byteLength, 0);
  const result = new Uint8Array(totalSize);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return result;
}

/** Create a ReadableStream from a Uint8Array (single chunk). */
function bytesToStream(data: Uint8Array): ReadableStream<Uint8Array> {
  return new ReadableStream({
    start(controller) {
      controller.enqueue(data);
      controller.close();
    },
  });
}

/** Create a ReadableStream from a Uint8Array, delivering it in small chunks. */
function bytesToChunkedStream(data: Uint8Array, chunkSize: number): ReadableStream<Uint8Array> {
  let offset = 0;
  return new ReadableStream({
    pull(controller) {
      if (offset >= data.byteLength) {
        controller.close();
        return;
      }
      const end = Math.min(offset + chunkSize, data.byteLength);
      controller.enqueue(data.subarray(offset, end));
      offset = end;
    },
  });
}

describe("VectorDB", () => {
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

    // --- open ---
    it("opens with zero records", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      expect(db.size).toBe(0);
    });

    it("exposes dimensions property", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      expect(db.dimensions).toBe(4);
    });

    // --- set and get ---
    it("stores and retrieves a vector by key", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("a", [1, 2, 3, 4]);
      expect(db.size).toBe(1);

      const result = db.get("a");
      expect(result).toBeDefined();
      expect(result![0]).toBeCloseTo(1);
      expect(result![1]).toBeCloseTo(2);
      expect(result![2]).toBeCloseTo(3);
      expect(result![3]).toBeCloseTo(4);
    });

    it("last write wins when writing different value to the same key", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("a", [1, 0, 0, 0]);
      db.set("a", [0, 1, 0, 0]);

      expect(db.size).toBe(1); // only one entry
      const result = db.get("a");
      expect(result![0]).toBeCloseTo(0);
      expect(result![1]).toBeCloseTo(1);
    });

    it("last write wins with multiple overwrites", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("key1", [1, 0, 0, 0]);
      db.set("key1", [0, 1, 0, 0]);
      db.set("key1", [0, 0, 1, 0]);

      expect(db.size).toBe(1);
      const result = db.get("key1");
      expect(result![2]).toBeCloseTo(1);
    });

    it("returns undefined for non-existent key", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      expect(db.get("nonexistent")).toBeUndefined();
    });

    it("validates vector dimensions on set", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      expect(() => db.set("a", [1, 2, 3, 4, 5, 6, 7, 8])).toThrow("dimension mismatch");
    });

    // --- setMany and getMany ---
    it("stores and retrieves multiple entries with setMany/getMany", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.setMany([
        ["a", [1, 0, 0, 0]],
        ["b", [0, 1, 0, 0]],
        ["c", [0, 0, 1, 0]],
      ]);

      expect(db.size).toBe(3);

      const results = db.getMany(["a", "b", "c", "d"]);
      expect(results[0]![0]).toBeCloseTo(1);
      expect(results[1]![1]).toBeCloseTo(1);
      expect(results[2]![2]).toBeCloseTo(1);
      expect(results[3]).toBeUndefined();
    });

    it("setMany with duplicate keys in batch uses last value", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.setMany([
        ["x", [1, 0, 0, 0]],
        ["x", [0, 0, 0, 1]],
      ]);

      expect(db.size).toBe(1);
      const result = db.get("x");
      expect(result![3]).toBeCloseTo(1);
    });

    // --- query ---
    it("query returns ranked results by similarity", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("x-axis", [1, 0, 0, 0]);
      db.set("y-axis", [0, 1, 0, 0]);
      db.set("xy-axis", [1, 1, 0, 0]);

      const results = db.query([1, 0, 0, 0]);
      expect(results.length).toBe(3);

      // x-axis should be the best match (identical direction, similarity ≈ 1)
      expect(results[0].key).toBe("x-axis");
      expect(results[0].similarity).toBeCloseTo(1.0, 2);

      // xy-axis should be second (partially aligned)
      expect(results[1].key).toBe("xy-axis");
      expect(results[1].similarity).toBeGreaterThan(0);

      // y-axis should be last (orthogonal, similarity ≈ 0)
      expect(results[2].key).toBe("y-axis");
      expect(results[2].similarity).toBeCloseTo(0.0, 2);
    });

    it("query respects limit option", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("a", [1, 0, 0, 0]);
      db.set("b", [0, 1, 0, 0]);
      db.set("c", [0, 0, 1, 0]);

      const results = db.query([1, 0, 0, 0], { limit: 2 });
      expect(results.length).toBe(2);
    });

    it("query on empty database returns empty array", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      const results = db.query([1, 0, 0, 0]);
      expect(results).toEqual([]);
    });

    it("query validates vector dimensions", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("a", [1, 0, 0, 0]);
      expect(() => db.query([1, 2, 3, 4, 5, 6, 7, 8])).toThrow("dimension mismatch");
    });

    it("query results support iterable mode for pagination", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      for (let i = 0; i < 5; i++) {
        const vec = [0, 0, 0, 0];
        vec[0] = 1 - i * 0.2;
        db.set(`t${i}`, vec);
      }

      const results = db.query([1, 0, 0, 0], { normalize: false, iterable: true });

      // Spread into array
      const all = [...results];
      expect(all).toHaveLength(5);

      // Partial iteration (simulate pagination)
      const page: { key: string; similarity: number }[] = [];
      for (const item of results) {
        page.push(item);
        if (page.length === 2) break;
      }
      expect(page).toHaveLength(2);
    });

    it("query after overwrite uses updated vector", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("point", [1, 0, 0, 0]);
      db.set("other", [0, 1, 0, 0]);

      // Overwrite 'point' to be along y-axis
      db.set("point", [0, 1, 0, 0]);

      const results = db.query([0, 1, 0, 0]);
      // Both 'point' and 'other' are now along y-axis, so both should have similarity ≈ 1
      expect(results[0].similarity).toBeCloseTo(1.0, 2);
      expect(results[1].similarity).toBeCloseTo(1.0, 2);
      expect(db.size).toBe(2);
    });

    it("query respects minSimilarity option", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("x-axis", [1, 0, 0, 0]);
      db.set("y-axis", [0, 1, 0, 0]);
      db.set("xy-axis", [1, 1, 0, 0]);

      // Only return results with similarity ≥ 0.5 from the x-axis query
      const results = db.query([1, 0, 0, 0], { minSimilarity: 0.5 });
      // x-axis: similarity ≈ 1, xy-axis: similarity ≈ 0.71
      // y-axis: similarity ≈ 0 (excluded)
      expect(results.length).toBe(2);
      expect(results[0].key).toBe("x-axis");
      expect(results[1].key).toBe("xy-axis");
    });

    it("query minSimilarity works with iterable mode", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("x-axis", [1, 0, 0, 0]);
      db.set("y-axis", [0, 1, 0, 0]);
      db.set("xy-axis", [1, 1, 0, 0]);

      const results = [...db.query([1, 0, 0, 0], { minSimilarity: 0.5, iterable: true })];
      expect(results.length).toBe(2);
      expect(results[0].key).toBe("x-axis");
      expect(results[1].key).toBe("xy-axis");
    });

    it("query limit defaults to Infinity (returns all results)", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      for (let i = 0; i < 10; i++) {
        const vec = [0, 0, 0, 0];
        vec[i % 4] = 1;
        db.set(`v${i}`, vec);
      }

      // Without limit, all 10 results should be returned
      const results = db.query([1, 0, 0, 0]);
      expect(results.length).toBe(10);
    });

    it("query with order ascend returns least similar first", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("x-axis", [1, 0, 0, 0]);
      db.set("y-axis", [0, 1, 0, 0]);
      db.set("xy-axis", [1, 1, 0, 0]);

      const results = db.query([1, 0, 0, 0], { order: "ascend" });
      expect(results.length).toBe(3);
      // y-axis (similarity ≈ 0) should be first in ascending order
      expect(results[0].key).toBe("y-axis");
      expect(results[0].similarity).toBeCloseTo(0.0, 2);
      // x-axis (similarity ≈ 1) should be last
      expect(results[2].key).toBe("x-axis");
      expect(results[2].similarity).toBeCloseTo(1.0, 2);
    });

    it("query with order ascend and limit returns bottomK", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("x-axis", [1, 0, 0, 0]);
      db.set("y-axis", [0, 1, 0, 0]);
      db.set("xy-axis", [1, 1, 0, 0]);

      const results = db.query([1, 0, 0, 0], { order: "ascend", limit: 1 });
      expect(results.length).toBe(1);
      expect(results[0].key).toBe("y-axis");
    });

    it("query respects maxSimilarity option", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("x-axis", [1, 0, 0, 0]);
      db.set("y-axis", [0, 1, 0, 0]);
      db.set("xy-axis", [1, 1, 0, 0]);

      // Only return results with similarity ≤ 0.8 from the x-axis query
      const results = db.query([1, 0, 0, 0], { maxSimilarity: 0.8 });
      // x-axis: similarity ≈ 1 (excluded), xy-axis: similarity ≈ 0.71, y-axis: similarity ≈ 0
      expect(results.length).toBe(2);
      expect(results[0].key).toBe("xy-axis");
      expect(results[1].key).toBe("y-axis");
    });

    it("query with both minSimilarity and maxSimilarity", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("x-axis", [1, 0, 0, 0]);
      db.set("y-axis", [0, 1, 0, 0]);
      db.set("xy-axis", [1, 1, 0, 0]);

      // Only return results with 0.5 ≤ similarity ≤ 0.8
      const results = db.query([1, 0, 0, 0], { minSimilarity: 0.5, maxSimilarity: 0.8 });
      // xy-axis: similarity ≈ 0.71 (included)
      // x-axis ≈ 1.0 (excluded), y-axis ≈ 0.0 (excluded)
      expect(results.length).toBe(1);
      expect(results[0].key).toBe("xy-axis");
    });

    it("query maxSimilarity works with iterable mode", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("x-axis", [1, 0, 0, 0]);
      db.set("y-axis", [0, 1, 0, 0]);
      db.set("xy-axis", [1, 1, 0, 0]);

      const results = [...db.query([1, 0, 0, 0], { maxSimilarity: 0.8, iterable: true })];
      expect(results.length).toBe(2);
      expect(results[0].key).toBe("xy-axis");
      expect(results[1].key).toBe("y-axis");
    });

    it("query order ascend with iterable mode", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("x-axis", [1, 0, 0, 0]);
      db.set("y-axis", [0, 1, 0, 0]);
      db.set("xy-axis", [1, 1, 0, 0]);

      const results = [...db.query([1, 0, 0, 0], { order: "ascend", iterable: true })];
      expect(results.length).toBe(3);
      expect(results[0].key).toBe("y-axis");
      expect(results[2].key).toBe("x-axis");
    });

    it("query supports full similarity range [-1, 1] with opposite vectors", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("same", [1, 0, 0, 0]); // similarity ≈ 1
      db.set("ortho", [0, 1, 0, 0]); // similarity ≈ 0
      db.set("opposite", [-1, 0, 0, 0]); // similarity ≈ -1

      const results = db.query([1, 0, 0, 0]);
      expect(results.length).toBe(3);
      expect(results[0].key).toBe("same");
      expect(results[0].similarity).toBeCloseTo(1.0, 2);
      expect(results[1].key).toBe("ortho");
      expect(results[1].similarity).toBeCloseTo(0.0, 2);
      expect(results[2].key).toBe("opposite");
      expect(results[2].similarity).toBeCloseTo(-1.0, 2);
    });

    it("query minSimilarity works with negative values", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("same", [1, 0, 0, 0]); // similarity ≈ 1
      db.set("ortho", [0, 1, 0, 0]); // similarity ≈ 0
      db.set("opposite", [-1, 0, 0, 0]); // similarity ≈ -1

      // minSimilarity = -0.5 should include same and ortho, exclude opposite
      const results = db.query([1, 0, 0, 0], { minSimilarity: -0.5 });
      expect(results.length).toBe(2);
      expect(results[0].key).toBe("same");
      expect(results[1].key).toBe("ortho");
    });

    it("query maxSimilarity works with negative values", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("same", [1, 0, 0, 0]); // similarity ≈ 1
      db.set("ortho", [0, 1, 0, 0]); // similarity ≈ 0
      db.set("opposite", [-1, 0, 0, 0]); // similarity ≈ -1

      // maxSimilarity = -0.5 should include only opposite
      const results = db.query([1, 0, 0, 0], { maxSimilarity: -0.5 });
      expect(results.length).toBe(1);
      expect(results[0].key).toBe("opposite");
    });

    it("query ascending order with negative similarities", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("same", [1, 0, 0, 0]);
      db.set("ortho", [0, 1, 0, 0]);
      db.set("opposite", [-1, 0, 0, 0]);

      const results = db.query([1, 0, 0, 0], { order: "ascend" });
      expect(results.length).toBe(3);
      expect(results[0].key).toBe("opposite");
      expect(results[0].similarity).toBeCloseTo(-1.0, 2);
      expect(results[2].key).toBe("same");
      expect(results[2].similarity).toBeCloseTo(1.0, 2);
    });

    // --- flush and persistence ---
    it("flush persists data and reopen loads it", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db1.set("alpha", [1, 0, 0, 0]);
      db1.set("beta", [0, 1, 0, 0]);
      await db1.flush();

      // Reopen same storage
      const db2 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      expect(db2.size).toBe(2);
      expect(db2.get("alpha")![0]).toBeCloseTo(1);
      expect(db2.get("beta")![1]).toBeCloseTo(1);
    });

    it("flush persists overwritten values correctly", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db1.set("key", [1, 0, 0, 0]);
      db1.set("key", [0, 0, 0, 1]);
      await db1.flush();

      const db2 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      expect(db2.size).toBe(1);
      expect(db2.get("key")![3]).toBeCloseTo(1);
    });

    // --- close ---
    it("close flushes and prevents further operations", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("a", [1, 0, 0, 0]);
      await db.close();

      // Should be persisted
      const db2 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });
      expect(db2.size).toBe(1);

      // Original instance should be closed
      expect(() => db.set("b", [0, 0, 0, 0])).toThrow("closed");
    });

    it("close is idempotent", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      await db.close();
      await db.close(); // should not throw
    });

    // --- clear ---
    it("clear removes all data", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("a", [1, 0, 0, 0]);
      db.set("b", [0, 1, 0, 0]);
      expect(db.size).toBe(2);

      await db.clear();
      expect(db.size).toBe(0);
      expect(db.get("a")).toBeUndefined();
    });

    it("clear allows reuse of the database", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("old", [1, 0, 0, 0]);
      await db.clear();

      db.set("new", [0, 1, 0, 0]);
      expect(db.size).toBe(1);
      expect(db.get("new")![1]).toBeCloseTo(1);
    });

    // --- normalization ---
    it("normalizes vectors by default", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("a", [3, 0, 0, 0]);
      const result = db.get("a");
      // Should be normalized to unit length
      expect(result![0]).toBeCloseTo(1.0);
    });

    it("skips normalization when normalize is false", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("a", [3, 0, 0, 0]);
      const result = db.get("a");
      expect(result![0]).toBeCloseTo(3.0);
    });

    it("per-call normalize option overrides default", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("a", [3, 0, 0, 0], { normalize: true });
      const result = db.get("a");
      expect(result![0]).toBeCloseTo(1.0);
    });

    // --- export and import (streaming) ---
    it("export returns a readable stream of the database", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("a", [1, 2, 3, 4]);
      db.set("b", [5, 6, 7, 8]);

      const stream = await db.export();
      expect(stream).toBeInstanceOf(ReadableStream);

      const bytes = await streamToBytes(stream);
      expect(bytes.byteLength).toBeGreaterThan(0);
    });

    it("export of empty database returns a valid stream", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      const stream = await db.export();
      const bytes = await streamToBytes(stream);
      expect(bytes.byteLength).toBeGreaterThan(0);
    });

    it("import restores data from an exported stream", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db1.set("alpha", [1, 0, 0, 0]);
      db1.set("beta", [0, 1, 0, 0]);

      const stream = await db1.export();

      // Import into a fresh database
      const storage2 = new InMemoryStorageProvider();
      const db2 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage: storage2,
        wasmBinary,
      });

      await db2.import(stream);
      expect(db2.size).toBe(2);
      expect(db2.get("alpha")![0]).toBeCloseTo(1);
      expect(db2.get("beta")![1]).toBeCloseTo(1);
    });

    it("import overrides existing data", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db1.set("a", [1, 0, 0, 0]);
      const stream = await db1.export();

      // Create db2 with different data
      const storage2 = new InMemoryStorageProvider();
      const db2 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage: storage2,
        wasmBinary,
      });

      db2.set("x", [0, 0, 0, 1]);
      db2.set("y", [0, 0, 1, 0]);
      expect(db2.size).toBe(2);

      await db2.import(stream);
      expect(db2.size).toBe(1);
      expect(db2.get("a")![0]).toBeCloseTo(1);
      expect(db2.get("x")).toBeUndefined();
      expect(db2.get("y")).toBeUndefined();
    });

    it("import throws on dimension mismatch", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db1.set("a", [1, 0, 0, 0]);
      const bytes = await streamToBytes(await db1.export());

      const storage2 = new InMemoryStorageProvider();
      const db2 = await VectorDB.open({
        dimensions: 8,
        normalize: false,
        storage: storage2,
        wasmBinary,
      });

      await expect(db2.import(bytesToStream(bytes))).rejects.toThrow("dimension");
    });

    it("import of empty export clears the database", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      const stream = await db1.export();

      const storage2 = new InMemoryStorageProvider();
      const db2 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage: storage2,
        wasmBinary,
      });

      db2.set("existing", [1, 0, 0, 0]);
      await db2.import(stream);
      expect(db2.size).toBe(0);
    });

    it("imported data is queryable", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db1.set("x-axis", [1, 0, 0, 0]);
      db1.set("y-axis", [0, 1, 0, 0]);
      db1.set("xy-axis", [1, 1, 0, 0]);

      const stream = await db1.export();

      const storage2 = new InMemoryStorageProvider();
      const db2 = await VectorDB.open({
        dimensions: 4,
        storage: storage2,
        wasmBinary,
      });

      await db2.import(stream);

      const results = db2.query([1, 0, 0, 0]);
      expect(results.length).toBe(3);
      expect(results[0].key).toBe("x-axis");
      expect(results[0].similarity).toBeCloseTo(1.0, 2);
    });

    it("export and import preserve data after set operations on imported db", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db1.set("a", [1, 0, 0, 0]);
      const stream = await db1.export();

      const storage2 = new InMemoryStorageProvider();
      const db2 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage: storage2,
        wasmBinary,
      });

      await db2.import(stream);
      db2.set("b", [0, 1, 0, 0]);

      expect(db2.size).toBe(2);
      expect(db2.get("a")![0]).toBeCloseTo(1);
      expect(db2.get("b")![1]).toBeCloseTo(1);
    });

    it("export throws on closed database", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      await db.close();
      await expect(db.export()).rejects.toThrow("closed");
    });

    it("import throws on closed database", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      const stream = await db1.export();

      const storage2 = new InMemoryStorageProvider();
      const db2 = await VectorDB.open({
        dimensions: 4,
        storage: storage2,
        wasmBinary,
      });

      await db2.close();
      await expect(db2.import(stream)).rejects.toThrow("closed");
    });

    it("import throws on invalid stream (bad magic)", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      const badBlob = new Uint8Array(24);
      await expect(db.import(bytesToStream(badBlob))).rejects.toThrow();
    });

    it("import throws on stream too short for header", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      const tooShort = new Uint8Array(10);
      await expect(db.import(bytesToStream(tooShort))).rejects.toThrow();
    });

    it("import throws on truncated stream body", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db1.set("a", [1, 0, 0, 0]);
      const bytes = await streamToBytes(await db1.export());

      // Truncate the blob to have valid header but incomplete body
      const truncated = bytes.slice(0, 25);

      const storage2 = new InMemoryStorageProvider();
      const db2 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage: storage2,
        wasmBinary,
      });

      await expect(db2.import(bytesToStream(truncated))).rejects.toThrow("unexpected end of stream");
    });

    it("import works correctly with chunked stream (small chunks)", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db1.set("alpha", [1, 0, 0, 0]);
      db1.set("beta", [0, 1, 0, 0]);
      db1.set("gamma", [0, 0, 1, 0]);

      const bytes = await streamToBytes(await db1.export());

      // Feed it back as a stream with tiny 7-byte chunks (crosses header/vector/key boundaries)
      const chunkedStream = bytesToChunkedStream(bytes, 7);

      const storage2 = new InMemoryStorageProvider();
      const db2 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage: storage2,
        wasmBinary,
      });

      await db2.import(chunkedStream);
      expect(db2.size).toBe(3);
      expect(db2.get("alpha")![0]).toBeCloseTo(1);
      expect(db2.get("beta")![1]).toBeCloseTo(1);
      expect(db2.get("gamma")![2]).toBeCloseTo(1);
    });

    // --- has ---
    it("has returns true for existing key", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("a", [1, 0, 0, 0]);
      expect(db.has("a")).toBe(true);
    });

    it("has returns false for non-existent key", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      expect(db.has("nonexistent")).toBe(false);
    });

    it("has throws on closed database", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      await db.close();
      expect(() => db.has("a")).toThrow("closed");
    });

    // --- delete ---
    it("delete removes an existing entry and returns true", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("a", [1, 0, 0, 0]);
      db.set("b", [0, 1, 0, 0]);
      expect(db.size).toBe(2);

      const result = db.delete("a");
      expect(result).toBe(true);
      expect(db.size).toBe(1);
      expect(db.get("a")).toBeUndefined();
      expect(db.has("a")).toBe(false);
      expect(db.get("b")).toBeDefined();
    });

    it("delete returns false for non-existent key", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      expect(db.delete("nonexistent")).toBe(false);
    });

    it("delete last entry leaves empty database", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("only", [1, 2, 3, 4]);
      db.delete("only");
      expect(db.size).toBe(0);
      expect(db.get("only")).toBeUndefined();
    });

    it("delete preserves remaining entries and query works", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("x-axis", [1, 0, 0, 0]);
      db.set("y-axis", [0, 1, 0, 0]);
      db.set("z-axis", [0, 0, 1, 0]);

      db.delete("y-axis");
      expect(db.size).toBe(2);

      const results = db.query([1, 0, 0, 0]);
      expect(results.length).toBe(2);
      expect(results[0].key).toBe("x-axis");
      expect(results.find((r) => r.key === "y-axis")).toBeUndefined();
    });

    it("delete then set reuses the database correctly", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("a", [1, 0, 0, 0]);
      db.set("b", [0, 1, 0, 0]);
      db.delete("a");

      db.set("c", [0, 0, 1, 0]);
      expect(db.size).toBe(2);
      expect(db.get("a")).toBeUndefined();
      expect(db.get("b")![1]).toBeCloseTo(1);
      expect(db.get("c")![2]).toBeCloseTo(1);
    });

    it("delete persists correctly after flush", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db1.set("a", [1, 0, 0, 0]);
      db1.set("b", [0, 1, 0, 0]);
      db1.delete("a");
      await db1.flush();

      const db2 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      expect(db2.size).toBe(1);
      expect(db2.get("a")).toBeUndefined();
      expect(db2.get("b")![1]).toBeCloseTo(1);
    });

    it("delete throws on closed database", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      await db.close();
      expect(() => db.delete("a")).toThrow("closed");
    });

    // --- keys ---
    it("keys returns an iterable of all keys", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("a", [1, 0, 0, 0]);
      db.set("b", [0, 1, 0, 0]);
      db.set("c", [0, 0, 1, 0]);

      const keys = [...db.keys()];
      expect(keys).toHaveLength(3);
      expect(keys).toContain("a");
      expect(keys).toContain("b");
      expect(keys).toContain("c");
    });

    it("keys returns empty iterable for empty database", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      expect([...db.keys()]).toEqual([]);
    });

    it("keys throws on closed database", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      await db.close();
      expect(() => db.keys()).toThrow("closed");
    });

    // --- entries ---
    it("entries returns an iterable of [key, value] pairs", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("a", [1, 0, 0, 0]);
      db.set("b", [0, 1, 0, 0]);

      const entries = [...db.entries()];
      expect(entries).toHaveLength(2);

      const aEntry = entries.find(([key]) => key === "a");
      expect(aEntry).toBeDefined();
      expect(aEntry![1][0]).toBeCloseTo(1);

      const bEntry = entries.find(([key]) => key === "b");
      expect(bEntry).toBeDefined();
      expect(bEntry![1][1]).toBeCloseTo(1);
    });

    it("entries returns empty iterable for empty database", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      expect([...db.entries()]).toEqual([]);
    });

    it("entries throws on closed database", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      await db.close();
      expect(() => db.entries()).toThrow("closed");
    });

    // --- Symbol.iterator ---
    it("supports spread operator via Symbol.iterator", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("a", [1, 0, 0, 0]);
      db.set("b", [0, 1, 0, 0]);

      const spread = [...db];
      expect(spread).toHaveLength(2);

      // Same as entries()
      const entries = [...db.entries()];
      expect(spread).toEqual(entries);
    });

    it("supports for-of iteration", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("a", [1, 0, 0, 0]);
      db.set("b", [0, 1, 0, 0]);

      const collected: [string, number[]][] = [];
      for (const entry of db) {
        collected.push(entry);
      }
      expect(collected).toHaveLength(2);
    });

    it("import works correctly with single-byte stream chunks", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db1.set("k", [1, 2, 3, 4]);

      const bytes = await streamToBytes(await db1.export());
      const chunkedStream = bytesToChunkedStream(bytes, 1);

      const storage2 = new InMemoryStorageProvider();
      const db2 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage: storage2,
        wasmBinary,
      });

      await db2.import(chunkedStream);
      expect(db2.size).toBe(1);
      expect(db2.get("k")![0]).toBeCloseTo(1);
      expect(db2.get("k")![3]).toBeCloseTo(4);
    });
  }

  it("throws VectorCapacityExceededError when full", async () => {
    const err = new VectorCapacityExceededError(100);
    expect(err).toBeInstanceOf(VectorCapacityExceededError);
    expect(err.message).toContain("100");
  });

  // --- storage decoupling ---
  it("defaults to in-memory storage when no storage is provided", async () => {
    const db = await VectorDB.open({
      dimensions: 4,
      wasmBinary: null,
    });

    db.set("a", [1, 0, 0, 0]);
    expect(db.size).toBe(1);
    expect(db.get("a")).toBeDefined();

    // Flush should succeed with default in-memory storage
    await db.flush();
    await db.close();
  });

  it("accepts an explicit storage provider via options", async () => {
    const customStorage = new InMemoryStorageProvider();
    const db = await VectorDB.open({
      dimensions: 4,
      normalize: false,
      storage: customStorage,
      wasmBinary: null,
    });

    db.set("key1", [1, 2, 3, 4]);
    await db.flush();

    // Reopen with the same storage to verify persistence
    const db2 = await VectorDB.open({
      dimensions: 4,
      normalize: false,
      storage: customStorage,
      wasmBinary: null,
    });

    expect(db2.size).toBe(1);
    expect(db2.get("key1")![0]).toBeCloseTo(1);
  });
});
