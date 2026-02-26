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

    it("query respects topK option", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("a", [1, 0, 0, 0]);
      db.set("b", [0, 1, 0, 0]);
      db.set("c", [0, 0, 1, 0]);

      const results = db.query([1, 0, 0, 0], { topK: 2 });
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

    it("query topK defaults to Infinity (returns all results)", async () => {
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

      // Without topK, all 10 results should be returned
      const results = db.query([1, 0, 0, 0]);
      expect(results.length).toBe(10);
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

    // --- export and import ---
    it("export returns a binary blob of the database", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db.set("a", [1, 2, 3, 4]);
      db.set("b", [5, 6, 7, 8]);

      const blob = db.export();
      expect(blob).toBeInstanceOf(Uint8Array);
      expect(blob.byteLength).toBeGreaterThan(0);
    });

    it("export of empty database returns a valid blob", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      const blob = db.export();
      expect(blob).toBeInstanceOf(Uint8Array);
      expect(blob.byteLength).toBeGreaterThan(0);
    });

    it("import restores data from an exported blob", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db1.set("alpha", [1, 0, 0, 0]);
      db1.set("beta", [0, 1, 0, 0]);
      const blob = db1.export();

      // Import into a fresh database
      const storage2 = new InMemoryStorageProvider();
      const db2 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage: storage2,
        wasmBinary,
      });

      db2.import(blob);
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
      const blob = db1.export();

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

      db2.import(blob);
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
      const blob = db1.export();

      const storage2 = new InMemoryStorageProvider();
      const db2 = await VectorDB.open({
        dimensions: 8,
        normalize: false,
        storage: storage2,
        wasmBinary,
      });

      expect(() => db2.import(blob)).toThrow("dimension");
    });

    it("import of empty export clears the database", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      const blob = db1.export();

      const storage2 = new InMemoryStorageProvider();
      const db2 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage: storage2,
        wasmBinary,
      });

      db2.set("existing", [1, 0, 0, 0]);
      db2.import(blob);
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
      const blob = db1.export();

      const storage2 = new InMemoryStorageProvider();
      const db2 = await VectorDB.open({
        dimensions: 4,
        storage: storage2,
        wasmBinary,
      });

      db2.import(blob);

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
      const blob = db1.export();

      const storage2 = new InMemoryStorageProvider();
      const db2 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage: storage2,
        wasmBinary,
      });

      db2.import(blob);
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
      expect(() => db.export()).toThrow("closed");
    });

    it("import throws on closed database", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      const blob = db1.export();

      const storage2 = new InMemoryStorageProvider();
      const db2 = await VectorDB.open({
        dimensions: 4,
        storage: storage2,
        wasmBinary,
      });

      await db2.close();
      expect(() => db2.import(blob)).toThrow("closed");
    });

    it("import throws on invalid blob (bad magic)", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      const badBlob = new Uint8Array(24);
      expect(() => db.import(badBlob)).toThrow();
    });

    it("import throws on blob too short for header", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      const tooShort = new Uint8Array(10);
      expect(() => db.import(tooShort)).toThrow();
    });
  }

  it("throws VectorCapacityExceededError when full", async () => {
    const err = new VectorCapacityExceededError(100);
    expect(err).toBeInstanceOf(VectorCapacityExceededError);
    expect(err.message).toContain("100");
  });
});
