import { describe, it, expect, beforeEach } from "vitest";
import { readFileSync } from "fs";
import { resolve } from "path";
import { VectorDB } from "../vector-db";
import { InMemoryStorageProvider } from "../storage";
import { VectorCapacityExceededError } from "../errors";
import { compileWatToWasm } from "../wasm-compute";

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

      db.set("a", new Float32Array([1, 2, 3, 4]));
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

      db.set("a", new Float32Array([1, 0, 0, 0]));
      db.set("a", new Float32Array([0, 1, 0, 0]));

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

      db.set("key1", new Float32Array([1, 0, 0, 0]));
      db.set("key1", new Float32Array([0, 1, 0, 0]));
      db.set("key1", new Float32Array([0, 0, 1, 0]));

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

      expect(() => db.set("a", new Float32Array(8))).toThrow("dimension mismatch");
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
        ["a", new Float32Array([1, 0, 0, 0])],
        ["b", new Float32Array([0, 1, 0, 0])],
        ["c", new Float32Array([0, 0, 1, 0])],
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
        ["x", new Float32Array([1, 0, 0, 0])],
        ["x", new Float32Array([0, 0, 0, 1])],
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

      db.set("x-axis", new Float32Array([1, 0, 0, 0]));
      db.set("y-axis", new Float32Array([0, 1, 0, 0]));
      db.set("xy-axis", new Float32Array([1, 1, 0, 0]));

      const results = db.query(new Float32Array([1, 0, 0, 0]));
      expect(results.length).toBe(3);

      // x-axis should be the best match (identical direction)
      expect(results.get(0).key).toBe("x-axis");
      expect(results.get(0).score).toBeCloseTo(1.0, 2);

      // xy-axis should be second (partially aligned)
      expect(results.get(1).key).toBe("xy-axis");
      expect(results.get(1).score).toBeGreaterThan(0);

      // y-axis should be last (orthogonal)
      expect(results.get(2).key).toBe("y-axis");
      expect(results.get(2).score).toBeCloseTo(0.0, 2);
    });

    it("query respects topK option", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("a", new Float32Array([1, 0, 0, 0]));
      db.set("b", new Float32Array([0, 1, 0, 0]));
      db.set("c", new Float32Array([0, 0, 1, 0]));

      const results = db.query(new Float32Array([1, 0, 0, 0]), { topK: 2 });
      expect(results.length).toBe(2);
    });

    it("query on empty database returns empty result", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      const results = db.query(new Float32Array([1, 0, 0, 0]));
      expect(results.length).toBe(0);
    });

    it("query validates vector dimensions", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("a", new Float32Array([1, 0, 0, 0]));
      expect(() => db.query(new Float32Array(8))).toThrow("dimension mismatch");
    });

    it("query results support pagination", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      for (let i = 0; i < 5; i++) {
        const vec = new Float32Array(4);
        vec[0] = 1 - i * 0.2;
        db.set(`t${i}`, vec);
      }

      const results = db.query(new Float32Array([1, 0, 0, 0]), { normalize: false });

      const page0 = results.getPage(0, 2);
      expect(page0).toHaveLength(2);

      const page1 = results.getPage(1, 2);
      expect(page1).toHaveLength(2);

      const page2 = results.getPage(2, 2);
      expect(page2).toHaveLength(1);
    });

    it("query after overwrite uses updated vector", async () => {
      const db = await VectorDB.open({
        dimensions: 4,
        storage,
        wasmBinary,
      });

      db.set("point", new Float32Array([1, 0, 0, 0]));
      db.set("other", new Float32Array([0, 1, 0, 0]));

      // Overwrite 'point' to be along y-axis
      db.set("point", new Float32Array([0, 1, 0, 0]));

      const results = db.query(new Float32Array([0, 1, 0, 0]));
      // Both 'point' and 'other' are now along y-axis, so both should score high
      expect(results.get(0).score).toBeCloseTo(1.0, 2);
      expect(results.get(1).score).toBeCloseTo(1.0, 2);
      expect(db.size).toBe(2);
    });

    // --- flush and persistence ---
    it("flush persists data and reopen loads it", async () => {
      const db1 = await VectorDB.open({
        dimensions: 4,
        normalize: false,
        storage,
        wasmBinary,
      });

      db1.set("alpha", new Float32Array([1, 0, 0, 0]));
      db1.set("beta", new Float32Array([0, 1, 0, 0]));
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

      db1.set("key", new Float32Array([1, 0, 0, 0]));
      db1.set("key", new Float32Array([0, 0, 0, 1]));
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

      db.set("a", new Float32Array([1, 0, 0, 0]));
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
      expect(() => db.set("b", new Float32Array(4))).toThrow("closed");
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

      db.set("a", new Float32Array([1, 0, 0, 0]));
      db.set("b", new Float32Array([0, 1, 0, 0]));
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

      db.set("old", new Float32Array([1, 0, 0, 0]));
      await db.clear();

      db.set("new", new Float32Array([0, 1, 0, 0]));
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

      db.set("a", new Float32Array([3, 0, 0, 0]));
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

      db.set("a", new Float32Array([3, 0, 0, 0]));
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

      db.set("a", new Float32Array([3, 0, 0, 0]), { normalize: true });
      const result = db.get("a");
      expect(result![0]).toBeCloseTo(1.0);
    });
  }

  it("throws VectorCapacityExceededError when full", async () => {
    const err = new VectorCapacityExceededError(100);
    expect(err).toBeInstanceOf(VectorCapacityExceededError);
    expect(err.message).toContain("100");
  });
});
