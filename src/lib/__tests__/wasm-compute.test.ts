import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "fs";
import { resolve } from "path";
import { compileWatToWasm, instantiateWasm, type WasmExports } from "../wasm-compute";

describe("WASM SIMD compute", () => {
  let memory: WebAssembly.Memory;
  let wasm: WasmExports;

  beforeAll(async () => {
    const watSource = readFileSync(resolve(__dirname, "../simd.wat"), "utf-8");
    const wasmBinary = await compileWatToWasm(watSource);
    memory = new WebAssembly.Memory({ initial: 1 }); // 64KB
    wasm = await instantiateWasm(wasmBinary, memory);
  });

  function writeFloat32Array(offset: number, data: Float32Array): void {
    new Float32Array(memory.buffer, offset, data.length).set(data);
  }

  function readFloat32Array(offset: number, length: number): Float32Array {
    return new Float32Array(memory.buffer.slice(offset, offset + length * 4));
  }

  describe("normalize", () => {
    it("normalizes a vector to unit length", () => {
      const vec = new Float32Array([3, 4, 0, 0]);
      const ptr = 0;
      writeFloat32Array(ptr, vec);

      wasm.normalize(ptr, 4);

      const result = readFloat32Array(ptr, 4);
      const mag = Math.sqrt(result[0] ** 2 + result[1] ** 2 + result[2] ** 2 + result[3] ** 2);
      expect(mag).toBeCloseTo(1.0, 4);
      expect(result[0]).toBeCloseTo(0.6, 4);
      expect(result[1]).toBeCloseTo(0.8, 4);
    });

    it("normalizes a non-multiple-of-4 dimension vector", () => {
      const vec = new Float32Array([1, 2, 3]);
      const ptr = 0;
      writeFloat32Array(ptr, vec);

      wasm.normalize(ptr, 3);

      const result = readFloat32Array(ptr, 3);
      let sumSq = 0;
      for (let i = 0; i < 3; i++) sumSq += result[i] * result[i];
      expect(Math.sqrt(sumSq)).toBeCloseTo(1.0, 4);
    });

    it("handles zero vector", () => {
      const vec = new Float32Array([0, 0, 0, 0]);
      const ptr = 0;
      writeFloat32Array(ptr, vec);

      wasm.normalize(ptr, 4);

      const result = readFloat32Array(ptr, 4);
      expect(result[0]).toBe(0);
      expect(result[1]).toBe(0);
    });
  });

  describe("search_all", () => {
    it("computes dot products correctly", () => {
      const dimensions = 4;
      const dbSize = 2;
      const queryPtr = 0;
      const dbPtr = dimensions * 4; // after query
      const scoresPtr = dbPtr + dbSize * dimensions * 4; // after db

      const query = new Float32Array([1, 0, 0, 0]);
      const db = new Float32Array([
        1, 0, 0, 0, // identical to query
        0, 1, 0, 0, // orthogonal
      ]);

      writeFloat32Array(queryPtr, query);
      writeFloat32Array(dbPtr, db);

      wasm.search_all(queryPtr, dbPtr, scoresPtr, dbSize, dimensions);

      const scores = readFloat32Array(scoresPtr, dbSize);
      expect(scores[0]).toBeCloseTo(1.0, 4);
      expect(scores[1]).toBeCloseTo(0.0, 4);
    });

    it("computes correct scores for normalized vectors", () => {
      const dimensions = 4;
      const dbSize = 3;
      const queryPtr = 0;
      const dbPtr = dimensions * 4;
      const scoresPtr = dbPtr + dbSize * dimensions * 4;

      // Create and normalize vectors manually
      const q = new Float32Array([1, 2, 3, 0]);
      const mag = Math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2);
      for (let i = 0; i < 3; i++) q[i] /= mag;

      const v1 = new Float32Array([1, 2, 3, 0]); // same direction
      const v1mag = Math.sqrt(14);
      for (let i = 0; i < 3; i++) v1[i] /= v1mag;

      const v2 = new Float32Array([-1, -2, -3, 0]); // opposite
      for (let i = 0; i < 3; i++) v2[i] /= v1mag;

      const v3 = new Float32Array([0, 0, 0, 1]); // orthogonal

      writeFloat32Array(queryPtr, q);
      writeFloat32Array(dbPtr, new Float32Array([...v1, ...v2, ...v3]));

      wasm.search_all(queryPtr, dbPtr, scoresPtr, dbSize, dimensions);

      const scores = readFloat32Array(scoresPtr, dbSize);
      expect(scores[0]).toBeCloseTo(1.0, 3); // same direction
      expect(scores[1]).toBeCloseTo(-1.0, 3); // opposite
      expect(scores[2]).toBeCloseTo(0.0, 3); // orthogonal
    });

    it("handles non-multiple-of-4 dimensions", () => {
      const dimensions = 3;
      const dbSize = 1;
      const queryPtr = 0;
      const dbPtr = 16; // aligned
      const scoresPtr = dbPtr + dbSize * dimensions * 4 + 4; // extra space

      const query = new Float32Array([1, 0, 0]);
      const db = new Float32Array([0.5, 0.5, 0]);

      writeFloat32Array(queryPtr, query);
      writeFloat32Array(dbPtr, db);

      wasm.search_all(queryPtr, dbPtr, scoresPtr, dbSize, dimensions);

      const scores = readFloat32Array(scoresPtr, dbSize);
      expect(scores[0]).toBeCloseTo(0.5, 4);
    });

    it("handles empty database", () => {
      const queryPtr = 0;
      const dbPtr = 16;
      const scoresPtr = 32;

      writeFloat32Array(queryPtr, new Float32Array([1, 0, 0, 0]));

      // Should not crash with dbSize=0
      wasm.search_all(queryPtr, dbPtr, scoresPtr, 0, 4);
    });
  });
});
