import { describe, it, expect } from "vitest";
import { MemoryManager } from "../memory-manager";

describe("MemoryManager", () => {
  it("initializes with correct layout", () => {
    const mm = new MemoryManager(4);
    expect(mm.queryOffset).toBe(0);
    expect(mm.dbOffset).toBe(65536); // aligned to 64KB page
    expect(mm.vectorCount).toBe(0);
  });

  it("calculates dbOffset as page-aligned", () => {
    // For 1536-dim vectors: 1536 * 4 = 6144 bytes < 64KB
    const mm = new MemoryManager(1536);
    expect(mm.dbOffset).toBe(65536);

    // For very large dimensions that exceed one page
    const mm2 = new MemoryManager(20000); // 20000 * 4 = 80000 bytes > 64KB
    expect(mm2.dbOffset).toBe(131072); // 2 pages
  });

  it("writes and reads query vector", () => {
    const mm = new MemoryManager(4);
    const query = new Float32Array([1, 2, 3, 4]);
    mm.writeQuery(query);

    const read = new Float32Array(mm.memory.buffer, mm.queryOffset, 4);
    expect(read[0]).toBe(1);
    expect(read[1]).toBe(2);
    expect(read[2]).toBe(3);
    expect(read[3]).toBe(4);
  });

  it("appends vectors and updates count", () => {
    const mm = new MemoryManager(3);
    const v1 = new Float32Array([1, 0, 0]);
    const v2 = new Float32Array([0, 1, 0]);

    mm.ensureCapacity(2);
    mm.appendVectors([v1, v2]);

    expect(mm.vectorCount).toBe(2);

    const read1 = mm.readVector(0);
    expect(read1[0]).toBe(1);
    expect(read1[1]).toBe(0);

    const read2 = mm.readVector(1);
    expect(read2[0]).toBe(0);
    expect(read2[1]).toBe(1);
  });

  it("computes scores offset after DB", () => {
    const mm = new MemoryManager(4);
    mm.ensureCapacity(10);
    mm.appendVectors(Array.from({ length: 10 }, () => new Float32Array(4)));

    // DB at page 1 (65536), 10 vectors of 4*4 = 160 bytes
    expect(mm.scoresOffset).toBe(65536 + 10 * 4 * 4);
  });

  it("loads bulk vector bytes", () => {
    const mm = new MemoryManager(3);
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const bytes = new Uint8Array(data.buffer);

    mm.ensureCapacity(2);
    mm.loadVectorBytes(bytes, 2);

    expect(mm.vectorCount).toBe(2);
    const v0 = mm.readVector(0);
    expect(v0[0]).toBe(1);
    expect(v0[1]).toBe(2);
    expect(v0[2]).toBe(3);
  });

  it("grows memory when needed", () => {
    const mm = new MemoryManager(1536);
    const initialSize = mm.memory.buffer.byteLength;

    // Ensure capacity for many vectors
    mm.ensureCapacity(1000);
    expect(mm.memory.buffer.byteLength).toBeGreaterThan(initialSize);
  });

  it("calculates maxVectors based on WASM limit", () => {
    const mm = new MemoryManager(1536);
    // With 1536 dims: each vector = 1536*4 + 4 = 6148 bytes
    // Available = 4GB - 64KB ≈ 4294901760 bytes
    // Max ≈ 4294901760 / 6148 ≈ 698,697
    expect(mm.maxVectors).toBeGreaterThan(600000);
    expect(mm.maxVectors).toBeLessThan(800000);
  });
});
