import { describe, it, expect } from "vitest";
import { normalize, searchAll } from "../compute";

describe("normalize", () => {
  it("normalizes a vector to unit length", () => {
    const vec = new Float32Array([3, 4]);
    normalize(vec);
    expect(vec[0]).toBeCloseTo(0.6, 5);
    expect(vec[1]).toBeCloseTo(0.8, 5);
    // Verify unit length
    const mag = Math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
    expect(mag).toBeCloseTo(1.0, 5);
  });

  it("normalizes a higher-dimensional vector", () => {
    const vec = new Float32Array([1, 2, 3, 4]);
    normalize(vec);
    let sumSq = 0;
    for (let i = 0; i < vec.length; i++) sumSq += vec[i] * vec[i];
    expect(Math.sqrt(sumSq)).toBeCloseTo(1.0, 5);
  });

  it("handles a zero vector gracefully", () => {
    const vec = new Float32Array([0, 0, 0]);
    normalize(vec);
    expect(vec[0]).toBe(0);
    expect(vec[1]).toBe(0);
    expect(vec[2]).toBe(0);
  });

  it("handles a single-element vector", () => {
    const vec = new Float32Array([5]);
    normalize(vec);
    expect(vec[0]).toBeCloseTo(1.0, 5);
  });

  it("handles already-normalized vector", () => {
    const vec = new Float32Array([0.6, 0.8]);
    normalize(vec);
    expect(vec[0]).toBeCloseTo(0.6, 5);
    expect(vec[1]).toBeCloseTo(0.8, 5);
  });
});

describe("searchAll", () => {
  it("computes dot products for a single database vector", () => {
    const dimensions = 3;
    const query = new Float32Array([1, 0, 0]);
    const db = new Float32Array([0.5, 0.5, 0]);
    const scores = new Float32Array(1);

    searchAll(query, db, scores, 1, dimensions);
    expect(scores[0]).toBeCloseTo(0.5, 5);
  });

  it("computes dot products for multiple database vectors", () => {
    const dimensions = 2;
    // Two normalized vectors
    const query = new Float32Array([1, 0]);
    const db = new Float32Array([
      1, 0, // identical to query
      0, 1, // orthogonal to query
    ]);
    const scores = new Float32Array(2);

    searchAll(query, db, scores, 2, dimensions);
    expect(scores[0]).toBeCloseTo(1.0, 5); // identical
    expect(scores[1]).toBeCloseTo(0.0, 5); // orthogonal
  });

  it("computes correct scores for normalized vectors (cosine similarity)", () => {
    const dimensions = 3;
    const q = new Float32Array([1, 2, 3]);
    normalize(q);

    const v1 = new Float32Array([1, 2, 3]); // same direction
    normalize(v1);
    const v2 = new Float32Array([-1, -2, -3]); // opposite direction
    normalize(v2);
    const v3 = new Float32Array([0, 0, 1]); // different direction
    normalize(v3);

    const db = new Float32Array([...v1, ...v2, ...v3]);
    const scores = new Float32Array(3);

    searchAll(q, db, scores, 3, dimensions);
    expect(scores[0]).toBeCloseTo(1.0, 4); // same direction = 1
    expect(scores[1]).toBeCloseTo(-1.0, 4); // opposite = -1
    expect(scores[2]).toBeGreaterThan(-1);
    expect(scores[2]).toBeLessThan(1);
  });

  it("handles empty database", () => {
    const query = new Float32Array([1, 0, 0]);
    const db = new Float32Array(0);
    const scores = new Float32Array(0);

    searchAll(query, db, scores, 0, 3);
    expect(scores.length).toBe(0);
  });

  it("handles high-dimensional vectors (128 dims)", () => {
    const dimensions = 128;
    const query = new Float32Array(dimensions);
    query[0] = 1; // unit vector along first axis

    const dbSize = 10;
    const db = new Float32Array(dbSize * dimensions);
    for (let i = 0; i < dbSize; i++) {
      db[i * dimensions + i % dimensions] = 1; // unit vectors along different axes
    }

    const scores = new Float32Array(dbSize);
    searchAll(query, db, scores, dbSize, dimensions);

    expect(scores[0]).toBeCloseTo(1.0, 5); // same axis as query
    for (let i = 1; i < dbSize; i++) {
      expect(scores[i]).toBeCloseTo(0.0, 5); // orthogonal
    }
  });
});
