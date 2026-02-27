import { describe, expect, it } from "vitest";
import { iterableResults, queryResults } from "../result-set";

describe("queryResults", () => {
  const keys = ["apple", "banana", "cherry", "date", "elderberry"];
  const resolveKey = (index: number) => keys[index];

  it("sorts results by descending similarity (default order)", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = queryResults(scores, resolveKey, { limit: Infinity, order: "descend" });

    expect(results).toHaveLength(5);
    expect(results[0].key).toBe("banana");
    expect(results[0].similarity).toBeCloseTo(0.9, 4);
    expect(results[1].key).toBe("date");
    expect(results[1].similarity).toBeCloseTo(0.7, 4);
    expect(results[2].key).toBe("elderberry");
    expect(results[3].key).toBe("apple");
    expect(results[4].key).toBe("cherry");
  });

  it("sorts results by ascending similarity", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = queryResults(scores, resolveKey, { limit: Infinity, order: "ascend" });

    expect(results).toHaveLength(5);
    expect(results[0].key).toBe("cherry");
    expect(results[0].similarity).toBeCloseTo(0.1, 4);
    expect(results[1].key).toBe("apple");
    expect(results[1].similarity).toBeCloseTo(0.3, 4);
    expect(results[4].key).toBe("banana");
    expect(results[4].similarity).toBeCloseTo(0.9, 4);
  });

  it("respects limit", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = queryResults(scores, resolveKey, { limit: 3, order: "descend" });

    expect(results).toHaveLength(3);
    expect(results[0].key).toBe("banana");
    expect(results[0].similarity).toBeCloseTo(0.9, 4);
    expect(results[2].key).toBe("elderberry");
  });

  it("handles empty scores", () => {
    const scores = new Float32Array(0);
    const results = queryResults(scores, resolveKey, { limit: 10, order: "descend" });
    expect(results).toEqual([]);
  });

  it("handles limit larger than result count", () => {
    const scores = new Float32Array([0.5, 0.8]);
    const results = queryResults(scores, resolveKey, { limit: 100, order: "descend" });
    expect(results).toHaveLength(2);
  });

  it("filters results by minSimilarity", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = queryResults(scores, resolveKey, { limit: Infinity, order: "descend", minSimilarity: 0.5 });

    expect(results).toHaveLength(3);
    expect(results[0].key).toBe("banana");
    expect(results[0].similarity).toBeCloseTo(0.9, 4);
    expect(results[1].key).toBe("date");
    expect(results[1].similarity).toBeCloseTo(0.7, 4);
    expect(results[2].key).toBe("elderberry");
    expect(results[2].similarity).toBeCloseTo(0.5, 4);
  });

  it("minSimilarity is inclusive", () => {
    const scores = new Float32Array([0.5]);
    const results = queryResults(scores, resolveKey, { limit: 10, order: "descend", minSimilarity: 0.5 });
    expect(results).toHaveLength(1);
  });

  it("filters results by maxSimilarity", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = queryResults(scores, resolveKey, { limit: Infinity, order: "descend", maxSimilarity: 0.7 });

    expect(results).toHaveLength(4);
    expect(results[0].key).toBe("date");
    expect(results[0].similarity).toBeCloseTo(0.7, 4);
    expect(results[1].key).toBe("elderberry");
    expect(results[2].key).toBe("apple");
    expect(results[3].key).toBe("cherry");
  });

  it("maxSimilarity is inclusive", () => {
    const scores = new Float32Array([0.5]);
    const results = queryResults(scores, resolveKey, { limit: 10, order: "descend", maxSimilarity: 0.5 });
    expect(results).toHaveLength(1);
  });

  it("filters by both minSimilarity and maxSimilarity", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = queryResults(scores, resolveKey, { limit: Infinity, order: "descend", minSimilarity: 0.3, maxSimilarity: 0.7 });

    expect(results).toHaveLength(3);
    expect(results[0].key).toBe("date");
    expect(results[1].key).toBe("elderberry");
    expect(results[2].key).toBe("apple");
  });

  it("handles limit Infinity", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = queryResults(scores, resolveKey, { limit: Infinity, order: "descend" });
    expect(results).toHaveLength(5);
  });

  it("supports full similarity range [-1, 1]", () => {
    const scores = new Float32Array([-1.0, -0.5, 0.0, 0.5, 1.0]);
    const results = queryResults(scores, resolveKey, { limit: Infinity, order: "descend" });

    expect(results).toHaveLength(5);
    expect(results[0].similarity).toBeCloseTo(1.0, 5);
    expect(results[4].similarity).toBeCloseTo(-1.0, 5);
  });

  it("handles tiny floating point values near boundaries", () => {
    const epsilon = 1e-7;
    const scores = new Float32Array([0.5 - epsilon, 0.5, 0.5 + epsilon]);
    const results = queryResults(scores, resolveKey, { limit: Infinity, order: "descend", minSimilarity: 0.5 });

    // 0.5 and 0.5 + epsilon should pass, 0.5 - epsilon should not
    expect(results).toHaveLength(2);
  });

  it("ascending order with limit returns bottomK", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = queryResults(scores, resolveKey, { limit: 2, order: "ascend" });

    expect(results).toHaveLength(2);
    expect(results[0].key).toBe("cherry");
    expect(results[0].similarity).toBeCloseTo(0.1, 4);
    expect(results[1].key).toBe("apple");
    expect(results[1].similarity).toBeCloseTo(0.3, 4);
  });

  it("ascending order with minSimilarity and maxSimilarity", () => {
    const scores = new Float32Array([-1.0, -0.5, 0.0, 0.5, 1.0]);
    const results = queryResults(scores, resolveKey, { limit: Infinity, order: "ascend", minSimilarity: -0.5, maxSimilarity: 0.5 });

    expect(results).toHaveLength(3);
    expect(results[0].similarity).toBeCloseTo(-0.5, 5);
    expect(results[1].similarity).toBeCloseTo(0.0, 5);
    expect(results[2].similarity).toBeCloseTo(0.5, 5);
  });
});

describe("iterableResults", () => {
  const keys = ["apple", "banana", "cherry", "date", "elderberry"];
  const resolveKey = (index: number) => keys[index];

  it("sorts results by descending similarity", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = [...iterableResults(scores, resolveKey, { limit: Infinity, order: "descend" })];

    expect(results).toHaveLength(5);
    expect(results[0].key).toBe("banana");
    expect(results[0].similarity).toBeCloseTo(0.9, 4);
    expect(results[1].key).toBe("date");
    expect(results[4].key).toBe("cherry");
  });

  it("sorts results by ascending similarity", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = [...iterableResults(scores, resolveKey, { limit: Infinity, order: "ascend" })];

    expect(results).toHaveLength(5);
    expect(results[0].key).toBe("cherry");
    expect(results[0].similarity).toBeCloseTo(0.1, 4);
    expect(results[4].key).toBe("banana");
  });

  it("respects limit", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = [...iterableResults(scores, resolveKey, { limit: 3, order: "descend" })];

    expect(results).toHaveLength(3);
    expect(results[0].key).toBe("banana");
  });

  it("handles empty scores", () => {
    const results = [...iterableResults(new Float32Array(0), resolveKey, { limit: 10, order: "descend" })];
    expect(results).toEqual([]);
  });

  it("only resolves keys lazily (on consumption)", () => {
    let callCount = 0;
    const lazyResolver = (index: number) => {
      callCount++;
      return keys[index];
    };

    const scores = new Float32Array([0.3, 0.9, 0.1]);
    const iterable = iterableResults(scores, lazyResolver, { limit: Infinity, order: "descend" });

    expect(callCount).toBe(0); // no key resolved yet

    const iter = iterable[Symbol.iterator]();
    iter.next();
    expect(callCount).toBe(1); // resolved only 1

    iter.next();
    expect(callCount).toBe(2);
  });

  it("is re-iterable", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1]);
    const iterable = iterableResults(scores, resolveKey, { limit: Infinity, order: "descend" });

    const first = [...iterable];
    const second = [...iterable];
    expect(first).toEqual(second);
  });

  it("supports partial iteration (early break)", () => {
    const scores = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]);
    const iterable = iterableResults(scores, resolveKey, { limit: Infinity, order: "descend" });

    const partial: string[] = [];
    for (const item of iterable) {
      partial.push(item.key);
      if (partial.length === 2) break;
    }
    expect(partial).toHaveLength(2);
    expect(partial[0]).toBe("elderberry"); // similarity 0.5 (highest)
    expect(partial[1]).toBe("date"); // similarity 0.4
  });

  it("filters by minSimilarity", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = [...iterableResults(scores, resolveKey, { limit: Infinity, order: "descend", minSimilarity: 0.5 })];

    expect(results).toHaveLength(3);
    expect(results[0].key).toBe("banana");
    expect(results[1].key).toBe("date");
    expect(results[2].key).toBe("elderberry");
  });

  it("minSimilarity is inclusive", () => {
    const scores = new Float32Array([0.5]);
    const results = [...iterableResults(scores, resolveKey, { limit: Infinity, order: "descend", minSimilarity: 0.5 })];
    expect(results).toHaveLength(1);
  });

  it("filters by maxSimilarity", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = [...iterableResults(scores, resolveKey, { limit: Infinity, order: "descend", maxSimilarity: 0.7 })];

    expect(results).toHaveLength(4);
    expect(results[0].key).toBe("date");
    expect(results[3].key).toBe("cherry");
  });

  it("maxSimilarity is inclusive", () => {
    const scores = new Float32Array([0.5]);
    const results = [...iterableResults(scores, resolveKey, { limit: Infinity, order: "descend", maxSimilarity: 0.5 })];
    expect(results).toHaveLength(1);
  });

  it("ascending order with limit returns bottomK", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = [...iterableResults(scores, resolveKey, { limit: 2, order: "ascend" })];

    expect(results).toHaveLength(2);
    expect(results[0].key).toBe("cherry");
    expect(results[1].key).toBe("apple");
  });

  it("supports full similarity range [-1, 1]", () => {
    const scores = new Float32Array([-1.0, -0.5, 0.0, 0.5, 1.0]);
    const results = [...iterableResults(scores, resolveKey, { limit: Infinity, order: "ascend" })];

    expect(results).toHaveLength(5);
    expect(results[0].similarity).toBeCloseTo(-1.0, 5);
    expect(results[4].similarity).toBeCloseTo(1.0, 5);
  });
});
