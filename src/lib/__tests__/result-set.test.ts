import { describe, it, expect } from "vitest";
import { topKResults, iterableResults } from "../result-set";

describe("topKResults", () => {
  const keys = ["apple", "banana", "cherry", "date", "elderberry"];
  const resolveKey = (index: number) => keys[index];

  it("sorts results by descending score", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = topKResults(scores, resolveKey, 5);

    expect(results).toHaveLength(5);
    expect(results[0].key).toBe("banana");
    expect(results[0].score).toBeCloseTo(0.9, 4);
    expect(results[1].key).toBe("date");
    expect(results[1].score).toBeCloseTo(0.7, 4);
    expect(results[2].key).toBe("elderberry");
    expect(results[3].key).toBe("apple");
    expect(results[4].key).toBe("cherry");
  });

  it("respects topK limit", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = topKResults(scores, resolveKey, 3);

    expect(results).toHaveLength(3);
    expect(results[0].key).toBe("banana");
    expect(results[0].score).toBeCloseTo(0.9, 4);
    expect(results[2].key).toBe("elderberry");
  });

  it("handles empty scores", () => {
    const scores = new Float32Array(0);
    const results = topKResults(scores, resolveKey, 10);
    expect(results).toEqual([]);
  });

  it("handles topK larger than result count", () => {
    const scores = new Float32Array([0.5, 0.8]);
    const results = topKResults(scores, resolveKey, 100);
    expect(results).toHaveLength(2);
  });
});

describe("iterableResults", () => {
  const keys = ["apple", "banana", "cherry", "date", "elderberry"];
  const resolveKey = (index: number) => keys[index];

  it("sorts results by descending score", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = [...iterableResults(scores, resolveKey, 5)];

    expect(results).toHaveLength(5);
    expect(results[0].key).toBe("banana");
    expect(results[0].score).toBeCloseTo(0.9, 4);
    expect(results[1].key).toBe("date");
    expect(results[4].key).toBe("cherry");
  });

  it("respects topK limit", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const results = [...iterableResults(scores, resolveKey, 3)];

    expect(results).toHaveLength(3);
    expect(results[0].key).toBe("banana");
  });

  it("handles empty scores", () => {
    const results = [...iterableResults(new Float32Array(0), resolveKey, 10)];
    expect(results).toEqual([]);
  });

  it("only resolves keys lazily (on consumption)", () => {
    let callCount = 0;
    const lazyResolver = (index: number) => {
      callCount++;
      return keys[index];
    };

    const scores = new Float32Array([0.3, 0.9, 0.1]);
    const iterable = iterableResults(scores, lazyResolver, 3);

    expect(callCount).toBe(0); // no key resolved yet

    const iter = iterable[Symbol.iterator]();
    iter.next();
    expect(callCount).toBe(1); // resolved only 1

    iter.next();
    expect(callCount).toBe(2);
  });

  it("is re-iterable", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1]);
    const iterable = iterableResults(scores, resolveKey, 3);

    const first = [...iterable];
    const second = [...iterable];
    expect(first).toEqual(second);
  });

  it("supports partial iteration (early break)", () => {
    const scores = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]);
    const iterable = iterableResults(scores, resolveKey, 5);

    const partial: string[] = [];
    for (const item of iterable) {
      partial.push(item.key);
      if (partial.length === 2) break;
    }
    expect(partial).toHaveLength(2);
    expect(partial[0]).toBe("elderberry"); // score 0.5
    expect(partial[1]).toBe("date"); // score 0.4
  });
});
