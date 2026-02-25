import { describe, it, expect } from "vitest";
import { ResultSet } from "../result-set";

describe("ResultSet", () => {
  const keys = ["apple", "banana", "cherry", "date", "elderberry"];
  const resolveKey = (index: number) => keys[index];

  it("sorts results by descending score", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const rs = ResultSet.fromScores(scores, resolveKey, 5);

    expect(rs.length).toBe(5);
    expect(rs.get(0).key).toBe("banana");
    expect(rs.get(0).score).toBeCloseTo(0.9, 4);
    expect(rs.get(1).key).toBe("date");
    expect(rs.get(1).score).toBeCloseTo(0.7, 4);
    expect(rs.get(2).key).toBe("elderberry");
    expect(rs.get(3).key).toBe("apple");
    expect(rs.get(4).key).toBe("cherry");
  });

  it("respects topK limit", () => {
    const scores = new Float32Array([0.3, 0.9, 0.1, 0.7, 0.5]);
    const rs = ResultSet.fromScores(scores, resolveKey, 3);

    expect(rs.length).toBe(3);
    expect(rs.get(0).key).toBe("banana");
    expect(rs.get(0).score).toBeCloseTo(0.9, 4);
    expect(rs.get(2).key).toBe("elderberry");
  });

  it("throws on out-of-bounds rank", () => {
    const scores = new Float32Array([0.5, 0.8]);
    const rs = ResultSet.fromScores(scores, resolveKey, 2);

    expect(() => rs.get(-1)).toThrow(RangeError);
    expect(() => rs.get(2)).toThrow(RangeError);
  });

  it("returns correct pages", () => {
    const scores = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]);
    const rs = ResultSet.fromScores(scores, resolveKey, 5);

    const page0 = rs.getPage(0, 2);
    expect(page0).toHaveLength(2);
    expect(page0[0].key).toBe("elderberry"); // score 0.5
    expect(page0[1].key).toBe("date"); // score 0.4

    const page1 = rs.getPage(1, 2);
    expect(page1).toHaveLength(2);
    expect(page1[0].key).toBe("cherry"); // score 0.3
    expect(page1[1].key).toBe("banana"); // score 0.2

    const page2 = rs.getPage(2, 2);
    expect(page2).toHaveLength(1); // only 1 remaining
    expect(page2[0].key).toBe("apple"); // score 0.1
  });

  it("handles empty results", () => {
    const scores = new Float32Array(0);
    const rs = ResultSet.fromScores(scores, resolveKey, 10);
    expect(rs.length).toBe(0);
    expect(rs.getPage(0, 10)).toEqual([]);
  });

  it("only resolves keys lazily (on access)", () => {
    let callCount = 0;
    const lazyResolver = (index: number) => {
      callCount++;
      return keys[index];
    };

    const scores = new Float32Array([0.3, 0.9, 0.1]);
    const rs = ResultSet.fromScores(scores, lazyResolver, 3);

    expect(callCount).toBe(0); // no key resolved yet

    rs.get(0);
    expect(callCount).toBe(1); // resolved only 1

    rs.getPage(0, 2);
    expect(callCount).toBe(3); // resolved 2 more
  });

  it("handles topK larger than result count", () => {
    const scores = new Float32Array([0.5, 0.8]);
    const rs = ResultSet.fromScores(scores, resolveKey, 100);
    expect(rs.length).toBe(2);
  });
});
