/**
 * RESULT HELPERS
 *
 * Utility functions for sorting scores and producing query results.
 * Two modes:
 *   1. topKResults  — eagerly materializes a ResultItem[] (default query path)
 *   2. iterableResults — returns a lazy Iterable<ResultItem> where keys are
 *      resolved only as each item is consumed (for pagination / streaming)
 */

export interface ResultItem {
  key: string;
  score: number;
}

export type KeyResolver = (index: number) => string;

/**
 * Sort scores descending and return the top K results as a plain array.
 * All keys are resolved eagerly.
 */
export function topKResults(
  scores: Float32Array,
  resolveKey: KeyResolver,
  topK: number,
): ResultItem[] {
  const n = scores.length;
  if (n === 0) return [];

  const indices = new Uint32Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;
  indices.sort((a, b) => scores[b] - scores[a]);

  const k = Math.min(topK, n);
  const results: ResultItem[] = new Array(k);
  for (let i = 0; i < k; i++) {
    const idx = indices[i];
    results[i] = { key: resolveKey(idx), score: scores[idx] };
  }
  return results;
}

/**
 * Sort scores descending and return a lazy iterable over the top K results.
 * Keys are resolved only when each item is consumed, saving allocations
 * when the caller iterates partially (e.g., pagination).
 *
 * The returned iterable is re-iterable — each call to [Symbol.iterator]()
 * produces a fresh cursor over the same pre-sorted data.
 */
export function iterableResults(
  scores: Float32Array,
  resolveKey: KeyResolver,
  topK: number,
): Iterable<ResultItem> {
  const n = scores.length;
  if (n === 0) return [];

  const indices = new Uint32Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;
  indices.sort((a, b) => scores[b] - scores[a]);

  const k = Math.min(topK, n);

  return {
    [Symbol.iterator](): Iterator<ResultItem> {
      let i = 0;
      return {
        next(): IteratorResult<ResultItem> {
          if (i >= k) return { done: true, value: undefined };
          const idx = indices[i++];
          return {
            done: false,
            value: { key: resolveKey(idx), score: scores[idx] },
          };
        },
      };
    },
  };
}
