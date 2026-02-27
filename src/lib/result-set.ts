/**
 * RESULT HELPERS
 *
 * Utility functions for sorting scores and producing query results.
 * Two modes:
 *   1. queryResults  — eagerly materializes a ResultItem[] (default query path)
 *   2. iterableResults — returns a lazy Iterable<ResultItem> where keys are
 *      resolved only as each item is consumed (for pagination / streaming)
 *
 * Similarity is the dot product of query and stored vectors. For normalized
 * vectors (the default), this equals cosine similarity, ranging from 1
 * (identical) to -1 (opposite).
 */

export interface ResultItem {
  key: string;
  similarity: number;
}

export type KeyResolver = (index: number) => string;

export interface ResultOptions {
  limit: number;
  order: "ascend" | "descend";
  minSimilarity?: number;
  maxSimilarity?: number;
}

/** Check whether a score falls within the [minSimilarity, maxSimilarity] range. */
function inRange(similarity: number, minSimilarity?: number, maxSimilarity?: number): boolean {
  if (minSimilarity !== undefined && similarity < minSimilarity) return false;
  if (maxSimilarity !== undefined && similarity > maxSimilarity) return false;
  return true;
}

/**
 * Sort scores and return the top results as a plain array.
 * All keys are resolved eagerly.
 *
 * `order` controls sort direction:
 *   - "descend" (default) — highest similarity first
 *   - "ascend" — lowest similarity first
 *
 * Results outside [minSimilarity, maxSimilarity] are excluded.
 */
export function queryResults(
  scores: Float32Array,
  resolveKey: KeyResolver,
  options: ResultOptions,
): ResultItem[] {
  const { limit, order, minSimilarity, maxSimilarity } = options;
  const n = scores.length;
  if (n === 0) return [];

  const indices = new Uint32Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;

  if (order === "ascend") {
    indices.sort((a, b) => scores[a] - scores[b]);
  } else {
    indices.sort((a, b) => scores[b] - scores[a]);
  }

  const k = Math.min(limit, n);
  const results: ResultItem[] = [];
  for (let i = 0; i < n && results.length < k; i++) {
    const idx = indices[i];
    const similarity = scores[idx];
    if (!inRange(similarity, minSimilarity, maxSimilarity)) continue;
    results.push({ key: resolveKey(idx), similarity });
  }
  return results;
}

/**
 * Sort scores and return a lazy iterable over the results.
 * Keys are resolved only when each item is consumed, saving allocations
 * when the caller iterates partially (e.g., pagination).
 *
 * Results outside [minSimilarity, maxSimilarity] are skipped.
 *
 * The returned iterable is re-iterable — each call to [Symbol.iterator]()
 * produces a fresh cursor over the same pre-sorted data.
 */
export function iterableResults(
  scores: Float32Array,
  resolveKey: KeyResolver,
  options: ResultOptions,
): Iterable<ResultItem> {
  const { limit, order, minSimilarity, maxSimilarity } = options;
  const n = scores.length;
  if (n === 0) return [];

  const indices = new Uint32Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;

  if (order === "ascend") {
    indices.sort((a, b) => scores[a] - scores[b]);
  } else {
    indices.sort((a, b) => scores[b] - scores[a]);
  }

  return {
    [Symbol.iterator](): Iterator<ResultItem> {
      let i = 0;
      let emitted = 0;
      return {
        next(): IteratorResult<ResultItem> {
          while (i < n && emitted < limit) {
            const idx = indices[i++];
            const similarity = scores[idx];
            if (!inRange(similarity, minSimilarity, maxSimilarity)) continue;
            emitted++;
            return {
              done: false,
              value: { key: resolveKey(idx), similarity },
            };
          }
          return { done: true, value: undefined };
        },
      };
    },
  };
}
