/**
 * LAZY RESULT SET
 *
 * Holds pointers to sorted TypedArrays. Prevents JS heap overflow when K is massive.
 * Strings are only instantiated from the Lexicon when explicitly requested.
 */

export interface ResultItem {
  key: string;
  score: number;
}

export type KeyResolver = (index: number) => string;

export class ResultSet {
  /** Total number of results */
  readonly length: number;

  /**
   * Sorted indices into the original database (by descending score).
   * sortedIndices[0] is the index of the best match.
   */
  private readonly sortedIndices: Uint32Array;

  /** Raw scores array (not sorted, indexed by original DB position) */
  private readonly scores: Float32Array;

  /** Function to lazily resolve key from the slot index */
  private readonly resolveKey: KeyResolver;

  constructor(
    scores: Float32Array,
    sortedIndices: Uint32Array,
    resolveKey: KeyResolver,
    topK: number,
  ) {
    this.scores = scores;
    this.sortedIndices = sortedIndices;
    this.resolveKey = resolveKey;
    this.length = Math.min(topK, sortedIndices.length);
  }

  /**
   * Sort scores and return a ResultSet with lazy key resolution.
   *
   * @param scores - Float32Array of scores (one per DB vector)
   * @param resolveKey - Function to resolve key by original index
   * @param topK - Maximum number of results to include
   */
  static fromScores(
    scores: Float32Array,
    resolveKey: KeyResolver,
    topK: number,
  ): ResultSet {
    const n = scores.length;

    // Create index array for sorting
    const indices = new Uint32Array(n);
    for (let i = 0; i < n; i++) indices[i] = i;

    // Sort indices by descending score
    indices.sort((a, b) => scores[b] - scores[a]);

    return new ResultSet(scores, indices, resolveKey, topK);
  }

  /** Fetch a single result by its rank (0 is best match) */
  get(rank: number): ResultItem {
    if (rank < 0 || rank >= this.length) {
      throw new RangeError(`Rank ${rank} out of bounds [0, ${this.length})`);
    }
    const dbIndex = this.sortedIndices[rank];
    return {
      key: this.resolveKey(dbIndex),
      score: this.scores[dbIndex],
    };
  }

  /** Helper for UI pagination. Instantiates strings only for the requested page. */
  getPage(page: number, pageSize: number): ResultItem[] {
    const start = page * pageSize;
    const end = Math.min(start + pageSize, this.length);
    const results: ResultItem[] = [];
    for (let i = start; i < end; i++) {
      results.push(this.get(i));
    }
    return results;
  }
}
