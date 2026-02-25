/**
 * Pure JavaScript compute functions for vector operations.
 * These serve as the reference implementation and fallback when WASM SIMD is unavailable.
 */

/**
 * Normalizes a vector in-place to unit length.
 * After normalization, cosine similarity reduces to a simple dot product.
 */
export function normalize(vec: Float32Array): void {
  let sumSq = 0;
  for (let i = 0; i < vec.length; i++) {
    sumSq += vec[i] * vec[i];
  }
  const mag = Math.sqrt(sumSq);
  if (mag === 0) return;
  const invMag = 1 / mag;
  for (let i = 0; i < vec.length; i++) {
    vec[i] *= invMag;
  }
}

/**
 * Computes dot products of query against all vectors in the database.
 * Writes scores to the output array.
 *
 * @param query - Normalized query vector (length = dimensions)
 * @param db - Contiguous flat array of normalized vectors (length = dbSize * dimensions)
 * @param scores - Output array for dot product scores (length = dbSize)
 * @param dbSize - Number of vectors in the database
 * @param dimensions - Dimensionality of each vector
 */
export function searchAll(
  query: Float32Array,
  db: Float32Array,
  scores: Float32Array,
  dbSize: number,
  dimensions: number,
): void {
  for (let i = 0; i < dbSize; i++) {
    let dot = 0;
    const offset = i * dimensions;
    for (let j = 0; j < dimensions; j++) {
      dot += query[j] * db[offset + j];
    }
    scores[i] = dot;
  }
}
