/**
 * Thrown when the database exceeds the 4GB WebAssembly 32-bit memory limit,
 * or the browser's available RAM.
 */
export class VectorCapacityExceededError extends Error {
  constructor(maxVectors: number) {
    super(`Capacity exceeded. Max vectors for this dimension size is ~${maxVectors}.`);
    this.name = "VectorCapacityExceededError";
  }
}
