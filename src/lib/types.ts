/**
 * Contract for external embedding providers (OpenAI, HuggingFace, local WebGPU, etc.)
 */
export type EmbeddingFunction = (texts: string[]) => Promise<Float32Array[]>;

export interface EngineConfig {
  /** Name of the OPFS directory */
  name: string;
  /** Vector dimensions (e.g., 1536 for OpenAI text-embedding-3-small) */
  dimensions: number;
  /** User-provided embedding function */
  embedder: EmbeddingFunction;
}
