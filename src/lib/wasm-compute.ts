/**
 * WASM SIMD compute layer.
 * Compiles the hand-written WAT module and provides typed wrappers
 * that operate on shared WebAssembly.Memory.
 */

export interface WasmExports {
  normalize(ptr: number, dimensions: number): void;
  search_all(queryPtr: number, dbPtr: number, scoresPtr: number, dbSize: number, dimensions: number): void;
}

/**
 * Compiles a WAT string into a WASM module using the wabt library.
 * This is used at build time or test time; at runtime, the pre-compiled binary is used.
 */
export async function compileWatToWasm(watSource: string): Promise<Uint8Array> {
  const wabt = await import("wabt");
  const wabtModule = await wabt.default();
  const parsed = wabtModule.parseWat("simd.wat", watSource, {
    simd: true,
  });
  parsed.resolveNames();
  parsed.validate();
  const { buffer } = parsed.toBinary({});
  parsed.destroy();
  return new Uint8Array(buffer);
}

/**
 * Instantiates a WASM module with the given memory and returns typed exports.
 */
export async function instantiateWasm(
  wasmBinary: Uint8Array,
  memory: WebAssembly.Memory,
): Promise<WasmExports> {
  const importObject = { env: { memory } };
  const result = await WebAssembly.instantiate(wasmBinary, importObject);
  // WebAssembly.instantiate with a buffer returns { instance, module }
  const instance = (result as unknown as { instance: WebAssembly.Instance }).instance;
  return instance.exports as unknown as WasmExports;
}
