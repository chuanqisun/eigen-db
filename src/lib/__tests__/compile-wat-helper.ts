/**
 * Test-only helper: compiles WAT source to WASM binary using wabt.
 * Kept out of the lib source so the `wabt` dependency is invisible to the library's type surface.
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
