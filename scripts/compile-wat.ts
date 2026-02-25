/**
 * Build script: Compiles simd.wat to a WASM binary and generates
 * a TypeScript module with the embedded binary as a base64 string.
 */

import { readFileSync, writeFileSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

async function main() {
  const wabt = await import("wabt");
  const wabtModule = await wabt.default();

  const watPath = resolve(__dirname, "../src/lib/simd.wat");
  const outPath = resolve(__dirname, "../src/lib/simd-binary.ts");

  const watSource = readFileSync(watPath, "utf-8");
  const parsed = wabtModule.parseWat("simd.wat", watSource, { simd: true });
  parsed.resolveNames();
  parsed.validate();
  const { buffer } = parsed.toBinary({});
  parsed.destroy();

  const base64 = Buffer.from(buffer).toString("base64");

  const tsModule = `// AUTO-GENERATED - Do not edit. Run: npx tsx scripts/compile-wat.ts
const SIMD_WASM_BASE64 = "${base64}";

export function getSimdWasmBinary(): Uint8Array {
  const binaryString = atob(SIMD_WASM_BASE64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}
`;

  writeFileSync(outPath, tsModule);
  console.log(`Compiled ${watPath} -> ${outPath} (${buffer.byteLength} bytes WASM)`);
}

main().catch(console.error);
