import { readFileSync } from "fs";
import { resolve } from "path";
import { bench, describe } from "vitest";
import { normalize as jsNormalize, searchAll as jsSearchAll } from "../compute";
import { compileWatToWasm, instantiateWasm } from "../wasm-compute";

const watSource = readFileSync(resolve(__dirname, "../simd.wat"), "utf-8");

/**
 * Benchmarks comparing JS vs WASM SIMD performance for vector operations.
 */
describe("normalize benchmark", async () => {
  const dimensions = 1536;
  const vec = new Float32Array(dimensions);
  for (let i = 0; i < dimensions; i++) vec[i] = Math.random();

  const wasmBinary = await compileWatToWasm(watSource);
  const memory = new WebAssembly.Memory({ initial: 1 });
  const wasm = await instantiateWasm(wasmBinary, memory);
  const ptr = 0;

  bench("JS normalize (1536 dims)", () => {
    const v = new Float32Array(vec);
    jsNormalize(v);
  });

  bench("WASM SIMD normalize (1536 dims)", () => {
    new Float32Array(memory.buffer, ptr, dimensions).set(vec);
    wasm.normalize(ptr, dimensions);
  });
});

const dimensions = 1536;
const dbSizes = [100, 1000, 10000];

for (const dbSize of dbSizes) {
  describe(`searchAll benchmark (${dbSize} vectors)`, async () => {
    // Prepare data
    const query = new Float32Array(dimensions);
    for (let i = 0; i < dimensions; i++) query[i] = Math.random();
    jsNormalize(query);

    const db = new Float32Array(dbSize * dimensions);
    for (let i = 0; i < db.length; i++) db[i] = Math.random();

    const jsScores = new Float32Array(dbSize);

    bench(`JS searchAll (${dbSize} vectors × ${dimensions} dims)`, () => {
      jsSearchAll(query, db, jsScores, dbSize, dimensions);
    });

    // WASM benchmark
    const wasmBinary = await compileWatToWasm(watSource);
    const totalBytes = dimensions * 4 + dbSize * dimensions * 4 + dbSize * 4;
    const pages = Math.ceil(totalBytes / 65536);
    const memory = new WebAssembly.Memory({ initial: pages });
    const wasm = await instantiateWasm(wasmBinary, memory);

    const queryPtr = 0;
    const dbPtr = dimensions * 4;
    const scoresPtr = dbPtr + dbSize * dimensions * 4;

    new Float32Array(memory.buffer, queryPtr, dimensions).set(query);
    new Float32Array(memory.buffer, dbPtr, dbSize * dimensions).set(db);

    bench(`WASM SIMD searchAll (${dbSize} vectors × ${dimensions} dims)`, () => {
      wasm.search_all(queryPtr, dbPtr, scoresPtr, dbSize, dimensions);
    });
  });
}
