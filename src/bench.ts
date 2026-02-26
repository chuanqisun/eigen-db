/**
 * Browser benchmark: JS vs WASM SIMD performance comparison.
 * Accessible at /bench.html when running `npm run dev`.
 */
import { normalize as jsNormalize, searchAll as jsSearchAll } from "./lib/compute";
import { getSimdWasmBinary } from "./lib/simd-binary";
import { instantiateWasm } from "./lib/wasm-compute";

const DIMENSIONS = 1536;
const WARMUP_ITERATIONS = 50;
const DB_SIZES = [100, 1000, 10000];

interface BenchResult {
  name: string;
  jsOps: number;
  wasmOps: number;
  speedup: number;
}

/** Measure operations per second for a function. */
function measureOps(fn: () => void, minTimeMs = 500): number {
  // Warmup
  for (let i = 0; i < WARMUP_ITERATIONS; i++) fn();

  // Timed run: keep calling until minTimeMs elapsed, count iterations
  let iterations = 0;
  const start = performance.now();
  while (performance.now() - start < minTimeMs) {
    fn();
    iterations++;
  }
  const elapsed = performance.now() - start;
  return (iterations / elapsed) * 1000;
}

async function runBenchmarks(onStatus: (msg: string) => void): Promise<BenchResult[]> {
  const results: BenchResult[] = [];

  // Set up WASM
  onStatus("Instantiating WASM module…");
  const wasmBinary = getSimdWasmBinary();

  // --- Normalize benchmark ---
  onStatus("Benchmarking normalize…");
  await tick();

  const vec = new Float32Array(DIMENSIONS);
  for (let i = 0; i < DIMENSIONS; i++) vec[i] = Math.random();

  const normMemory = new WebAssembly.Memory({ initial: 1 });
  const normWasm = await instantiateWasm(wasmBinary, normMemory);

  const jsNormOps = measureOps(() => {
    const v = new Float32Array(vec);
    jsNormalize(v);
  });

  const wasmNormOps = measureOps(() => {
    new Float32Array(normMemory.buffer, 0, DIMENSIONS).set(vec);
    normWasm.normalize(0, DIMENSIONS);
  });

  results.push({
    name: `normalize (${DIMENSIONS} dims)`,
    jsOps: jsNormOps,
    wasmOps: wasmNormOps,
    speedup: wasmNormOps / jsNormOps,
  });

  // --- searchAll benchmarks at different DB sizes ---
  for (const dbSize of DB_SIZES) {
    onStatus(`Benchmarking searchAll (${dbSize.toLocaleString()} vectors)…`);
    await tick();

    const query = new Float32Array(DIMENSIONS);
    for (let i = 0; i < DIMENSIONS; i++) query[i] = Math.random();
    jsNormalize(query);

    const db = new Float32Array(dbSize * DIMENSIONS);
    for (let i = 0; i < db.length; i++) db[i] = Math.random();

    const jsScores = new Float32Array(dbSize);

    const jsOps = measureOps(() => {
      jsSearchAll(query, db, jsScores, dbSize, DIMENSIONS);
    });

    // WASM setup
    const totalBytes = DIMENSIONS * 4 + dbSize * DIMENSIONS * 4 + dbSize * 4;
    const pages = Math.ceil(totalBytes / 65536);
    const memory = new WebAssembly.Memory({ initial: pages });
    const wasm = await instantiateWasm(wasmBinary, memory);

    const queryPtr = 0;
    const dbPtr = DIMENSIONS * 4;
    const scoresPtr = dbPtr + dbSize * DIMENSIONS * 4;

    new Float32Array(memory.buffer, queryPtr, DIMENSIONS).set(query);
    new Float32Array(memory.buffer, dbPtr, dbSize * DIMENSIONS).set(db);

    const wasmOps = measureOps(() => {
      wasm.search_all(queryPtr, dbPtr, scoresPtr, dbSize, DIMENSIONS);
    });

    results.push({
      name: `searchAll (${dbSize.toLocaleString()} × ${DIMENSIONS} dims)`,
      jsOps,
      wasmOps,
      speedup: wasmOps / jsOps,
    });
  }

  return results;
}

function renderResults(results: BenchResult[]): string {
  let html = `<table>
    <thead><tr>
      <th>Operation</th><th>JS (ops/s)</th><th>WASM SIMD (ops/s)</th><th>Speedup</th>
    </tr></thead><tbody>`;

  for (const r of results) {
    html += `<tr>
      <td>${r.name}</td>
      <td>${formatNumber(r.jsOps)}</td>
      <td>${formatNumber(r.wasmOps)}</td>
      <td class="speedup">${r.speedup.toFixed(1)}×</td>
    </tr>`;
  }

  html += `</tbody></table>`;
  return html;
}

function formatNumber(n: number): string {
  if (n >= 1000) return Math.round(n).toLocaleString();
  if (n >= 1) return n.toFixed(1);
  return n.toFixed(3);
}

/** Yield to the event loop so the UI can update. */
function tick(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

// --- UI wiring ---
const btn = document.getElementById("run-btn") as HTMLButtonElement;
const statusEl = document.getElementById("status")!;
const resultsEl = document.getElementById("results")!;
const envInfo = document.getElementById("env-info")!;

envInfo.textContent = `${navigator.userAgent}`;

btn.addEventListener("click", async () => {
  btn.disabled = true;
  resultsEl.innerHTML = "";

  try {
    const results = await runBenchmarks((msg) => {
      statusEl.textContent = msg;
    });
    statusEl.textContent = "Done!";
    resultsEl.innerHTML = renderResults(results);
  } catch (err) {
    statusEl.textContent = `Error: ${err instanceof Error ? err.message : err}`;
  } finally {
    btn.disabled = false;
  }
});
