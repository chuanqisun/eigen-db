import { DB, InMemoryStorageProvider } from "./lib/index";
import "./style.css";

const DIMENSIONS = 128;

/** Simple character-frequency embedder for demo purposes */
function embed(text: string): number[] {
  const vec = new Array<number>(DIMENSIONS).fill(0);
  const lower = text.toLowerCase();
  for (let i = 0; i < lower.length; i++) {
    vec[lower.charCodeAt(i) % DIMENSIONS] += 1;
  }
  return vec;
}

/** Generate random strings of a given length */
function randomString(len: number): string {
  const chars = "abcdefghijklmnopqrstuvwxyz0123456789";
  let s = "";
  for (let i = 0; i < len; i++) {
    s += chars[(Math.random() * chars.length) | 0];
  }
  return s;
}

const EIGEN_DB_MIME = "application/x-eigen-db";

async function main() {
  const app = document.querySelector<HTMLDivElement>("#app")!;
  app.innerHTML = `
    <h1>web-vector-base demo</h1>
    <p>In-browser vector search powered by WASM SIMD. Uses a simple character-frequency embedder.</p>

    <fieldset>
      <legend>Generate dataset</legend>
      <label>Number of records: <input id="gen-count" type="number" value="1000" min="1" max="100000" /></label>
      <button id="gen-btn">Generate &amp; index</button>
    </fieldset>

    <fieldset>
      <legend>Search</legend>
      <label>Query: <input id="search-input" type="text" value="abc" /></label>
      <label>Top K: <input id="top-k" type="number" value="10" min="1" max="1000" /></label>
      <button id="search-btn">Search</button>
    </fieldset>

    <fieldset>
      <legend>Export / Import</legend>
      <button id="export-btn" disabled>Export database (.bin)</button>
      <label>Import: <input id="import-file" type="file" accept=".bin" /></label>
    </fieldset>

    <div id="status">Initializing…</div>
    <div id="results"></div>
  `;

  const status = document.querySelector<HTMLDivElement>("#status")!;
  const resultsDiv = document.querySelector<HTMLDivElement>("#results")!;
  const exportBtn = document.querySelector<HTMLButtonElement>("#export-btn")!;
  const importFile = document.querySelector<HTMLInputElement>("#import-file")!;

  let db: DB | null = null;

  function log(msg: string) {
    status.textContent = msg;
  }

  function updateExportButton() {
    exportBtn.disabled = !db || db.size === 0;
  }

  // Generate dataset
  document.querySelector("#gen-btn")!.addEventListener("click", async () => {
    const count = parseInt((document.querySelector("#gen-count") as HTMLInputElement).value, 10);
    if (!count || count < 1) return;

    log(`Generating ${count} random strings…`);
    await new Promise((r) => setTimeout(r, 0)); // let UI update

    const entries: [string, number[]][] = [];
    for (let i = 0; i < count; i++) {
      const text = randomString(8 + ((Math.random() * 24) | 0));
      entries.push([text, embed(text)]);
    }

    log(`Indexing ${count} records…`);
    await new Promise((r) => setTimeout(r, 0));

    const start = performance.now();
    db = await DB.open({
      dimensions: DIMENSIONS,
      storage: new InMemoryStorageProvider(),
    });
    db.setMany(entries);
    const elapsed = (performance.now() - start).toFixed(1);

    log(`Indexed ${db.size} records in ${elapsed} ms`);
    resultsDiv.innerHTML = "";
    updateExportButton();
  });

  // Search
  document.querySelector("#search-btn")!.addEventListener("click", async () => {
    if (!db || db.size === 0) {
      log("Generate a dataset first.");
      return;
    }

    const query = (document.querySelector("#search-input") as HTMLInputElement).value.trim();
    const topK = parseInt((document.querySelector("#top-k") as HTMLInputElement).value, 10);
    if (!query) return;

    log("Searching…");
    const start = performance.now();
    const queryVec = embed(query);
    const results = db.query(queryVec, { topK });
    const elapsed = (performance.now() - start).toFixed(1);

    log(`Search: ${elapsed} ms — ${results.length} results from ${db.size} records (top ${topK})`);

    let html = "<table><thead><tr><th>#</th><th>Similarity</th><th>Key</th></tr></thead><tbody>";
    for (let i = 0; i < results.length; i++) {
      const r = results[i];
      html += `<tr><td>${i + 1}</td><td>${r.similarity.toFixed(4)}</td><td>${r.key}</td></tr>`;
    }
    html += "</tbody></table>";
    resultsDiv.innerHTML = html;
  });

  // Export — download the database as a .bin file
  exportBtn.addEventListener("click", async () => {
    if (!db || db.size === 0) {
      log("No data to export.");
      return;
    }

    log("Exporting…");
    try {
      const stream = await db.export();
      const response = new Response(stream, {
        headers: { "Content-Type": EIGEN_DB_MIME },
      });
      const blob = await response.blob();

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "database.bin";
      a.click();
      URL.revokeObjectURL(url);

      log(`Exported ${db.size} records.`);
    } catch (err) {
      log(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  });

  // Import — upload a .bin file to replace the database
  importFile.addEventListener("change", async () => {
    const file = importFile.files?.[0];
    if (!file) return;

    log(`Importing ${file.name}…`);
    try {
      if (!db) {
        db = await DB.open({
          dimensions: DIMENSIONS,
          storage: new InMemoryStorageProvider(),
        });
      }

      await db.import(file.stream());

      log(`Imported ${db.size} records from ${file.name}.`);
      resultsDiv.innerHTML = "";
      updateExportButton();
    } catch (err) {
      log(`Import failed: ${err instanceof Error ? err.message : String(err)}`);
    }

    // Reset file input so same file can be re-imported
    importFile.value = "";
  });

  log("Ready. Generate a dataset to begin.");
}

main();
