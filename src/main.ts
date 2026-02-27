import { DB, OPFSStorageProvider } from "./lib/index";
import "./style.css";

const DIMENSIONS = 128;
const DB_DIR = "eigen-db-demo";

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
  const status = document.querySelector<HTMLDivElement>("#status")!;
  const resultsDiv = document.querySelector<HTMLDivElement>("#results")!;
  const exportBtn = document.querySelector<HTMLButtonElement>("#export-btn")!;
  const importFile = document.querySelector<HTMLInputElement>("#import-file")!;
  const loadBtn = document.querySelector<HTMLButtonElement>("#load-btn")!;
  const deleteBtn = document.querySelector<HTMLButtonElement>("#delete-btn")!;

  let db: DB | null = null;

  function supportsOPFS(): boolean {
    return typeof navigator !== "undefined" && !!navigator.storage?.getDirectory;
  }

  function createStorage() {
    return new OPFSStorageProvider(DB_DIR);
  }

  async function openDbWithTimer(reason: string): Promise<void> {
    const start = performance.now();
    db = await DB.open({
      dimensions: DIMENSIONS,
      storage: createStorage(),
    });
    const elapsed = (performance.now() - start).toFixed(1);
    log(`${reason}: loaded ${db.size} records in ${elapsed} ms`);
    updateExportButton();
  }

  function log(msg: string) {
    status.textContent = msg;
  }

  function updateExportButton() {
    exportBtn.disabled = !db || db.size === 0;
  }

  if (!supportsOPFS()) {
    const msg = "OPFS is not supported in this browser context (requires secure context).";
    log(msg);
    loadBtn.disabled = true;
    deleteBtn.disabled = true;
    exportBtn.disabled = true;
    return;
  }

  // Load any existing persisted DB on startup
  await openDbWithTimer("Startup");

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
      storage: createStorage(),
    });
    db.setMany(entries);
    await db.flush();
    const elapsed = (performance.now() - start).toFixed(1);

    log(`Indexed and flushed ${db.size} records in ${elapsed} ms`);
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
    const results = db.query(queryVec, { limit: topK });
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
          storage: createStorage(),
        });
      }

      const start = performance.now();
      await db.import(file.stream());
      const elapsed = (performance.now() - start).toFixed(1);

      log(`Imported ${db.size} records from ${file.name} in ${elapsed} ms.`);
      resultsDiv.innerHTML = "";
      updateExportButton();
    } catch (err) {
      log(`Import failed: ${err instanceof Error ? err.message : String(err)}`);
    }

    // Reset file input so same file can be re-imported
    importFile.value = "";
  });

  // Load persisted DB on demand
  loadBtn.addEventListener("click", async () => {
    log("Loading from OPFS…");
    try {
      await openDbWithTimer("Load");
      resultsDiv.innerHTML = "";
    } catch (err) {
      log(`Load failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  });

  // Delete persisted DB and clear active state
  deleteBtn.addEventListener("click", async () => {
    log("Deleting DB from OPFS…");
    try {
      const start = performance.now();
      if (!db) {
        db = await DB.open({
          dimensions: DIMENSIONS,
          storage: createStorage(),
        });
      }
      await db.clear();
      db = await DB.open({
        dimensions: DIMENSIONS,
        storage: createStorage(),
      });
      const elapsed = (performance.now() - start).toFixed(1);

      resultsDiv.innerHTML = "";
      updateExportButton();
      log(`Deleted DB and reinitialized empty store in ${elapsed} ms.`);
    } catch (err) {
      log(`Delete failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  });

  log("Ready. OPFS-backed DB is loaded.");
}

main();
