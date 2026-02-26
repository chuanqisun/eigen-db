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

    <div id="status">Initializing…</div>
    <div id="results"></div>
  `;

  const status = document.querySelector<HTMLDivElement>("#status")!;
  const resultsDiv = document.querySelector<HTMLDivElement>("#results")!;

  let db: DB | null = null;

  function log(msg: string) {
    status.textContent = msg;
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

  log("Ready. Generate a dataset to begin.");
}

main();
