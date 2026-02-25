import "./style.css";
import { VectorEngine } from "./lib/index";

/**
 * Demo: In-browser Vector Search Engine
 *
 * This demo uses a simple character-frequency embedder to demonstrate
 * the VectorEngine API. In production, you'd use OpenAI, HuggingFace, etc.
 */

const DIMENSIONS = 128;

/** Simple character-frequency embedder for demo purposes */
const demoEmbedder = async (texts: string[]): Promise<Float32Array[]> => {
  return texts.map((text) => {
    const vec = new Float32Array(DIMENSIONS);
    const lower = text.toLowerCase();
    for (let i = 0; i < lower.length; i++) {
      vec[lower.charCodeAt(i) % DIMENSIONS] += 1;
    }
    return vec;
  });
};

async function main() {
  const app = document.querySelector<HTMLDivElement>("#app")!;
  app.innerHTML = `
    <div style="max-width: 800px; margin: 0 auto; padding: 2rem; font-family: system-ui;">
      <h1>🔍 Web Vector Base Demo</h1>
      <p>In-browser vector search powered by WASM SIMD</p>

      <div style="margin: 1rem 0;">
        <h3>Add Texts</h3>
        <textarea id="add-input" rows="4" style="width: 100%; box-sizing: border-box;"
          placeholder="Enter texts (one per line) to add to the database...">WebAssembly SIMD provides near-native performance in the browser.
Origin Private File System allows high-performance local storage.
Cosine similarity measures the angle between two vectors.
TypeScript adds static typing to JavaScript.
Vector databases enable semantic search capabilities.
Machine learning models can run directly in the browser.</textarea>
        <button id="add-btn" style="margin-top: 0.5rem;">Add to Database</button>
      </div>

      <div style="margin: 1rem 0;">
        <h3>Search</h3>
        <input id="search-input" type="text" style="width: 100%; box-sizing: border-box;"
          placeholder="Enter search query..." value="How fast is WASM?" />
        <button id="search-btn" style="margin-top: 0.5rem;">Search</button>
      </div>

      <div id="status" style="margin: 1rem 0; padding: 0.5rem; background: #f0f0f0; border-radius: 4px;">
        Initializing...
      </div>

      <div id="results"></div>
    </div>
  `;

  const status = document.querySelector<HTMLDivElement>("#status")!;
  const resultsDiv = document.querySelector<HTMLDivElement>("#results")!;

  try {
    const db = await VectorEngine.open({
      name: "demo-vector-db",
      dimensions: DIMENSIONS,
      embedder: demoEmbedder,
    });

    status.textContent = `Engine ready. Database has ${db.size} records.`;

    document.querySelector("#add-btn")!.addEventListener("click", async () => {
      const input = document.querySelector<HTMLTextAreaElement>("#add-input")!;
      const texts = input.value
        .split("\n")
        .map((t) => t.trim())
        .filter((t) => t.length > 0);

      if (texts.length === 0) return;

      status.textContent = `Adding ${texts.length} texts...`;
      const start = performance.now();
      await db.add(texts);
      const elapsed = (performance.now() - start).toFixed(1);
      status.textContent = `Added ${texts.length} texts in ${elapsed}ms. Database now has ${db.size} records.`;
      input.value = "";
    });

    document.querySelector("#search-btn")!.addEventListener("click", async () => {
      const input = document.querySelector<HTMLInputElement>("#search-input")!;
      const query = input.value.trim();
      if (!query || db.size === 0) {
        resultsDiv.innerHTML = "<p>Add some texts first, then search.</p>";
        return;
      }

      status.textContent = "Searching...";
      const start = performance.now();
      const results = await db.search(query, 10);
      const elapsed = (performance.now() - start).toFixed(1);
      status.textContent = `Search completed in ${elapsed}ms. Found ${results.length} results.`;

      const page = results.getPage(0, 10);
      let html = `<h3>Results for "${query}"</h3>`;
      html += `<table style="width: 100%; border-collapse: collapse;">`;
      html += `<thead><tr style="text-align: left; border-bottom: 2px solid #333;">`;
      html += `<th style="padding: 0.5rem;">Rank</th>`;
      html += `<th style="padding: 0.5rem;">Score</th>`;
      html += `<th style="padding: 0.5rem;">Text</th>`;
      html += `</tr></thead><tbody>`;
      for (let i = 0; i < page.length; i++) {
        const r = page[i];
        html += `<tr style="border-bottom: 1px solid #ddd;">`;
        html += `<td style="padding: 0.5rem;">${i + 1}</td>`;
        html += `<td style="padding: 0.5rem;">${r.score.toFixed(4)}</td>`;
        html += `<td style="padding: 0.5rem;">${r.text}</td>`;
        html += `</tr>`;
      }
      html += `</tbody></table>`;
      resultsDiv.innerHTML = html;
    });
  } catch (e) {
    status.textContent = `Error: ${e instanceof Error ? e.message : String(e)}`;
  }
}

main();
