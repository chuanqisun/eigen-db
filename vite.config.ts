import { resolve } from "path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, "src/lib/index.ts"),
      name: "EigenDB",
      fileName: "eigen-db",
    },
    sourcemap: true,
  },
  test: {
    include: ["src/**/*.test.ts"],
  },
});
