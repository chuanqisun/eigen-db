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
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        bench: resolve(__dirname, "bench.html"),
      },
    },
  },
  test: {
    include: ["src/**/*.test.ts"],
  },
  base: "/eigen-db/",
});
