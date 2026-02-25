import { resolve } from "path";
import { defineConfig } from "vitest/config";

export default defineConfig({
  build: {
    lib: {
      entry: resolve(__dirname, "src/lib/index.ts"),
      name: "WebVectorBase",
      fileName: "web-vector-base",
    },
  },
  test: {
    include: ["src/**/*.test.ts"],
  },
});
