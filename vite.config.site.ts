import { resolve } from "path";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        bench: resolve(__dirname, "bench.html"),
      },
    },
  },
  base: "/eigen-db/",
});
