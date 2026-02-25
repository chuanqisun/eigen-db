import { describe, it, expect } from "vitest";
import {
  encodeLexicon,
  decodeLexicon,
  decodeLexiconAt,
  buildLexiconIndex,
  decodeLexiconAtOffset,
} from "../lexicon";

describe("Lexicon", () => {
  describe("encodeLexicon / decodeLexicon", () => {
    it("round-trips an array of strings", () => {
      const texts = ["hello", "world", "vector search"];
      const encoded = encodeLexicon(texts);
      const decoded = decodeLexicon(encoded);
      expect(decoded).toEqual(texts);
    });

    it("handles empty array", () => {
      const encoded = encodeLexicon([]);
      expect(encoded.byteLength).toBe(0);
      expect(decodeLexicon(encoded)).toEqual([]);
    });

    it("handles empty strings", () => {
      const texts = ["", "hello", ""];
      const encoded = encodeLexicon(texts);
      const decoded = decodeLexicon(encoded);
      expect(decoded).toEqual(texts);
    });

    it("handles unicode strings", () => {
      const texts = ["日本語", "émojis 🎉", "Ñoño"];
      const encoded = encodeLexicon(texts);
      const decoded = decodeLexicon(encoded);
      expect(decoded).toEqual(texts);
    });

    it("encodes with correct binary format", () => {
      const texts = ["hi"];
      const encoded = encodeLexicon(texts);
      // "hi" = 2 bytes UTF-8, so total = 4 (length) + 2 (data) = 6 bytes
      expect(encoded.byteLength).toBe(6);
      const view = new DataView(encoded.buffer);
      expect(view.getUint32(0, true)).toBe(2); // length prefix
    });
  });

  describe("decodeLexiconAt", () => {
    it("accesses entries by index", () => {
      const texts = ["alpha", "beta", "gamma"];
      const encoded = encodeLexicon(texts);

      expect(decodeLexiconAt(encoded, 0)).toBe("alpha");
      expect(decodeLexiconAt(encoded, 1)).toBe("beta");
      expect(decodeLexiconAt(encoded, 2)).toBe("gamma");
    });
  });

  describe("buildLexiconIndex / decodeLexiconAtOffset", () => {
    it("builds offset index for O(1) access", () => {
      const texts = ["one", "two", "three"];
      const encoded = encodeLexicon(texts);
      const index = buildLexiconIndex(encoded);

      expect(index.length).toBe(3);
      expect(index[0]).toBe(0);

      for (let i = 0; i < texts.length; i++) {
        expect(decodeLexiconAtOffset(encoded, index[i])).toBe(texts[i]);
      }
    });

    it("handles single entry", () => {
      const encoded = encodeLexicon(["solo"]);
      const index = buildLexiconIndex(encoded);
      expect(index.length).toBe(1);
      expect(decodeLexiconAtOffset(encoded, index[0])).toBe("solo");
    });
  });

  describe("incremental append", () => {
    it("supports appending encoded chunks", () => {
      const part1 = encodeLexicon(["first", "second"]);
      const part2 = encodeLexicon(["third"]);

      // Simulate append: concatenate buffers
      const combined = new Uint8Array(part1.byteLength + part2.byteLength);
      combined.set(part1, 0);
      combined.set(part2, part1.byteLength);

      const decoded = decodeLexicon(combined);
      expect(decoded).toEqual(["first", "second", "third"]);
    });
  });
});
