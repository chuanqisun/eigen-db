/**
 * Lexicon: length-prefixed UTF-8 encoding for text strings.
 *
 * Format: Each entry is [4-byte uint32 length][UTF-8 bytes]
 * This allows efficient sequential reading and appending.
 */

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/**
 * Encodes an array of strings into a length-prefixed binary format.
 */
export function encodeLexicon(texts: string[]): Uint8Array {
  const encoded = texts.map((t) => encoder.encode(t));
  const totalSize = encoded.reduce((sum, e) => sum + 4 + e.byteLength, 0);

  const buffer = new ArrayBuffer(totalSize);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  let offset = 0;

  for (const e of encoded) {
    view.setUint32(offset, e.byteLength, true); // little-endian
    offset += 4;
    bytes.set(e, offset);
    offset += e.byteLength;
  }

  return bytes;
}

/**
 * Decodes all strings from a length-prefixed binary buffer.
 */
export function decodeLexicon(data: Uint8Array): string[] {
  const result: string[] = [];
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let offset = 0;

  while (offset < data.byteLength) {
    const len = view.getUint32(offset, true);
    offset += 4;
    const text = decoder.decode(data.subarray(offset, offset + len));
    result.push(text);
    offset += len;
  }

  return result;
}

/**
 * Decodes a single string at a given index from the lexicon.
 * Returns the string and the byte offset of the next entry.
 */
export function decodeLexiconAt(data: Uint8Array, index: number): string {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let offset = 0;

  for (let i = 0; i < index; i++) {
    const len = view.getUint32(offset, true);
    offset += 4 + len;
  }

  const len = view.getUint32(offset, true);
  offset += 4;
  return decoder.decode(data.subarray(offset, offset + len));
}

/**
 * Builds an index of byte offsets for each entry in the lexicon.
 * Enables O(1) access to any entry by index.
 */
export function buildLexiconIndex(data: Uint8Array): Uint32Array {
  const offsets: number[] = [];
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let offset = 0;

  while (offset < data.byteLength) {
    offsets.push(offset);
    const len = view.getUint32(offset, true);
    offset += 4 + len;
  }

  return new Uint32Array(offsets);
}

/**
 * Decodes a string at a given byte offset in the lexicon.
 */
export function decodeLexiconAtOffset(data: Uint8Array, byteOffset: number): string {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const len = view.getUint32(byteOffset, true);
  return decoder.decode(data.subarray(byteOffset + 4, byteOffset + 4 + len));
}
