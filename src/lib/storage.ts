/**
 * Storage abstraction for append-only binary files.
 * Supports OPFS for browser and in-memory for testing.
 */

export interface StorageProvider {
  /** Read the entire contents of a file. Returns empty Uint8Array if file doesn't exist. */
  readAll(fileName: string): Promise<Uint8Array>;

  /** Append data to a file (creates if it doesn't exist). */
  append(fileName: string, data: Uint8Array): Promise<void>;

  /** Write data to a file, replacing all existing content. */
  write(fileName: string, data: Uint8Array): Promise<void>;

  /** Delete the storage directory and all files. */
  destroy(): Promise<void>;
}

/**
 * OPFS-backed storage provider for browser environments.
 * Uses Origin Private File System for high-performance persistent storage.
 */
export class OPFSStorageProvider implements StorageProvider {
  private dirHandle: FileSystemDirectoryHandle | null = null;
  private dirName: string;

  constructor(dirName: string) {
    this.dirName = dirName;
  }

  private async getDir(): Promise<FileSystemDirectoryHandle> {
    if (!this.dirHandle) {
      const root = await navigator.storage.getDirectory();
      this.dirHandle = await root.getDirectoryHandle(this.dirName, { create: true });
    }
    return this.dirHandle;
  }

  async readAll(fileName: string): Promise<Uint8Array> {
    try {
      const dir = await this.getDir();
      const fileHandle = await dir.getFileHandle(fileName);
      const file = await fileHandle.getFile();
      const buffer = await file.arrayBuffer();
      return new Uint8Array(buffer);
    } catch {
      return new Uint8Array(0);
    }
  }

  async append(fileName: string, data: Uint8Array): Promise<void> {
    const dir = await this.getDir();
    const fileHandle = await dir.getFileHandle(fileName, { create: true });
    const writable = await fileHandle.createWritable({ keepExistingData: true });
    const file = await fileHandle.getFile();
    await writable.seek(file.size);
    await writable.write(data as unknown as BufferSource);
    await writable.close();
  }

  async write(fileName: string, data: Uint8Array): Promise<void> {
    const dir = await this.getDir();
    const fileHandle = await dir.getFileHandle(fileName, { create: true });
    const writable = await fileHandle.createWritable({ keepExistingData: false });
    await writable.write(data as unknown as BufferSource);
    await writable.close();
  }

  async destroy(): Promise<void> {
    const root = await navigator.storage.getDirectory();
    await root.removeEntry(this.dirName, { recursive: true });
    this.dirHandle = null;
  }
}

/**
 * In-memory storage provider for testing.
 */
export class InMemoryStorageProvider implements StorageProvider {
  private files = new Map<string, Uint8Array[]>();

  async readAll(fileName: string): Promise<Uint8Array> {
    const chunks = this.files.get(fileName);
    if (!chunks || chunks.length === 0) return new Uint8Array(0);

    const totalSize = chunks.reduce((sum, c) => sum + c.byteLength, 0);
    const result = new Uint8Array(totalSize);
    let offset = 0;
    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return result;
  }

  async append(fileName: string, data: Uint8Array): Promise<void> {
    if (!this.files.has(fileName)) {
      this.files.set(fileName, []);
    }
    this.files.get(fileName)!.push(new Uint8Array(data));
  }

  async write(fileName: string, data: Uint8Array): Promise<void> {
    this.files.set(fileName, [new Uint8Array(data)]);
  }

  async destroy(): Promise<void> {
    this.files.clear();
  }
}
