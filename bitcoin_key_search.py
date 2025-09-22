import os
import requests
import gzip
import base58
import tempfile
import heapq
import shutil

HASH_RAW = "hash160_raw.bin"
HASH_SORTED = "hash160_sorted.bin"
ADDRESS_FILE = "Bitcoin_addresses_LATEST.txt"
CHUNK_SIZE = 20      # 20 bytes per hash160
SORT_BLOCK = 10_000_000  # Anzahl Hashes pro Chunk, RAM-schonend

def download_and_decompress():
    if os.path.exists(ADDRESS_FILE):
        print(f"{ADDRESS_FILE} exists, skipping download.")
        return True
    url = "http://addresses.loyce.club/Bitcoin_addresses_LATEST.txt.gz"
    try:
        print(f"Downloading {url}...")
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        with gzip.open(resp.raw, 'rb') as gz_file, open(ADDRESS_FILE, 'wb') as f_out:
            shutil.copyfileobj(gz_file, f_out)
        print("Download complete.")
        return True
    except Exception as e:
        print("Download failed:", e)
        return False

def addresses_to_hash160(input_file, output_file):
    print("Converting addresses to hash160...")
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'wb') as f_out:
        for line in f_in:
            addr = line.strip()
            try:
                raw = base58.b58decode_check(addr)
                hash160 = raw[1:]  # drop version byte
                f_out.write(hash160)
            except:
                continue
    print(f"Conversion done: {output_file}")

def external_sort(input_file, output_file):
    print("Sorting hash160s (external merge)...")
    temp_dir = tempfile.mkdtemp()
    chunks = []
    try:
        # 1. Split & sort in chunks
        with open(input_file, 'rb') as f:
            while True:
                data = f.read(CHUNK_SIZE * SORT_BLOCK)
                if not data:
                    break
                block = [data[i:i+CHUNK_SIZE] for i in range(0, len(data), CHUNK_SIZE)]
                block.sort()
                temp_path = os.path.join(temp_dir, f"chunk_{len(chunks)}.bin")
                with open(temp_path, 'wb') as cf:
                    cf.writelines(block)
                chunks.append(temp_path)

        # 2. Merge sorted chunks
        files = [open(c, 'rb') for c in chunks]
        heap = []
        for idx, f in enumerate(files):
            b = f.read(CHUNK_SIZE)
            if b:
                heap.append((b, idx))
        heapq.heapify(heap)

        with open(output_file, 'wb') as out_f:
            last = None
            while heap:
                val, idx = heapq.heappop(heap)
                if val != last:  # deduplicate
                    out_f.write(val)
                    last = val
                next_bytes = files[idx].read(CHUNK_SIZE)
                if next_bytes:
                    heapq.heappush(heap, (next_bytes, idx))

        print(f"Sorting and deduplication complete: {output_file}")
    finally:
        for f in files: f.close()
        shutil.rmtree(temp_dir)

def prepare_addresses():
    if not download_and_decompress():
        return None
    addresses_to_hash160(ADDRESS_FILE, HASH_RAW)
    external_sort(HASH_RAW, HASH_SORTED)
    os.remove(HASH_RAW)
    return HASH_SORTED

if __name__ == "__main__":
    final_file = prepare_addresses()
    if final_file:
        print("GPU-ready hash160 file:", final_file)
