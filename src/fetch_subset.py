"""Download only the files we need from the ESA Mission 1 zip on Zenodo.

The whole zip is 3.78 GB. HTTP range requests let us read its index and pull
single members, which is much faster on a slow connection. Pieces are stored as
they arrive, so an interrupted run resumes where it stopped.

The zip's members use Deflate64, which Python's zipfile cannot read, so members
are fetched as raw bytes and inflated with the inflate64 package.

Usage: python src/fetch_subset.py [--list]
"""
import argparse
import io
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = "https://zenodo.org/api/records/15237121/files/ESA-Mission1.zip/content"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"
CHANNELS = [f"channel_{i}" for i in range(41, 47)]
INDEX_CHUNK = 1 << 20
PIECE = 4 << 20
WORKERS = 8


def fetch_range(start, end):
    for attempt in range(60):
        try:
            req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
            raise IOError(f"short read {len(data)}")
        except Exception as e:
            wait = min(5 + attempt * 5, 60)
            print(f"  retry {attempt + 1} on bytes {start}-{end}: {e!r}, waiting {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"gave up on bytes {start}-{end}")


class RemoteFile(io.RawIOBase):
    """Seekable read-only file over HTTP ranges, enough for zipfile to read the index."""

    def __init__(self):
        req = urllib.request.Request(URL, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0
        self.cache_start = 0
        self.cache = b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        offset = self.pos - self.cache_start
        if offset < 0 or offset + n > len(self.cache):
            end = min(self.pos + max(n, INDEX_CHUNK), self.size) - 1
            self.cache = fetch_range(self.pos, end)
            self.cache_start = self.pos
            offset = 0
        data = self.cache[offset:offset + n]
        self.pos += len(data)
        return data


def wanted(name):
    base = name.rsplit("/", 1)[-1]
    if base in {"labels.csv", "anomaly_types.csv", "channels.csv", "telecommands.csv"}:
        return True
    return "/channels/" in f"/{name}" and base[:-4] in CHANNELS


def fetch_raw(info, parts_dir):
    """Fetch a member's compressed bytes as numbered pieces, skipping pieces already on disk."""
    header = fetch_range(info.header_offset, info.header_offset + 29)
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    start = info.header_offset + 30 + name_len + extra_len
    parts_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for k, offset in enumerate(range(0, info.compress_size, PIECE)):
        length = min(PIECE, info.compress_size - offset)
        path = parts_dir / f"{k:05d}"
        if not (path.exists() and path.stat().st_size == length):
            jobs.append((path, start + offset, start + offset + length - 1))

    def job(path, a, b):
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(fetch_range(a, b))
        tmp.replace(path)
        return b - a + 1

    t0, done = time.time(), 0
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(job, *j) for j in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            done += fut.result()
            if n % 5 == 0 or n == len(jobs):
                rate = done / max(time.time() - t0, 1e-3) / 1e3
                print(f"  {info.filename}: {n}/{len(jobs)} pieces, {rate:.0f} KB/s", flush=True)
    return sorted(p for p in parts_dir.iterdir() if p.suffix != ".tmp")


def inflate(info, pieces, target):
    if info.compress_type == zipfile.ZIP_STORED:
        decompress = lambda b: b
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        decompress = zlib.decompressobj(-15).decompress
    elif info.compress_type == 9:
        import inflate64
        decompress = inflate64.Inflater().inflate
    else:
        raise ValueError(f"unknown compression {info.compress_type}")

    crc = 0
    tmp = Path(str(target) + ".part")
    with open(tmp, "wb") as dst:
        for piece in pieces:
            data = decompress(piece.read_bytes())
            crc = zlib.crc32(data, crc)
            dst.write(data)
    if crc != info.CRC or tmp.stat().st_size != info.file_size:
        raise IOError(f"checksum failed for {info.filename}")
    tmp.replace(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    with zipfile.ZipFile(RemoteFile()) as z:
        infos = z.infolist()
    picked = [i for i in infos if not i.is_dir() and wanted(i.filename)]
    print(f"{len(infos)} entries in zip, picked {len(picked)}, "
          f"{sum(i.compress_size for i in picked) / 1e6:.1f} MB to fetch", flush=True)
    if args.list:
        for i in picked:
            print(f"  {i.filename} {i.compress_size} method={i.compress_type}")
        return

    for i in picked:
        target = OUT / i.filename
        if target.exists() and target.stat().st_size == i.file_size:
            print(f"have {i.filename}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        parts_dir = OUT / "_parts" / i.filename.rsplit("/", 1)[-1]
        pieces = fetch_raw(i, parts_dir)
        inflate(i, pieces, target)
        for piece in pieces:
            piece.unlink()
        parts_dir.rmdir()
        print(f"got {i.filename}, checksum ok", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
