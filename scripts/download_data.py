"""Download and extract HDFS_v1 (Loghub, via Zenodo) into data/raw/.

Usage: uv run python scripts/download_data.py

Idempotent: skips the download if the zip is already complete, and prints a SHA-256
so teammates can confirm they all hold the same file.
"""

import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1"
EXPECTED_BYTES = 186_645_559
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
ZIP_PATH = RAW_DIR / "HDFS_v1.zip"
CHUNK = 1 << 20


def download() -> None:
    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size == EXPECTED_BYTES:
        print(f"{ZIP_PATH.name} already downloaded")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ZIP_PATH.with_suffix(".part")
    with urllib.request.urlopen(URL, timeout=60) as resp, open(tmp, "wb") as out:
        done = 0
        while chunk := resp.read(CHUNK):
            out.write(chunk)
            done += len(chunk)
            print(f"\r{done / 1e6:7.1f} / {EXPECTED_BYTES / 1e6:.1f} MB", end="", flush=True)
    print()
    if tmp.stat().st_size != EXPECTED_BYTES:
        raise RuntimeError(f"Unexpected size {tmp.stat().st_size}, expected {EXPECTED_BYTES}")
    tmp.rename(ZIP_PATH)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    download()
    print(f"sha256: {sha256(ZIP_PATH)}")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(RAW_DIR)
        for info in z.infolist():
            print(f"  {info.filename}  {info.file_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
