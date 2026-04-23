"""
download_model.py
=================
Run this once during Render build to download TinyLlama GGUF.
The model (~400 MB Q2_K) is saved to models/tinyllama.gguf.
"""

import os
import sys
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    "/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"
)
OUT_PATH = Path("models/tinyllama.gguf")


def download():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUT_PATH.exists():
        size_mb = OUT_PATH.stat().st_size / 1_048_576
        print(f"[download_model] Model already exists ({size_mb:.1f} MB) — skipping.")
        return

    print(f"[download_model] Downloading TinyLlama Q2_K from HuggingFace ...")
    print(f"[download_model] URL  : {MODEL_URL}")
    print(f"[download_model] Dest : {OUT_PATH}")

    def _progress(count, block_size, total_size):
        pct = count * block_size * 100 // total_size
        mb  = count * block_size / 1_048_576
        sys.stdout.write(f"\r  {mb:.1f} MB downloaded ({pct}%)   ")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(MODEL_URL, OUT_PATH, reporthook=_progress)
        print()   # newline after progress
        size_mb = OUT_PATH.stat().st_size / 1_048_576
        print(f"[download_model] Done! Saved {size_mb:.1f} MB to {OUT_PATH}")
    except Exception as exc:
        print(f"[download_model] ERROR: {exc}")
        if OUT_PATH.exists():
            OUT_PATH.unlink()   # remove partial download
        sys.exit(1)


if __name__ == "__main__":
    download()
