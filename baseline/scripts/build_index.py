"""Build a within-question FAISS index for one ePQA split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retriever import build_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    out_dir = build_index(ROOT, args.split, args.device, args.batch_size)
    print(f"wrote {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
