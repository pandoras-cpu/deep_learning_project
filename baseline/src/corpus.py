"""Load official ePQA rows.

Retrieval code may read candidate text. It must not read label or answer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

OBSERVED_COLUMNS = [
    "qid",
    "question",
    "ASIN",
    "candidate",
    "source",
    "qa_pair_id",
    "title",
    "label",
    "answer",
]


def data_dir(root: Path) -> Path:
    return root / "external" / "contextual-product-qa" / "ePQA"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(root: Path, split: str) -> pd.DataFrame:
    path = data_dir(root) / f"{split}.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    if list(frame.columns) != OBSERVED_COLUMNS:
        raise SystemExit(f"{split}.csv columns are {list(frame.columns)}")
    for column in OBSERVED_COLUMNS:
        frame[column] = frame[column].str.strip()
    frame["file_line"] = frame.index + 2
    frame["evidence_id"] = split + ":" + frame["file_line"].astype(str)
    return frame
