"""FAISS search over one split. This module does not import torch."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass(frozen=True)
class Hit:
    vector_id: int
    evidence_id: str
    qid: str
    asin: str
    source: str
    qa_pair_id: str
    text: str
    score: float


class CandidateIndex:
    def __init__(self, index: faiss.Index, records: list[dict], meta: dict):
        self.index = index
        self.records = records
        self.meta = meta
        self.qid_to_ids: dict[str, list[int]] = {}
        for record in records:
            self.qid_to_ids.setdefault(record["qid"], []).append(record["vector_id"])

    @classmethod
    def load(cls, index_dir: Path) -> "CandidateIndex":
        index = faiss.read_index(str(index_dir / "index.faiss"))
        records = [
            json.loads(line)
            for line in (index_dir / "mapping.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        if index.ntotal != len(records):
            raise SystemExit("FAISS index and mapping have different lengths")
        for label_key in ("label", "answer"):
            if any(label_key in record for record in records):
                raise SystemExit(f"mapping contains {label_key}")
        return cls(index, records, meta)

    def search(self, qid: str, query_vector: np.ndarray, top_k: int) -> list[Hit]:
        allowed = self.qid_to_ids.get(qid, [])
        if not allowed or top_k <= 0:
            return []
        ids = np.asarray(allowed, dtype=np.int64)
        selector = faiss.IDSelectorArray(ids)
        params = faiss.SearchParameters()
        params.sel = selector
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, found = self.index.search(query, min(top_k, len(ids)), params=params)
        hits = []
        for score, vector_id in zip(scores[0].tolist(), found[0].tolist()):
            if vector_id < 0:
                continue
            record = self.records[vector_id]
            hits.append(
                Hit(
                    vector_id=vector_id,
                    evidence_id=record["evidence_id"],
                    qid=record["qid"],
                    asin=record["asin"],
                    source=record["source"],
                    qa_pair_id=record["qa_pair_id"],
                    text=record["text"],
                    score=float(score),
                )
            )
        hits.sort(key=lambda hit: (-hit.score, hit.vector_id))
        return hits

    def search_unrestricted(self, query_vector: np.ndarray, top_k: int) -> list[Hit]:
        """Full-index search. Not the baseline. Used only to show cross-question hits."""
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, found = self.index.search(query, top_k)
        hits = []
        for score, vector_id in zip(scores[0].tolist(), found[0].tolist()):
            if vector_id < 0:
                continue
            record = self.records[vector_id]
            hits.append(
                Hit(
                    vector_id=int(vector_id),
                    evidence_id=record["evidence_id"],
                    qid=record["qid"],
                    asin=record["asin"],
                    source=record["source"],
                    qa_pair_id=record["qa_pair_id"],
                    text=record["text"],
                    score=float(score),
                )
            )
        return hits
