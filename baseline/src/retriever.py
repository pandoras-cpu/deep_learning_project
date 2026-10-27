"""Question-pool retriever.

Each question is ranked only against its own candidate passages. The index
stores candidate text, never labels or reference answers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from corpus import load_split, sha256_file, data_dir


class BgeEncoder:
    def __init__(self, config: dict, cache_dir: Path, device: str):
        spec = config["retriever"]
        self.query_prefix = spec["query_prefix"]
        if not self.query_prefix.endswith(" "):
            raise ValueError("query_prefix must keep the official trailing space")
        self.max_length = int(spec["max_length"])
        self.device = device
        revision = spec["revision"]
        model_id = spec["model_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], as_query: bool, batch_size: int = 64) -> np.ndarray:
        if as_query:
            texts = [self.query_prefix + text for text in texts]
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            cls = self.model(**encoded).last_hidden_state[:, 0]
            cls = F.normalize(cls, p=2, dim=1)
            outputs.append(cls.float().cpu().numpy())
        if not outputs:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        return np.concatenate(outputs, axis=0)


def build_index(root: Path, split: str, device: str, batch_size: int = 64) -> Path:
    config_path = root / "configs" / "baseline.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    spec = config["retriever"]
    if spec["pooling"] != "cls" or spec["normalize"] is not True:
        raise SystemExit("baseline retriever expects CLS pooling and L2 normalization")
    if spec["similarity"] != "inner_product":
        raise SystemExit("baseline similarity is inner product on normalized vectors")

    frame = load_split(root, split)
    passages = frame.loc[frame["candidate"].ne("")].copy()
    texts = passages["candidate"].tolist()
    cache_dir = root / config["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    encoder = BgeEncoder(config, cache_dir, device)
    embeddings = encoder.encode(texts, as_query=False, batch_size=batch_size)
    if embeddings.shape != (len(texts), encoder.model.config.hidden_size):
        raise SystemExit(f"unexpected embedding shape {embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise SystemExit(f"embeddings are not normalized: min={norms.min()} max={norms.max()}")

    # Import FAISS only after torch has finished encoding. Loading both OpenMP
    # copies in the other order aborts the process on this machine.
    import faiss

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    records = []
    for vector_id, row in enumerate(passages.itertuples(index=False)):
        records.append(
            {
                "vector_id": vector_id,
                "evidence_id": row.evidence_id,
                "qid": row.qid,
                "asin": row.ASIN,
                "question": row.question,
                "text": row.candidate,
                "source": row.source,
                "qa_pair_id": row.qa_pair_id,
                "file_line": int(row.file_line),
            }
        )
    token_lengths = encoder.tokenizer(
        texts, add_special_tokens=True, truncation=False, padding=False
    )["input_ids"]
    truncated = sum(len(ids) > encoder.max_length for ids in token_lengths)
    out_dir = root / spec["index_dir"] / split
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with (out_dir / "mapping.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    meta = {
        "split": split,
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "query_prefix": spec["query_prefix"],
        "passage_prefix": "",
        "pooling": "cls",
        "normalize": True,
        "similarity": "inner_product",
        "max_length": encoder.max_length,
        "device": device,
        "rows_in_csv": int(len(frame)),
        "empty_candidates_excluded": int(frame["candidate"].eq("").sum()),
        "vectors": len(records),
        "embedding_dim": int(embeddings.shape[1]),
        "passages_truncated_at_max_length": int(truncated),
        "tokenizer_class": type(encoder.tokenizer).__name__,
        "do_lower_case": bool(getattr(encoder.tokenizer, "do_lower_case", False)),
        "source_csv_sha256": sha256_file(data_dir(root) / f"{split}.csv"),
        "labels_in_index": False,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_dir
