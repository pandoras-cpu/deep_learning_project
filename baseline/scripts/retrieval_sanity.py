"""Inspect within-pool retrieval and score dev labels after ranking.

Labels are joined only after search. They are not stored in the index.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

INSPECT_QIDS = ("36", "62", "18", "31")
TOP_K = 3
QUERY_VECTORS = ROOT / "indexes" / "dev" / "query_vectors.npy"
QUERY_ROWS = ROOT / "indexes" / "dev" / "query_rows.json"


def dcg(gains: list[int]) -> float:
    return sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(gains))


def ndcg_at_k(labels: list[str], k: int) -> float | None:
    graded = [int(label) for label in labels]
    ideal = sorted(graded, reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0:
        return None
    return dcg(graded[:k]) / ideal_dcg


def recall_at_k(labels: list[str], relevant: set[str], k: int) -> float | None:
    gold = [label for label in labels if label in relevant]
    if not gold:
        return None
    return sum(label in relevant for label in labels[:k]) / len(gold)


def mrr(labels: list[str], relevant: set[str]) -> float | None:
    if not any(label in relevant for label in labels):
        return None
    for rank, label in enumerate(labels, start=1):
        if label in relevant:
            return 1.0 / rank
    return None


def mean_defined(values: list[float | None]) -> dict:
    defined = [value for value in values if value is not None]
    return {
        "mean": None if not defined else float(np.mean(defined)),
        "n_defined": len(defined),
        "n_undefined": len(values) - len(defined),
    }


def encode_queries() -> None:
    import torch

    from corpus import load_split
    from retriever import BgeEncoder

    with (ROOT / "configs" / "baseline.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    frame = load_split(ROOT, "dev")
    qids = []
    questions = []
    for qid, rows in frame.groupby("qid", sort=False):
        qids.append(str(qid))
        questions.append(rows["question"].iloc[0])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    encoder = BgeEncoder(config, ROOT / config["cache_dir"], device)
    vectors = encoder.encode(questions, as_query=True, batch_size=64)
    QUERY_VECTORS.parent.mkdir(parents=True, exist_ok=True)
    np.save(QUERY_VECTORS, vectors)
    QUERY_ROWS.write_text(
        json.dumps({"qids": qids, "questions": questions}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {QUERY_VECTORS.relative_to(ROOT)} {vectors.shape}")


def score() -> None:
    from candidate_index import CandidateIndex
    from corpus import load_split

    with (ROOT / "configs" / "baseline.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    index_dir = ROOT / config["retriever"]["index_dir"] / "dev"
    frame = load_split(ROOT, "dev")
    labels = dict(zip(frame["evidence_id"], frame["label"]))
    rows = json.loads(QUERY_ROWS.read_text(encoding="utf-8"))
    qids = rows["qids"]
    questions = rows["questions"]
    query_vectors = np.load(QUERY_VECTORS)
    if len(qids) != len(query_vectors):
        raise SystemExit("query vector count does not match question count")
    candidate_index = CandidateIndex.load(index_dir)

    per_question = []
    recalls_2 = []
    recalls_12 = []
    success_2 = []
    mrr_2 = []
    mrr_12 = []
    ndcgs = []
    for qid, question, vector in zip(qids, questions, query_vectors):
        hits = candidate_index.search(qid, vector, top_k=32)
        if any(hit.qid != qid for hit in hits):
            raise SystemExit(f"{qid} retrieved another question's evidence")
        ranked_labels = [labels[hit.evidence_id] for hit in hits]
        recalls_2.append(recall_at_k(ranked_labels, {"2"}, TOP_K))
        recalls_12.append(recall_at_k(ranked_labels, {"1", "2"}, TOP_K))
        has_full = any(label == "2" for label in ranked_labels)
        success_2.append(None if not has_full else float(any(label == "2" for label in ranked_labels[:TOP_K])))
        mrr_2.append(mrr(ranked_labels, {"2"}))
        mrr_12.append(mrr(ranked_labels, {"1", "2"}))
        ndcgs.append(ndcg_at_k(ranked_labels, TOP_K))
        per_question.append(
            {
                "qid": qid,
                "ranked_labels": ranked_labels,
                "top": [
                    {
                        "evidence_id": hit.evidence_id,
                        "score": round(hit.score, 4),
                        "label_not_used_in_ranking": labels[hit.evidence_id],
                        "source": hit.source,
                        "text": hit.text[:180],
                    }
                    for hit in hits[:TOP_K]
                ],
            }
        )

    demo_qid = "36"
    demo_position = qids.index(demo_qid)
    restricted = candidate_index.search(demo_qid, query_vectors[demo_position], top_k=TOP_K)
    unrestricted = candidate_index.search_unrestricted(query_vectors[demo_position], top_k=TOP_K)
    summary = {
        "split": "dev",
        "search": "within question candidate pool",
        "top_k": TOP_K,
        "n_questions": len(qids),
        "index_meta": candidate_index.meta,
        "relevance_definitions": {
            "full": "label == 2",
            "helpful_or_full": "label in {1, 2}",
            "ndcg_gain": "2^label - 1, labels 0/1/2",
        },
        "recall_at_3_label2": mean_defined(recalls_2),
        "success_at_3_label2": mean_defined(success_2),
        "mrr_label2": mean_defined(mrr_2),
        "recall_at_3_label1or2": mean_defined(recalls_12),
        "mrr_label1or2": mean_defined(mrr_12),
        "ndcg_at_3_graded": mean_defined(ndcgs),
        "unrestricted_search_is_not_baseline": {
            "qid": demo_qid,
            "question": questions[demo_position],
            "restricted_evidence_ids": [hit.evidence_id for hit in restricted],
            "unrestricted_hits": [
                {"evidence_id": hit.evidence_id, "qid": hit.qid, "score": round(hit.score, 4)}
                for hit in unrestricted
            ],
        },
        "inspections": [item for item in per_question if item["qid"] in INSPECT_QIDS],
    }
    results = ROOT / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "retrieval_sanity.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (results / "retrieval_sanity.md").write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "n_questions",
        "recall_at_3_label2",
        "success_at_3_label2",
        "mrr_label2",
        "recall_at_3_label1or2",
        "mrr_label1or2",
        "ndcg_at_3_graded",
    )}, indent=2))


def render_markdown(summary: dict) -> str:
    def line(name: str, item: dict) -> str:
        mean = "undefined" if item["mean"] is None else f"{item['mean']:.4f}"
        return (
            f"| {name} | {mean} | {item['n_defined']} | {item['n_undefined']} |"
        )

    meta = summary["index_meta"]
    lines = [
        "# ePQA dev retrieval sanity check",
        "",
        "排序只在当前问题自己的候选池内进行。`label` 和 `answer` 不在索引里，指标是检索之后才接合的。",
        "",
        f"- Encoder: `{meta['model_id']}` `{meta['revision'][:12]}`",
        f"- Query prefix: `{meta['query_prefix']}`",
        "- Passages: `candidate` text, CLS pooling, L2 normalized, inner product",
        f"- Vectors: {meta['vectors']}；排除空 candidate {meta['empty_candidates_excluded']} 行",
        f"- Truncated at {meta['max_length']} tokens: {meta['passages_truncated_at_max_length']}",
        "",
        "没有相关候选的问题不计入均值，也不记成 0。",
        "",
        "| metric | mean | defined questions | undefined |",
        "| --- | ---: | ---: | ---: |",
        line("Recall@3, relevant = label 2", summary["recall_at_3_label2"]),
        line("Success@3, at least one label 2", summary["success_at_3_label2"]),
        line("MRR, relevant = label 2", summary["mrr_label2"]),
        line("Recall@3, relevant = label 1 or 2", summary["recall_at_3_label1or2"]),
        line("MRR, relevant = label 1 or 2", summary["mrr_label1or2"]),
        line("nDCG@3, gain = 2^label - 1", summary["ndcg_at_3_graded"]),
        "",
        "## Why the pool is restricted",
        "",
        f"问题 `{summary['unrestricted_search_is_not_baseline']['qid']}` 的全索引 Top-3 不是 baseline。",
        "",
    ]
    for hit in summary["unrestricted_search_is_not_baseline"]["unrestricted_hits"]:
        lines.append(
            f"- `{hit['evidence_id']}` qid={hit['qid']} score={hit['score']}"
        )
    lines.extend(["", "## Manual top-3", ""])
    for item in summary["inspections"]:
        labels = ", ".join(hit["label_not_used_in_ranking"] for hit in item["top"])
        lines.append(f"### qid {item['qid']}")
        lines.append("")
        lines.append(f"Top-3 labels after ranking: {labels}")
        lines.append("")
        for hit in item["top"]:
            text = hit["text"].replace("\n", " ")
            lines.append(
                f"- `{hit['evidence_id']}` label={hit['label_not_used_in_ranking']} "
                f"source={hit['source']} score={hit['score']}: {text}"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    subprocess.check_call([sys.executable, str(Path(__file__).resolve()), "encode"])
    subprocess.check_call([sys.executable, str(Path(__file__).resolve()), "score"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "encode":
        encode_queries()
    elif len(sys.argv) > 1 and sys.argv[1] == "score":
        score()
    else:
        main()
