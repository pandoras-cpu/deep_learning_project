"""Write the measured metrics table and the blank human-annotation template.

UCR and Answerable Coverage are left blank. They need claim-level annotation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metric_row(name: str, scope: str, item: dict | None, notes: str) -> dict:
    if item is None:
        return {
            "metric": name,
            "scope": scope,
            "value": "",
            "n_defined": "",
            "n_undefined": "",
            "notes": notes,
        }
    return {
        "metric": name,
        "scope": scope,
        "value": f"{item['mean']:.4f}",
        "n_defined": item["n_defined"],
        "n_undefined": item["n_undefined"],
        "notes": notes,
    }


def main() -> None:
    retrieval = json.loads((RESULTS / "retrieval_sanity.json").read_text(encoding="utf-8"))
    standard = load_jsonl(RESULTS / "baseline_dev.jsonl")
    gold = load_jsonl(RESULTS / "gold_evidence_dev.jsonl")
    rows = [
        metric_row(
            "Recall@3",
            "dev, relevant = label 2",
            retrieval["recall_at_3_label2"],
            "Within-question candidate pool. Undefined when the question has no label 2.",
        ),
        metric_row(
            "Success@3",
            "dev, at least one label 2 in top 3",
            retrieval["success_at_3_label2"],
            "Within-question candidate pool.",
        ),
        metric_row(
            "MRR",
            "dev, relevant = label 2",
            retrieval["mrr_label2"],
            "Full within-pool ranking, not only top 3.",
        ),
        metric_row(
            "Recall@3",
            "dev, relevant = label 1 or 2",
            retrieval["recall_at_3_label1or2"],
            "Label 1 and label 2 are not collapsed in the label-2 metrics.",
        ),
        metric_row(
            "MRR",
            "dev, relevant = label 1 or 2",
            retrieval["mrr_label1or2"],
            "Full within-pool ranking.",
        ),
        metric_row(
            "nDCG@3",
            "dev, gain = 2^label - 1",
            retrieval["ndcg_at_3_graded"],
            "Graded labels 0, 1, and 2.",
        ),
        {
            "metric": "standard_rag_questions",
            "scope": "dev sample, seed 42",
            "value": len(standard),
            "n_defined": len(standard),
            "n_undefined": 0,
            "notes": "Top-3 within the question pool. Not an open-corpus run.",
        },
        {
            "metric": "gold_evidence_qa_generated",
            "scope": "same 30 questions",
            "value": sum(row["error"] is None for row in gold),
            "n_defined": sum(row["error"] is None for row in gold),
            "n_undefined": sum(row["error"] == "no_label2_evidence" for row in gold),
            "notes": "Not retrieval. Uses every non-empty label=2 candidate. Undefined means no label=2 evidence, so no answer was generated.",
        },
        metric_row(
            "UCR",
            "30-question generation sample",
            None,
            "Not computed. Requires human labels of atomic claims.",
        ),
        metric_row(
            "Answerable Coverage",
            "30-question generation sample",
            None,
            "Not computed. Requires human labels of evidence-supported required facts.",
        ),
    ]
    fieldnames = ["metric", "scope", "value", "n_defined", "n_undefined", "notes"]
    with (RESULTS / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    annotation_fields = [
        "example_id",
        "condition",
        "qid",
        "asin",
        "question",
        "evidence_ids",
        "evidence_text",
        "model_answer",
        "error",
        "claim_text",
        "claim_label",
        "answerable_fact",
        "fact_expressed_correctly",
        "abstention_appropriate",
        "correctness",
        "annotator_notes",
    ]
    with (RESULTS / "human_annotation_template.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=annotation_fields)
        writer.writeheader()
        for condition, records in (("standard_rag", standard), ("gold_evidence_qa", gold)):
            for record in records:
                writer.writerow(
                    {
                        "example_id": record["example_id"],
                        "condition": condition,
                        "qid": record["qid"],
                        "asin": record["asin"],
                        "question": record["question"],
                        "evidence_ids": " | ".join(item["evidence_id"] for item in record["retrieved"]),
                        "evidence_text": "\n\n".join(
                            f"[{item['evidence_id']}] {item['text']}" for item in record["retrieved"]
                        ),
                        "model_answer": record["answer"] or "",
                        "error": record["error"] or "",
                        "claim_text": "",
                        "claim_label": "",
                        "answerable_fact": "",
                        "fact_expressed_correctly": "",
                        "abstention_appropriate": "",
                        "correctness": "",
                        "annotator_notes": "",
                    }
                )
    print(f"wrote {RESULTS / 'metrics_summary.csv'}")
    print(f"wrote {RESULTS / 'human_annotation_template.csv'}")


if __name__ == "__main__":
    main()
