"""Score Qwen2.5 answers with Qwen3 extraction and the existing DeBERTa rules."""

from __future__ import annotations

import gc
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.metrics import aggregate, score_coverage, score_ucr
from evaluation.nli_verifier import NliVerifier
from evaluation.qwen_judge import QwenJudge
from evaluation.schema import Evidence

CONFIG = ROOT / "configs" / "evaluation_v2.yaml"
OUT = ROOT / "results" / "baseline_v2"
JUDGE_PATH = OUT / "judge.jsonl"
EVAL_PATH = OUT / "evaluations.jsonl"
SUMMARY_PATH = OUT / "summary.json"
THRESHOLD = 0.5


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def fact_dict(fact) -> dict:
    return {
        "fact": fact.text,
        "evidence_id": fact.evidence_ids[0] if fact.evidence_ids else None,
        "quote": fact.quote,
        "aligned_span": fact.aligned_span,
        "alignment_status": fact.alignment_status,
        "alignment_method": fact.alignment_method,
        "alignment_reason": fact.alignment_reason,
        "evidence_text": fact.evidence_text,
    }


def verify_fact(verifier: NliVerifier, fact: dict) -> dict:
    from evaluation.metrics import apply_confidence

    if fact.get("alignment_status") != "aligned" or not fact.get("aligned_span"):
        return {
            **fact,
            "fact_status": "evaluation_error",
            "fact_reason": fact.get("alignment_reason") or "alignment_failure",
        }
    scored = verifier.score_texts(fact["aligned_span"], fact["fact"])
    status = apply_confidence(scored.status, scored.max_probability, THRESHOLD)
    mapping = {
        "supported": "verified",
        "contradicted": "contradicted_by_citation",
        "uncertain": "uncertain",
    }
    return {
        **fact,
        "fact_status": mapping.get(status, "unsupported_by_citation"),
        "fact_reason": status,
        "fact_nli": {
            "nli_label": scored.nli_label,
            "scores": scored.scores,
            "max_probability": scored.max_probability,
            "truncated": scored.truncated,
        },
    }


def judge() -> None:
    rows = read_jsonl(OUT / "answers.jsonl")
    done = {row["example_id"] for row in read_jsonl(JUDGE_PATH)}
    client = QwenJudge(CONFIG, section="claim_model")
    for row in rows:
        if row["example_id"] in done:
            continue
        if row.get("error"):
            append(JUDGE_PATH, {
                "example_id": row["example_id"],
                "qid": row["qid"],
                "condition": "standard_rag",
                "question": row["question"],
                "answer": row.get("answer"),
                "generation_status": row["error"],
                "evidence": row["retrieved"],
                "claims": [],
                "facts": [],
                "matches": [],
            })
            continue
        evidence = [Evidence(item["evidence_id"], item["text"]) for item in row["retrieved"]]
        try:
            claims, claim_meta = client.extract_answer_claims(row["question"], row["answer"])
            facts, fact_meta = client.extract_supported_facts(row["question"], evidence)
            aligned = [fact for fact in facts if fact.alignment_status == "aligned" and fact.text]
            if aligned:
                matches, match_meta = client.match_answer_to_facts(row["question"], row["answer"], aligned)
            else:
                matches, match_meta = [], {"skipped": "no_aligned_facts"}
            record = {
                "example_id": row["example_id"],
                "qid": row["qid"],
                "condition": "standard_rag",
                "question": row["question"],
                "answer": row["answer"],
                "generation_status": "ok" if row.get("error") is None else row["error"],
                "evidence": row["retrieved"],
                "claims": [
                    {"span": claim.original_span, "claim": claim.text, "kind": claim.status}
                    for claim in claims
                ],
                "claim_meta": claim_meta,
                "facts": [fact_dict(fact) for fact in facts],
                "fact_meta": fact_meta,
                "matches": [match.__dict__ for match in matches],
                "match_meta": match_meta,
                "judge_model_id": client.model_id,
                "judge_revision": client.revision,
            }
        except Exception as exc:
            record = {
                "example_id": row["example_id"],
                "qid": row["qid"],
                "condition": "standard_rag",
                "question": row["question"],
                "answer": row.get("answer"),
                "generation_status": "evaluation_error",
                "error": f"{type(exc).__name__}: {exc}",
                "evidence": row["retrieved"],
                "claims": [],
                "facts": [],
                "matches": [],
            }
        append(JUDGE_PATH, record)
        print(row["example_id"], record["generation_status"], len(record["claims"]), len(record["facts"]), flush=True)
    del client
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def nli() -> None:
    verifier = NliVerifier(CONFIG)
    finished = []
    for row in read_jsonl(JUDGE_PATH):
        judgments = []
        statuses = []
        if row.get("generation_status") == "ok":
            evidence = [Evidence(item["evidence_id"], item["text"]) for item in row["evidence"]]
            for claim in row["claims"]:
                if claim["kind"] != "factual":
                    judgments.append({**claim, "status": "not_a_factual_claim"})
                    continue
                scored = verifier.score_claim(evidence, claim["claim"])
                judgments.append({
                    **claim,
                    "status": scored.status,
                    "decision_reason": scored.decision_reason,
                    "passage_judgments": scored.passage_judgments,
                })
                statuses.append(scored.status)
        fact_rows = [verify_fact(verifier, fact) if row.get("generation_status") == "ok" else fact for fact in row.get("facts", [])]
        verified = [fact for fact in fact_rows if fact.get("fact_status") == "verified"]
        match_by_fact = {item["fact"]: item for item in row.get("matches", [])}
        expressed = []
        for fact in verified:
            match = match_by_fact.get(fact["fact"])
            expressed.append(None if match is None else match["expressed"])
        ucr = score_ucr(statuses)
        coverage = score_coverage(expressed)
        if row.get("generation_status") != "ok":
            coverage = {
                "numerator": 0,
                "denominator": 0,
                "coverage": None,
                "undefined": True,
                "undefined_reason": row.get("generation_status"),
            }
            ucr["undefined"] = True
            ucr["undefined_reason"] = row.get("generation_status")
            ucr["ucr"] = None
        elif fact_rows and not verified:
            coverage = {
                "numerator": 0,
                "denominator": 0,
                "coverage": None,
                "undefined": True,
                "undefined_reason": "no_verified_required_facts",
            }
        finished.append({
            **row,
            "facts": fact_rows,
            "claim_judgments": judgments,
            "ucr": ucr,
            "coverage_evidence_conditional": coverage,
        })
    EVAL_PATH.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in finished), encoding="utf-8")
    summary = aggregate(finished, "standard_rag")
    summary["evaluator_version"] = "baseline-eval-v2"
    summary["human_validated"] = False
    summary["ucr_contradicted"] = sum(row["ucr"].get("contradicted", 0) for row in finished)
    summary["ucr_neutral_only"] = summary["unsupported"] - summary["ucr_contradicted"]
    summary["ucr_numerator"] = summary["unsupported"]
    summary["ucr_denominator"] = summary["supported"] + summary["unsupported"]
    summary["note"] = "unsupported in ucr_numerator includes contradicted. neutral_only is unsupported minus contradicted."
    summary["claim_status_counts"] = dict(Counter(
        item["status"] for row in finished for item in row["claim_judgments"]
    ))
    summary["fact_status_counts"] = dict(Counter(
        fact.get("fact_status") for row in finished for fact in row["facts"]
    ))
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "judge":
        judge()
    elif command == "nli":
        nli()
    elif command == "run":
        subprocess.run([sys.executable, __file__, "judge"], check=True)
        nli()
    else:
        raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main()
