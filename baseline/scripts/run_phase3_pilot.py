"""Five-example evaluation pilot. Does not rewrite baseline answers.

Judge and NLI run in separate processes so both models are not loaded together.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONFIG_PATH = ROOT / "configs" / "evaluation_v1.yaml"
JUDGE_OUTPUT = ROOT / "results" / "evaluation" / "pilot5_judge.jsonl"
NLI_OUTPUT = ROOT / "results" / "evaluation" / "pilot5_nli.jsonl"
SUMMARY_OUTPUT = ROOT / "results" / "evaluation" / "pilot5_summary.json"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evidence_from(record: dict) -> list[dict]:
    return [
        {"evidence_id": item["evidence_id"], "text": item["text"]}
        for item in record["retrieved"]
        if item["text"].strip()
    ]


def selected_records(config: dict) -> list[dict]:
    wanted = set(config["llm_judge"]["pilot_example_ids"])
    standard = {
        row["example_id"]: row
        for row in load_rows(ROOT / config["conditions"]["standard_rag"]["existing_output"])
    }
    gold = {
        row["example_id"]: row
        for row in load_rows(ROOT / config["conditions"]["gold_evidence_qa"]["existing_output"])
    }
    records = []
    for example_id in config["llm_judge"]["pilot_example_ids"]:
        if example_id not in wanted:
            continue
        for condition, source in (("standard_top3", standard), ("gold_evidence_qa", gold)):
            row = source[example_id]
            records.append(
                {
                    "example_id": example_id,
                    "qid": row["qid"],
                    "condition": condition,
                    "question": row["question"],
                    "answer": row["answer"],
                    "generation_error": row["error"],
                    "evidence": evidence_from(row),
                    "input_tokens_generation": row["input_tokens"],
                    "output_tokens_generation": row["output_tokens"],
                }
            )
    return records


def output_paths(mode: str) -> tuple[Path, Path, Path]:
    if mode == "after":
        return (
            ROOT / "results" / "evaluation" / "pilot5_after_judge.jsonl",
            ROOT / "results" / "evaluation" / "pilot5_after_nli.jsonl",
            ROOT / "results" / "evaluation" / "pilot5_after_summary.json",
        )
    return JUDGE_OUTPUT, NLI_OUTPUT, SUMMARY_OUTPUT


def run_judge(judge_output: Path) -> None:
    from evaluation.qwen_judge import QwenJudge
    from evaluation.schema import Evidence

    config = load_config()
    judge_config = config["llm_judge"]
    judge = QwenJudge(CONFIG_PATH)
    judge_output.parent.mkdir(parents=True, exist_ok=True)
    with judge_output.open("w", encoding="utf-8") as handle:
        for record in selected_records(config):
            output = {
                "example_id": record["example_id"],
                "qid": record["qid"],
                "condition": record["condition"],
                "question": record["question"],
                "answer": record["answer"],
                "evidence": record["evidence"],
                "judge_model": judge_config["model_id"],
                "judge_revision": judge_config["revision"],
                "same_model_judge": False,
                "prompt_versions": judge_config["prompt_versions"],
                "decoding": judge_config["decoding"],
                "preliminary": True,
                "claims": [],
                "facts": [],
                "rejected_facts": [],
                "matches": [],
                "judge_error": None,
            }
            if record["generation_error"] == "no_label2_evidence":
                output["generation_status"] = "gold_not_applicable_no_label2"
                output["coverage_undefined_reason"] = "no_required_facts"
            elif record["generation_error"]:
                output["generation_status"] = "generation_error"
                output["judge_error"] = record["generation_error"]
            else:
                output["generation_status"] = "ok"
                try:
                    claims, claim_meta = judge.extract_answer_claims(record["question"], record["answer"])
                    output["claims"] = [
                        {"span": claim.original_span, "claim": claim.text, "kind": claim.status}
                        for claim in claims
                    ]
                    output["claim_call"] = claim_meta
                    if record["evidence"]:
                        evidence = [
                            Evidence(evidence_id=item["evidence_id"], text=item["text"])
                            for item in record["evidence"]
                        ]
                        facts, fact_meta = judge.extract_supported_facts(record["question"], evidence)
                        output["facts"] = [
                            {
                                "fact": fact.text,
                                "evidence_id": fact.evidence_ids[0] if fact.evidence_ids else None,
                                "quote": fact.quote,
                                "aligned_span": fact.aligned_span,
                                "alignment_status": fact.alignment_status,
                                "alignment_method": fact.alignment_method,
                                "alignment_reason": fact.alignment_reason,
                                "evidence_text": fact.evidence_text,
                            }
                            for fact in facts
                        ]
                        output["fact_call"] = fact_meta
                        aligned = [fact for fact in facts if fact.alignment_status == "aligned" and fact.text]
                        if aligned:
                            matches, match_meta = judge.match_answer_to_facts(
                                record["question"], record["answer"], aligned
                            )
                            output["matches"] = [
                                {
                                    "fact": match.fact,
                                    "expressed": match.expressed,
                                    "answer_span": match.answer_span,
                                    "reason": match.reason,
                                }
                                for match in matches
                            ]
                            output["match_call"] = match_meta
                    else:
                        output["coverage_undefined_reason"] = "no_required_facts"
                except Exception as exc:  # noqa: BLE001
                    output["judge_error"] = f"{type(exc).__name__}: {exc}"
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{output['example_id']} {output['condition']} "
                f"claims={len(output['claims'])} facts={len(output['facts'])} error={output['judge_error']}",
                flush=True,
            )
    judge.close()


def verify_fact(verifier, fact: dict, threshold: float) -> dict:
    from evaluation.metrics import apply_confidence

    if fact.get("alignment_status") != "aligned" or not fact.get("aligned_span"):
        return {
            **fact,
            "fact_status": "evaluation_error",
            "fact_reason": fact.get("alignment_reason") or "alignment_failure",
            "fact_nli": None,
        }
    scored = verifier.score_texts(fact["aligned_span"], fact["fact"])
    status = apply_confidence(scored.status, scored.max_probability, threshold)
    if status == "supported":
        fact_status = "verified"
    elif status == "contradicted":
        fact_status = "contradicted_by_citation"
    elif status == "uncertain":
        fact_status = "uncertain"
    else:
        fact_status = "unsupported_by_citation"
    return {
        **fact,
        "fact_status": fact_status,
        "fact_reason": status,
        "fact_nli": {
            "nli_label": scored.nli_label,
            "scores": scored.scores,
            "max_probability": scored.max_probability,
            "truncated": scored.truncated,
            "token_count": scored.token_count,
        },
    }


def run_nli(judge_output: Path, nli_output: Path, summary_output: Path) -> None:
    from evaluation.metrics import aggregate, score_coverage, score_ucr
    from evaluation.nli_verifier import NliVerifier
    from evaluation.schema import Evidence

    config = load_config()
    threshold = float(config["llm_judge"]["low_confidence_below"])
    verifier = NliVerifier(CONFIG_PATH)
    records = load_rows(judge_output)
    finished = []
    with nli_output.open("w", encoding="utf-8") as handle:
        for record in records:
            judgments = []
            if record["generation_status"] != "ok":
                statuses = []
            else:
                evidence = [Evidence(evidence_id=item["evidence_id"], text=item["text"]) for item in record["evidence"]]
                for claim in record["claims"]:
                    if claim["kind"] != "factual":
                        judgments.append({**claim, "nli_status": None, "scores": None, "truncated": False})
                        continue
                    scored = verifier.score_claim(evidence, claim["claim"])
                    judgments.append(
                        {
                            **claim,
                            "nli_status": scored.status,
                            "decision_reason": scored.decision_reason,
                            "nli_label": scored.nli_label,
                            "scores": scored.scores,
                            "max_probability": scored.max_probability,
                            "truncated": scored.truncated,
                            "evidence_ids": list(scored.evidence_ids),
                            "passages": scored.passage_judgments,
                        }
                    )
                statuses = [item["nli_status"] for item in judgments if item["nli_status"]]
            verified_facts = []
            fact_rows = []
            for fact in record["facts"]:
                checked = verify_fact(verifier, fact, threshold) if record["generation_status"] == "ok" else fact
                fact_rows.append(checked)
                if checked.get("fact_status") == "verified":
                    verified_facts.append(checked)
            match_by_fact = {item["fact"]: item for item in record["matches"]}
            expressed = []
            for fact in verified_facts:
                match = match_by_fact.get(fact["fact"])
                expressed.append(None if match is None else match["expressed"])
            if record["generation_status"] != "ok":
                coverage = {
                    "numerator": 0,
                    "denominator": 0,
                    "coverage": None,
                    "undefined": True,
                    "undefined_reason": "gold_not_applicable_no_label2"
                    if record["generation_status"] == "gold_not_applicable_no_label2"
                    else record["generation_status"],
                }
                ucr = score_ucr([])
                ucr["undefined_reason"] = coverage["undefined_reason"]
            else:
                ucr = score_ucr(statuses)
                coverage = score_coverage(expressed)
                if fact_rows and not verified_facts:
                    coverage["undefined"] = True
                    coverage["coverage"] = None
                    coverage["numerator"] = 0
                    coverage["denominator"] = 0
                    coverage["undefined_reason"] = "no_verified_required_facts"
                coverage["evidence_ids"] = [item["evidence_id"] for item in record["evidence"]]
                coverage["verified_facts"] = len(verified_facts)
                coverage["fact_status_counts"] = {
                    status: sum(item.get("fact_status") == status for item in fact_rows)
                    for status in (
                        "verified",
                        "unsupported_by_citation",
                        "contradicted_by_citation",
                        "uncertain",
                        "evaluation_error",
                    )
                }
            enriched = {
                **record,
                "facts": fact_rows,
                "claim_judgments": judgments,
                "ucr": ucr,
                "coverage_evidence_conditional": coverage,
                "nli_model": verifier.model_id,
                "nli_revision": verifier.revision,
                "low_confidence_below": threshold,
            }
            finished.append(enriched)
            handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    summary = {
        "preliminary": True,
        "human_validated": False,
        "examples": config["llm_judge"]["pilot_example_ids"],
        "coverage_scope": "evidence shown to that condition; numerator and denominator use the same fact list",
        "discarded_denominator": "14/38 and 24/38 are not used",
        "conditions": [
            aggregate(finished, "standard_top3"),
            aggregate(finished, "gold_evidence_qa"),
        ],
    }
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def write_comparison() -> None:
    before_path = ROOT / "results" / "evaluation" / "pilot5_before_nli.jsonl"
    after_path = ROOT / "results" / "evaluation" / "pilot5_after_nli.jsonl"
    before = {(row["example_id"], row["condition"]): row for row in load_rows(before_path)}
    after = {(row["example_id"], row["condition"]): row for row in load_rows(after_path)}
    rows = []
    for key in after:
        old = before[key]
        new = after[key]
        rows.append(
            {
                "example_id": key[0],
                "condition": key[1],
                "before_claims": [
                    {"claim": item.get("claim"), "kind": item.get("kind"), "nli_status": item.get("nli_status")}
                    for item in old.get("claim_judgments", [])
                ],
                "after_claims": [
                    {
                        "claim": item.get("claim"),
                        "kind": item.get("kind"),
                        "nli_status": item.get("nli_status"),
                        "decision_reason": item.get("decision_reason"),
                    }
                    for item in new.get("claim_judgments", [])
                ],
                "before_facts": old.get("facts", []),
                "after_facts": [
                    {
                        "fact": item.get("fact"),
                        "alignment_status": item.get("alignment_status"),
                        "fact_status": item.get("fact_status"),
                        "fact_reason": item.get("fact_reason"),
                        "aligned_span": item.get("aligned_span"),
                    }
                    for item in new.get("facts", [])
                ],
                "before_ucr": old.get("ucr"),
                "after_ucr": new.get("ucr"),
                "before_coverage": old.get("coverage_evidence_conditional"),
                "after_coverage": new.get("coverage_evidence_conditional"),
            }
        )
    path = ROOT / "results" / "evaluation" / "pilot5_before_after.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main(mode: str) -> None:
    script = str(Path(__file__).resolve())
    subprocess.check_call([sys.executable, script, "judge", mode])
    subprocess.check_call([sys.executable, script, "nli", mode])
    if mode == "after":
        write_comparison()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    mode = sys.argv[2] if len(sys.argv) > 2 else "before"
    judge_path, nli_path, summary_path = output_paths(mode if command != "run" else command)
    if command == "judge":
        run_judge(judge_path)
    elif command == "nli":
        run_nli(judge_path, nli_path, summary_path)
    elif command in {"run", "after"}:
        main("after" if command == "after" else "before")
    else:
        raise SystemExit(f"unknown command {command}")
