"""Full 30-question preliminary-v1 evaluation. Does not change scoring rules or baseline answers."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

JUDGE_OUTPUT = ROOT / "results" / "evaluation" / "v1_judge.jsonl"
NLI_OUTPUT = ROOT / "results" / "evaluation" / "v1_evaluations.jsonl"
SUMMARY_OUTPUT = ROOT / "results" / "evaluation" / "v1_summary.json"
PAIRED_OUTPUT = ROOT / "results" / "evaluation" / "v1_paired_23.csv"
VERSION = "preliminary-v1"


def load_pilot():
    spec = importlib.util.spec_from_file_location("phase3_pilot", ROOT / "scripts" / "run_phase3_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            keys.add((row["example_id"], row["condition"]))
    return keys


def all_records(pilot) -> list[dict]:
    config = pilot.load_config()
    standard_rows = pilot.load_rows(ROOT / config["conditions"]["standard_rag"]["existing_output"])
    gold = {
        row["example_id"]: row
        for row in pilot.load_rows(ROOT / config["conditions"]["gold_evidence_qa"]["existing_output"])
    }
    records = []
    for row in standard_rows:
        for condition, source in (("standard_top3", row), ("gold_evidence_qa", gold[row["example_id"]])):
            records.append(
                {
                    "example_id": source["example_id"],
                    "qid": source["qid"],
                    "condition": condition,
                    "question": source["question"],
                    "answer": source["answer"],
                    "generation_error": source["error"],
                    "evidence": pilot.evidence_from(source),
                }
            )
    return records


def run_judge() -> None:
    from evaluation.qwen_judge import QwenJudge
    from evaluation.schema import Evidence

    pilot = load_pilot()
    config = pilot.load_config()
    judge_config = config["llm_judge"]
    finished = done_keys(JUDGE_OUTPUT)
    pending = [record for record in all_records(pilot) if (record["example_id"], record["condition"]) not in finished]
    if not pending:
        print("judge already complete")
        return
    judge = QwenJudge(pilot.CONFIG_PATH)
    JUDGE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with JUDGE_OUTPUT.open("a", encoding="utf-8") as handle:
        for record in pending:
            output = {
                "evaluator_version": VERSION,
                "human_validated": False,
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
                "matches": [],
                "judge_error": None,
            }
            if record["generation_error"] == "no_label2_evidence":
                output["generation_status"] = "gold_not_applicable_no_label2"
            elif record["generation_error"]:
                output["generation_status"] = "generation_error"
                output["judge_error"] = record["generation_error"]
            else:
                output["generation_status"] = "ok"
                try:
                    claims, _claim_meta = judge.extract_answer_claims(record["question"], record["answer"])
                    output["claims"] = [
                        {"span": claim.original_span, "claim": claim.text, "kind": claim.status}
                        for claim in claims
                    ]
                    if record["evidence"]:
                        evidence = [
                            Evidence(evidence_id=item["evidence_id"], text=item["text"])
                            for item in record["evidence"]
                        ]
                        facts, _fact_meta = judge.extract_supported_facts(record["question"], evidence)
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
                        aligned = [fact for fact in facts if fact.alignment_status == "aligned" and fact.text]
                        if aligned:
                            matches, _match_meta = judge.match_answer_to_facts(
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
                except Exception as exc:  # noqa: BLE001
                    output["judge_error"] = f"{type(exc).__name__}: {exc}"
                    output["generation_status"] = "evaluation_error"
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{output['example_id']} {output['condition']} "
                f"claims={len(output['claims'])} facts={len(output['facts'])} error={output['judge_error']}",
                flush=True,
            )
    judge.close()


def write_reports() -> None:
    rows = [json.loads(line) for line in NLI_OUTPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads(SUMMARY_OUTPUT.read_text(encoding="utf-8"))
    counts = {"supported": 0, "unsupported": 0, "contradicted": 0, "uncertain": 0, "evaluation_error": 0}
    reasons = {}
    fact_counts = {}
    for row in rows:
        for claim in row.get("claim_judgments", []):
            status = claim.get("nli_status")
            if status in counts:
                counts[status] += 1
            if status:
                reason = claim.get("decision_reason") or status
                reasons[reason] = reasons.get(reason, 0) + 1
        for fact in row.get("facts", []):
            status = fact.get("fact_status") or "missing"
            fact_counts[status] = fact_counts.get(status, 0) + 1
    by_key = {(row["example_id"], row["condition"]): row for row in rows}
    paired = []
    for example_id in sorted({row["example_id"] for row in rows}):
        standard = by_key[(example_id, "standard_top3")]
        gold = by_key[(example_id, "gold_evidence_qa")]
        if standard["generation_status"] != "ok" or gold["generation_status"] != "ok":
            continue
        paired.append(
            {
                "example_id": example_id,
                "qid": standard["qid"],
                "standard_ucr": standard["ucr"]["ucr"],
                "standard_ucr_undefined_reason": standard["ucr"]["undefined_reason"],
                "gold_ucr": gold["ucr"]["ucr"],
                "gold_ucr_undefined_reason": gold["ucr"]["undefined_reason"],
                "standard_coverage": standard["coverage_evidence_conditional"]["coverage"],
                "standard_coverage_numerator": standard["coverage_evidence_conditional"]["numerator"],
                "standard_coverage_denominator": standard["coverage_evidence_conditional"]["denominator"],
                "standard_coverage_undefined_reason": standard["coverage_evidence_conditional"]["undefined_reason"],
                "gold_coverage": gold["coverage_evidence_conditional"]["coverage"],
                "gold_coverage_numerator": gold["coverage_evidence_conditional"]["numerator"],
                "gold_coverage_denominator": gold["coverage_evidence_conditional"]["denominator"],
                "gold_coverage_undefined_reason": gold["coverage_evidence_conditional"]["undefined_reason"],
                "standard_evidence_count": len(standard["evidence"]),
                "gold_evidence_count": len(gold["evidence"]),
            }
        )
    summary["evaluator_version"] = VERSION
    summary["human_validated"] = False
    summary["label"] = "preliminary automated evaluation, not human-validated ground truth"
    summary["claim_status_counts"] = counts
    summary["ucr_counting"] = {
        "supported": "denominator only",
        "unsupported": "numerator and denominator",
        "contradicted": "numerator and denominator, also counted separately; included in the unsupported tally",
        "uncertain": "excluded from primary UCR",
        "evaluation_error": "excluded from primary UCR",
    }
    summary["decision_reasons"] = reasons
    summary["fact_status_counts"] = fact_counts
    summary["paired_23_count"] = len(paired)
    summary["gold_not_applicable"] = sum(row["generation_status"] == "gold_not_applicable_no_label2" for row in rows)
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with PAIRED_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    print(json.dumps({"claim_status_counts": counts, "paired": len(paired), "fact_status_counts": fact_counts}, indent=2))


def main() -> None:
    script = str(Path(__file__).resolve())
    subprocess.check_call([sys.executable, script, "judge"])
    pilot = load_pilot()
    pilot.run_nli(JUDGE_OUTPUT, NLI_OUTPUT, SUMMARY_OUTPUT)
    write_reports()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "judge":
        run_judge()
    elif len(sys.argv) > 1 and sys.argv[1] == "report":
        write_reports()
    else:
        main()
