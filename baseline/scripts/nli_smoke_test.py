"""Five synthetic NLI checks. Does not read baseline answers."""

from __future__ import annotations

import json
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.judge import UnconfiguredJudge  # noqa: E402
from evaluation.nli_verifier import NliVerifier  # noqa: E402
from evaluation.schema import Evidence  # noqa: E402

CONFIG = ROOT / "configs" / "evaluation_v1.yaml"
OUTPUT = ROOT / "results" / "evaluation" / "nli_smoke_test.json"
EVIDENCE = [
    Evidence(evidence_id="synthetic:4k", text="The camera supports 4K recording.")
]


def judgment_row(name: str, claim: str, verifier: NliVerifier, official_status: str | None = None) -> dict:
    if official_status is None:
        scored = verifier.score_claim(EVIDENCE, claim)
        return {
            "name": name,
            "claim": claim,
            "status": scored.status,
            "nli_label": scored.nli_label,
            "scores": scored.scores,
            "truncated": scored.truncated,
            "max_probability": scored.max_probability,
        }
    return {
        "name": name,
        "claim": claim,
        "status": official_status,
        "nli_label": None,
        "scores": None,
        "truncated": False,
        "max_probability": None,
        "note": "Meta-statement about missing evidence. Not sent to NLI and not counted as a product claim.",
    }


def main() -> None:
    verifier = NliVerifier(CONFIG)
    try:
        UnconfiguredJudge().extract_answer_claims("q", "a")
    except RuntimeError as exc:
        judge_status = str(exc)
    else:
        raise SystemExit("unconfigured judge returned claims")

    compound = "The camera supports 4K recording and is waterproof."
    rows = [
        judgment_row("supported_4k", "The camera supports 4K recording.", verifier),
        judgment_row("missing_waterproof", "The camera is waterproof.", verifier),
        judgment_row("negated_4k", "The camera does not support 4K recording.", verifier),
        {
            "name": "two_atomic_claims",
            "answer": compound,
            "split_by": "smoke_test_manual_not_llm",
            "claims": [
                judgment_row("supports_4k", "The camera supports 4K recording.", verifier),
                judgment_row("waterproof", "The camera is waterproof.", verifier),
            ],
        },
        judgment_row(
            "abstention_about_waterproofing",
            "The evidence does not specify waterproofing.",
            verifier,
            official_status="not_a_factual_claim",
        ),
    ]
    failures = []
    if rows[0]["status"] != "supported":
        failures.append("4K restatement was not supported")
    if rows[1]["status"] == "supported":
        failures.append("waterproof claim was marked supported")
    if rows[2]["status"] not in {"contradicted", "unsupported"}:
        failures.append("negated 4K claim was not contradicted or unsupported")
    split = rows[3]["claims"]
    if [item["status"] for item in split] != ["supported", "unsupported"]:
        failures.append(f"compound split statuses were {[item['status'] for item in split]}")
    if rows[4]["status"] != "not_a_factual_claim":
        failures.append("abstention was treated as a product claim")

    payload = {
        "model_id": verifier.model_id,
        "revision": verifier.revision,
        "device": verifier.device,
        "id2label": verifier.id2label,
        "load_seconds": round(verifier.load_seconds, 3),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1),
        "decision": "argmax",
        "judge": judge_status,
        "tests": rows,
        "failures": failures,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
