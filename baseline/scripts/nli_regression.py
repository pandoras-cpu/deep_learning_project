"""Focused NLI checks for the baseline-01 false positive and baseline-04 false negative."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.metrics import aggregate_passage_statuses  # noqa: E402
from evaluation.nli_verifier import NliVerifier  # noqa: E402
from evaluation.schema import Evidence  # noqa: E402
from evaluation.span_align import align_quote  # noqa: E402

OUTPUT = ROOT / "results" / "evaluation" / "nli_regression.json"
B01_EVIDENCE = [
    Evidence(
        "dev:6972",
        "you can if you want put on multiple photo's and even crop them to different shapes. "
        "Question: can you create a posterized version of your photos?",
    ),
    Evidence("dev:6974", "line tool editor-add points or curves to your lines to create distinct, custom, lines"),
    Evidence(
        "dev:6975",
        "easiest is to use a preset template and use clipart that comes with program. "
        "Question: is this program compatible with clip art or is it like clip art?",
    ),
]
B01_CLAIM = "You can custom crop your photos by putting on multiple photos and cropping them to different shapes."
B04_IF = "if you take the wheels off, the table will fall over."
B04_EVIDENCE = [
    Evidence(
        "dev:7875",
        "rayburn, i don't believe the sturdiness of the table would have anything to do with the casters. "
        "Question: hello, is the table sturdy enough to use without the casters (wheels)?",
    ),
    Evidence(
        "dev:7873",
        "note: it make take a small tap on the handle end of the screw mechanism to release it."
        "the casters come mounted on the cross bars.",
    ),
    Evidence(
        "dev:7874",
        "if you take the wheels off, the table will fall over. "
        "Question: can anyone tell me if this would fit under my softa with 1-1/2\" of space if i took the wheels off?",
    ),
]
B04_CLAIM = "The caster (wheels) can be removed after receiving the product."


def main() -> None:
    failures = []
    quote = 'measures exterior : 22"|l x 17"|w x 11.5"|h interior : 18"|l x 13"|w x 3"|h for pets up to 25lbs'
    evidence = 'measures exterior : 22|"l x 17|"w x 11.5|"h interior : 18|"l x 13|"w x 3|"h for pets up to 25lbs'
    aligned = align_quote("dev:2054", evidence, quote)
    if aligned["alignment_status"] != "aligned" or aligned["aligned_span"] != evidence:
        failures.append("punctuation alignment did not preserve the original evidence span")
    missing = align_quote("dev:1", "The camera supports 4K recording.", "completely unrelated waterproof claim text")
    if missing["alignment_status"] != "alignment_failure" or missing["evidence_text"] != "The camera supports 4K recording.":
        failures.append("alignment failure was not logged with the original evidence")
    if aggregate_passage_statuses(["supported", "contradicted"])[0] != "uncertain":
        failures.append("passage disagreement was not uncertain")

    verifier = NliVerifier(ROOT / "configs" / "evaluation_v1.yaml")
    if verifier.id2label != {0: "entailment", 1: "neutral", 2: "contradiction"}:
        failures.append(f"unexpected id2label {verifier.id2label}")
    control = verifier.score_texts("The camera supports 4K recording.", "The camera is waterproof.")
    if control.status == "supported" or control.truncated:
        failures.append("control missing-fact case was supported or truncated")
    swapped = verifier.score_texts("The camera is waterproof.", "The camera supports 4K recording.")
    if swapped.status == "supported":
        failures.append("reversed unrelated pair was supported")

    bundled = verifier.score_claim(B01_EVIDENCE, B01_CLAIM)
    if bundled.status == "supported":
        failures.append("baseline-01 custom-crop claim remained supported despite passage disagreement")
    if bundled.truncated or any(item["truncated"] for item in bundled.passage_judgments):
        failures.append("baseline-01 input was truncated")
    passage_6972 = next(item for item in bundled.passage_judgments if item["evidence_id"] == "dev:6972")
    passage_6975 = next(item for item in bundled.passage_judgments if item["evidence_id"] == "dev:6975")
    if passage_6972["status"] != "supported" or passage_6975["status"] != "contradicted":
        failures.append("baseline-01 per-passage labels changed unexpectedly")

    removed = verifier.score_claim(B04_EVIDENCE, B04_CLAIM)
    if removed.truncated:
        failures.append("baseline-04 input was truncated")
    if_sentence = verifier.score_texts(B04_IF, B04_CLAIM)
    narrow = verifier.score_texts(B04_IF, "The wheels can be taken off.")
    if if_sentence.status == "supported":
        failures.append("the bundled after-receipt claim was entailed by the conditional sentence")
    if narrow.status != "supported":
        failures.append("the narrower wheels-can-be-taken-off claim was not entailed")

    payload = {
        "id2label": verifier.id2label,
        "baseline_01": {
            "status": bundled.status,
            "reason": bundled.decision_reason,
            "truncated": bundled.truncated,
            "passages": bundled.passage_judgments,
        },
        "baseline_04_bundled_claim": {
            "status": removed.status,
            "reason": removed.decision_reason,
            "truncated": removed.truncated,
            "passages": removed.passage_judgments,
            "if_sentence_status": if_sentence.status,
            "if_sentence_scores": if_sentence.scores,
            "narrow_claim_status": narrow.status,
            "narrow_claim_scores": narrow.scores,
        },
        "alignment_original_span": aligned["aligned_span"],
        "failures": failures,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"failures": failures, "b01": bundled.status, "b04": removed.status, "narrow": narrow.status}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
