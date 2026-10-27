"""Generate the Qwen2.5 baseline from the saved Top-3 retrieval. Does not overwrite v1 answers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.prompts import QUESTION_ASPECTS_V1
from generator import Evidence, EvidenceGenerator

CONFIG = ROOT / "configs" / "baseline_v2.yaml"
RETRIEVED = ROOT / "results" / "baseline_dev_retrieved.jsonl"
OUT_DIR = ROOT / "results" / "baseline_v2"
ANSWERS = OUT_DIR / "answers.jsonl"
ASPECTS = OUT_DIR / "aspects.jsonl"
SMOKE = OUT_DIR / "smoke.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def aspects_for(generator: EvidenceGenerator, question: str) -> dict:
    text, input_tokens, output_tokens = generator.complete(
        [
            {"role": "system", "content": QUESTION_ASPECTS_V1},
            {"role": "user", "content": f"Question:\n{question.strip()}"},
        ]
    )
    record = {
        "prompt_version": "question-aspects-v1",
        "model_id": generator.config["model_id"],
        "model_revision": generator.config["revision"],
        "raw": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "aspects": [],
        "error": None,
    }
    try:
        start, end = text.find("{"), text.rfind("}")
        payload = json.loads(text[start : end + 1])
        record["aspects"] = [str(item).strip() for item in payload.get("aspects", []) if str(item).strip()]
    except (json.JSONDecodeError, ValueError, AttributeError) as exc:
        record["error"] = str(exc)
    return record


def generate_row(generator: EvidenceGenerator, row: dict) -> dict:
    evidence = [Evidence(item["evidence_id"], item["text"]) for item in row["retrieved"]]
    result = generator.generate(row["question"], evidence)
    for audit in row["audit_not_for_model"]:
        reference = audit.get("reference_answer") or ""
        if reference and reference in (result.answer or ""):
            raise RuntimeError(f"reference answer leaked into {row['example_id']}")
    return {
        "example_id": row["example_id"],
        "qid": row["qid"],
        "asin": row["asin"],
        "split": "dev",
        "condition": "standard_rag",
        "question": row["question"],
        "prompt_version": result.prompt_version,
        "model_id": generator.config["model_id"],
        "model_revision": generator.config["revision"],
        "decoding": generator.config["decoding"],
        "messages": result.messages,
        "retrieved": row["retrieved"],
        "answer": result.answer,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "error": result.error,
        "generation_status": "ok" if result.error is None else result.error,
    }


def main() -> None:
    smoke = "--smoke" in sys.argv
    rows = read_jsonl(RETRIEVED)
    if smoke:
        rows = rows[:2]
    done_answers = {row["example_id"] for row in read_jsonl(ANSWERS)}
    done_aspects = {row["example_id"] for row in read_jsonl(ASPECTS)}
    generator = EvidenceGenerator(CONFIG)
    if smoke:
        empty = generator.generate("Does it support 4K?", [])
        append(SMOKE, {
            "case": "empty_evidence",
            "error": empty.error,
            "answer": empty.answer,
            "model_called": empty.input_tokens is not None,
        })
    for row in rows:
        example_id = row["example_id"]
        if example_id not in done_answers:
            answer = generate_row(generator, row)
            append(ANSWERS if not smoke else SMOKE, answer)
            print(example_id, answer["generation_status"], answer["output_tokens"], flush=True)
        if example_id not in done_aspects:
            aspect = aspects_for(generator, row["question"])
            aspect["example_id"] = example_id
            aspect["qid"] = row["qid"]
            append(ASPECTS if not smoke else SMOKE, {"case": "aspects", **aspect})
            print(example_id, "aspects", len(aspect["aspects"]), aspect["error"], flush=True)
    if not smoke:
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        manifest = {
            "generator_config": "configs/baseline_v2.yaml",
            "model_id": config["model_id"],
            "revision": config["revision"],
            "prompt_version": config["prompt_version"],
            "retrieved_input": "results/baseline_dev_retrieved.jsonl",
            "retrieval_recomputed": False,
            "answers": str(ANSWERS.relative_to(ROOT)),
            "aspects": str(ASPECTS.relative_to(ROOT)),
            "answer_count": len(read_jsonl(ANSWERS)),
        }
        (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
