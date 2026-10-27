"""Gold Evidence QA for the frozen 30-question dev sample.

This is not retrieval. Evidence is every non-empty label=2 candidate for the
question. The generator, prompt, and decoding match the standard RAG run.
Labels are used only to choose this condition's evidence.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import torch
import transformers
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from corpus import load_split  # noqa: E402
from generator import Evidence, EvidenceGenerator  # noqa: E402

SAMPLE_PATH = ROOT / "data" / "eval" / "baseline_dev_sample.json"
CONFIG_PATH = ROOT / "configs" / "baseline.yaml"
OUTPUT_PATH = ROOT / "results" / "gold_evidence_dev.jsonl"


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    frame = load_split(ROOT, "dev")
    generator = EvidenceGenerator(CONFIG_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for example in sample["examples"]:
            rows = frame.loc[
                frame["qid"].eq(example["qid"])
                & frame["label"].eq("2")
                & frame["candidate"].ne("")
            ].sort_values("file_line")
            evidence = [
                Evidence(evidence_id=row.evidence_id, text=row.candidate)
                for row in rows.itertuples(index=False)
            ]
            output = {
                "condition": "gold_evidence_qa",
                "example_id": example["example_id"],
                "qid": example["qid"],
                "asin": example["asin"],
                "question": example["question"],
                "retrieved": [
                    {"evidence_id": item.evidence_id, "score": None, "text": item.text}
                    for item in evidence
                ],
                "prompt_version": generator.prompt_version,
                "model_id": config["model_id"],
                "model_revision": config["revision"],
                "decoding": config["decoding"],
                "sample_seed": sample["seed"],
                "package_versions": {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                },
                "audit_not_for_model": [
                    {
                        "evidence_id": row.evidence_id,
                        "label": row.label,
                        "reference_answer": row.answer,
                        "source": row.source,
                        "qa_pair_id": row.qa_pair_id,
                    }
                    for row in rows.itertuples(index=False)
                ],
            }
            if not evidence:
                output.update(
                    {
                        "answer": None,
                        "messages": [],
                        "input_tokens": None,
                        "output_tokens": None,
                        "error": "no_label2_evidence",
                    }
                )
            else:
                try:
                    result = generator.generate(example["question"], evidence)
                except Exception as exc:  # noqa: BLE001
                    output.update(
                        {
                            "answer": None,
                            "messages": None,
                            "input_tokens": None,
                            "output_tokens": None,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                else:
                    rendered = "\n".join(message["content"] for message in result.messages)
                    for item in output["audit_not_for_model"]:
                        answer = item["reference_answer"]
                        passage = next(
                            hit["text"]
                            for hit in output["retrieved"]
                            if hit["evidence_id"] == item["evidence_id"]
                        )
                        if (
                            answer
                            and answer in rendered
                            and answer not in passage
                            and answer not in example["question"]
                        ):
                            raise SystemExit(
                                f"{example['example_id']} reference answer leaked into the prompt"
                            )
                    output.update(
                        {
                            "answer": result.answer,
                            "messages": result.messages,
                            "input_tokens": result.input_tokens,
                            "output_tokens": result.output_tokens,
                            "error": result.error,
                        }
                    )
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"{output['example_id']} gold_n={len(evidence)} "
                f"error={output['error']} answer={output['answer']}"
            )


if __name__ == "__main__":
    main()
