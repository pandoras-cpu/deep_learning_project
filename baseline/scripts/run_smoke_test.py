"""Generate answers for manually inspected question-evidence examples.

Labels and reference answers are loaded only to record what was inspected.
They are not passed to the generator.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import pandas as pd
import torch
import transformers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generator import Evidence, EvidenceGenerator  # noqa: E402

EXAMPLES_PATH = ROOT / "data" / "eval" / "smoke_examples.json"
CONFIG_PATH = ROOT / "configs" / "baseline.yaml"
OUTPUT_PATH = ROOT / "results" / "smoke_test.jsonl"
DATA_DIR = ROOT / "external" / "contextual-product-qa" / "ePQA"


def load_split(split: str) -> pd.DataFrame:
    frame = pd.read_csv(
        DATA_DIR / f"{split}.csv",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    for column in frame.columns:
        frame[column] = frame[column].str.strip()
    frame["file_line"] = frame.index + 2
    return frame


def main() -> None:
    examples = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    frames: dict[str, pd.DataFrame] = {}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    generator = EvidenceGenerator(CONFIG_PATH)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for example in examples:
            split = example["split"]
            if split not in frames:
                frames[split] = load_split(split)
            frame = frames[split]
            rows = []
            for file_line in example["file_lines"]:
                matched = frame.loc[frame["file_line"].eq(file_line)]
                if len(matched) != 1:
                    raise SystemExit(f"file line {split}:{file_line} matched {len(matched)} rows")
                rows.append(matched.iloc[0])
            qids = {row["qid"] for row in rows}
            questions = {row["question"] for row in rows}
            asins = {row["ASIN"] for row in rows}
            if len(qids) != 1 or len(questions) != 1 or len(asins) != 1:
                raise SystemExit(f"{example['example_id']} mixes questions or products")
            evidence = [
                Evidence(evidence_id=f"{split}:{int(row['file_line'])}", text=row["candidate"])
                for row in rows
            ]
            record = {
                "example_id": example["example_id"],
                "qid": rows[0]["qid"],
                "asin": rows[0]["ASIN"],
                "question": rows[0]["question"],
                "evidence": [
                    {
                        "evidence_id": item.evidence_id,
                        "text": item.text,
                        "qa_pair_id": row["qa_pair_id"],
                        "source": row["source"],
                    }
                    for item, row in zip(evidence, rows)
                ],
                "prompt_version": generator.prompt_version,
                "model_id": generator.config["model_id"],
                "torch_dtype": generator.config["torch_dtype"],
                "device": "mps",
                "decoding": generator.config["decoding"],
                "max_input_tokens": generator.max_input_tokens,
                "package_versions": {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                },
                "inspection_note": example["inspection_note"],
                "label_not_in_prompt": [row["label"] for row in rows],
                "reference_answer_not_in_prompt": [row["answer"] for row in rows],
            }
            try:
                result = generator.generate(rows[0]["question"], evidence)
            except Exception as exc:  # noqa: BLE001
                record.update(
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
                for row in rows:
                    if row["label"] and row["label"] in rendered:
                        pass
                    if row["answer"] and row["answer"] in rendered and row["answer"] not in row["candidate"]:
                        raise SystemExit(
                            f"{example['example_id']} reference answer leaked into the prompt"
                        )
                record.update(
                    {
                        "answer": result.answer,
                        "messages": result.messages,
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                        "error": result.error,
                    }
                )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{record['example_id']} error={record['error']} answer={record['answer']}")


if __name__ == "__main__":
    main()
