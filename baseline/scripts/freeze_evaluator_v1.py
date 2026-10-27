"""Record the current evaluator as preliminary v1 without changing its rules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "evaluation_v1.yaml"
OUTPUT = ROOT / "results" / "evaluation" / "evaluator_preliminary_v1.json"

FILES = [
    "configs/evaluation_v1.yaml",
    "src/evaluation/prompts.py",
    "src/evaluation/metrics.py",
    "src/evaluation/nli_verifier.py",
    "src/evaluation/span_align.py",
    "src/evaluation/qwen_judge.py",
    "src/evaluation/schema.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = {
        "evaluator_version": "preliminary-v1",
        "human_validated": False,
        "label": "preliminary automated evaluation, not human-validated ground truth",
        "config": "configs/evaluation_v1.yaml",
        "file_sha256": {name: sha256(ROOT / name) for name in FILES},
        "judge": config["llm_judge"],
        "nli": config["nli"],
        "metrics": config["metrics"],
        "ucr_counting": {
            "supported": "excluded from the numerator, included in the denominator",
            "unsupported": "included in the numerator and the denominator",
            "contradicted": "subtype of unsupported: included in the numerator and the denominator, also counted separately",
            "uncertain": "excluded from primary UCR; included as unsupported in ucr_conservative",
            "evaluation_error": "excluded from primary UCR and from ucr_conservative",
            "no_factual_claim": "UCR undefined, not zero",
        },
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
