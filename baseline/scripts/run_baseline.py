"""Standard RAG baseline: within-pool retrieval, then the frozen generator.

Retrieval and generation run in separate processes. Torch and FAISS each ship
an OpenMP runtime, and loading both in one process aborts on this machine.
Labels and reference answers are stored only for later audit.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SPLIT = "dev"
SEED = 42
SAMPLE_SIZE = 30
SAMPLE_PATH = ROOT / "data" / "eval" / "baseline_dev_sample.json"
RETRIEVED_PATH = ROOT / "results" / "baseline_dev_retrieved.jsonl"
OUTPUT_PATH = ROOT / "results" / "baseline_dev.jsonl"
CONFIG_PATH = ROOT / "configs" / "baseline.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_sample() -> None:
    from corpus import load_split

    frame = load_split(ROOT, SPLIT)
    qids = sorted(frame["qid"].unique())
    chosen = random.Random(SEED).sample(qids, SAMPLE_SIZE)
    rows = []
    for number, qid in enumerate(chosen, start=1):
        question_rows = frame.loc[frame["qid"].eq(qid)]
        rows.append(
            {
                "example_id": f"baseline-{number:02d}",
                "split": SPLIT,
                "qid": qid,
                "asin": question_rows["ASIN"].iloc[0],
                "question": question_rows["question"].iloc[0],
            }
        )
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(
        json.dumps(
            {"seed": SEED, "split": SPLIT, "sample_size": SAMPLE_SIZE, "examples": rows},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {SAMPLE_PATH.relative_to(ROOT)}")


def retrieve() -> None:
    import numpy as np

    from candidate_index import CandidateIndex
    from corpus import load_split

    config = load_config()
    top_k = int(config["retriever"]["top_k"])
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    query_rows = json.loads((ROOT / "indexes" / SPLIT / "query_rows.json").read_text(encoding="utf-8"))
    vectors = np.load(ROOT / "indexes" / SPLIT / "query_vectors.npy")
    position = {qid: index for index, qid in enumerate(query_rows["qids"])}
    candidate_index = CandidateIndex.load(ROOT / config["retriever"]["index_dir"] / SPLIT)
    frame = load_split(ROOT, SPLIT)
    by_evidence = frame.set_index("evidence_id")
    RETRIEVED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RETRIEVED_PATH.open("w", encoding="utf-8") as handle:
        for example in sample["examples"]:
            hits = candidate_index.search(example["qid"], vectors[position[example["qid"]]], top_k)
            if any(hit.qid != example["qid"] for hit in hits):
                raise SystemExit(f"{example['example_id']} retrieved another question")
            record = {
                "example_id": example["example_id"],
                "split": SPLIT,
                "qid": example["qid"],
                "asin": example["asin"],
                "question": example["question"],
                "retrieved": [
                    {
                        "evidence_id": hit.evidence_id,
                        "score": hit.score,
                        "text": hit.text,
                    }
                    for hit in hits
                ],
                "audit_not_for_model": [
                    {
                        "evidence_id": hit.evidence_id,
                        "label": by_evidence.at[hit.evidence_id, "label"],
                        "reference_answer": by_evidence.at[hit.evidence_id, "answer"],
                        "source": hit.source,
                        "qa_pair_id": hit.qa_pair_id,
                    }
                    for hit in hits
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {RETRIEVED_PATH.relative_to(ROOT)}")


def generate(offset: int, limit: int) -> None:
    import torch
    import transformers

    from generator import Evidence, EvidenceGenerator

    config = load_config()
    records = [
        json.loads(line)
        for line in RETRIEVED_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = records[offset : offset + limit]
    generator = EvidenceGenerator(CONFIG_PATH)
    mode = "w" if offset == 0 else "a"
    with OUTPUT_PATH.open(mode, encoding="utf-8") as handle:
        for record in selected:
            evidence = [
                Evidence(evidence_id=item["evidence_id"], text=item["text"])
                for item in record["retrieved"]
                if item["text"].strip()
            ]
            output = {
                "example_id": record["example_id"],
                "qid": record["qid"],
                "asin": record["asin"],
                "question": record["question"],
                "retrieved": record["retrieved"],
                "prompt_version": generator.prompt_version,
                "model_id": config["model_id"],
                "model_revision": config["revision"],
                "retriever_model_id": config["retriever"]["model_id"],
                "retriever_revision": config["retriever"]["revision"],
                "top_k": config["retriever"]["top_k"],
                "search": "within question candidate pool",
                "torch_dtype": config["torch_dtype"],
                "device": "mps",
                "decoding": config["decoding"],
                "sample_seed": SEED,
                "package_versions": {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                },
                "audit_not_for_model": record["audit_not_for_model"],
            }
            try:
                result = generator.generate(record["question"], evidence)
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
                for item in record["audit_not_for_model"]:
                    answer = item["reference_answer"]
                    passage = next(
                        hit["text"] for hit in record["retrieved"] if hit["evidence_id"] == item["evidence_id"]
                    )
                    if answer and answer in rendered and answer not in passage and answer not in record["question"]:
                        raise SystemExit(f"{record['example_id']} reference answer leaked into the prompt")
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
            print(f"{output['example_id']} error={output['error']} answer={output['answer']}")


def main() -> None:
    select_sample()
    script = str(Path(__file__).resolve())
    subprocess.check_call([sys.executable, script, "retrieve"])
    subprocess.check_call([sys.executable, script, "generate", "0", "5"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "retrieve":
        retrieve()
    elif len(sys.argv) > 1 and sys.argv[1] == "generate":
        generate(int(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
