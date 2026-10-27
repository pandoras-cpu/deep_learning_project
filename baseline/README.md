# ePQA baseline

Local baseline for evidence-grounded product QA on Apple Silicon. No paid APIs.

## What is frozen

- Retriever: `BAAI/bge-small-en-v1.5` revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, question-level Top-3, dev index only.
- First generator run: `Qwen/Qwen3-4B-Instruct-2507`, prompt `evidence-only-v1`, outputs in `results/baseline_dev.jsonl`.
- Current generator run: `Qwen/Qwen2.5-3B-Instruct` revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, prompt `evidence-only-v2`, outputs in `results/baseline_v2/`.
- Entailment: DeBERTa `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` revision `6f5cf0a2b59cabb106aca4c287eed12e357e90eb`, CPU, per-passage aggregation from preliminary-v1.

The two generator runs use different models and prompts. Their UCR and coverage numbers are not a controlled comparison.

## Commands

From `Baseline/`, with `.venv` active:

```bash
# retrieval diagnostics already saved; retriever was not changed
# results/retrieval_sanity.json

# generate the 30 saved dev questions from results/baseline_dev_retrieved.jsonl
.venv/bin/python scripts/run_baseline_v2.py

# two-question smoke, plus the empty-evidence check
.venv/bin/python scripts/run_baseline_v2.py --smoke

# evaluate: Qwen3 extracts claims and evidence-supported facts, then DeBERTa scores them
.venv/bin/python scripts/run_evaluation_v2.py
```

Load one large model at a time. Do not import `torch` and `faiss` in the same process.

## Outputs

- `results/baseline_v2/answers.jsonl`: question, Top-3 evidence, answer, decoding.
- `results/baseline_v2/aspects.jsonl`: requested question parts. These do not enter the coverage denominator.
- `results/baseline_v2/evaluations.jsonl`: claims, facts, NLI decisions, UCR, coverage.
- `results/baseline_v2/summary.json`: micro UCR and evidence-conditioned coverage with numerator, denominator, and undefined counts.

Labels and reference answers stay in `audit_not_for_model` on the retrieval file. They are not sent to the generator or the support judge.

Automated scores are preliminary. They are not human-validated.
