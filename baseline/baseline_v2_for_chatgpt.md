# ePQA baseline v2 — handoff for ChatGPT

Read this file as the description of the current baseline. It was run on 2026-10-26. There was no training and no fine-tuning. All scores below are automatic and have not been checked by a human.

Do not mix this file with the earlier Qwen3-4B run. That older run used a different generator, a different prompt, and a different judge assignment. Its UCR and coverage numbers are not a controlled comparison with the numbers in this file. Retrieval is the one shared piece: the retriever, index, and Top-3 evidence are the same as that earlier run, so the retrieval metrics below were not recomputed.

## Task

Dataset: Amazon Science ePQA, official repository `amazon-science/contextual-product-qa`, commit `cec976cc2f2218aa76e562ecc9c7d79f1f3271bc`. Files are comma-separated. The released CSVs have no `context` column. `title` is the product title. Retrieval text is the `candidate` field only.

Each question (`qid`) has its own candidate pool of about 10 passages and one ASIN. The task is to rank passages inside that question's pool, then answer from the selected passages. It is not open-corpus retrieval and not product-wide retrieval. Evidence id is `split:file_line`.

Labels: `2` = fully answers the question, `1` = helpful but does not fully answer, `0` = irrelevant. `label` and the reference `answer` are never shown to retrieval, the generator prompt, or the support judge. They are used only to score retrieval and for offline audit.

Dev split: 977 questions, 9770 rows, exactly 10 candidates per question. The dev index has 9769 vectors because 1 empty candidate was excluded. Only the dev index exists. Train and test indexes were not built.

## What this version does

1. Retrieval. BGE embeds the question and each candidate in that question's pool. FAISS inner product selects Top-3. Search is restricted to the same `qid`. This step was not rerun. The saved dev index and the saved Top-3 file were reused because the retriever config is identical to the previous run.
2. Generation. Qwen2.5-3B-Instruct writes one answer per question from the question plus those three passages. All 30 sampled questions were regenerated for this version.
3. Question decomposition. The same Qwen2.5 model lists the aspects the question asks for. These aspects are saved for audit. They do not enter the coverage denominator.
4. Evaluation. Qwen3-4B extracts atomic claims from the answer and extracts required facts that the supplied Top-3 evidence supports. DeBERTa checks each claim against each passage, and checks each extracted fact against its cited span. Metrics are deterministic Python.

The evaluator never chooses evidence and never changes the answer. Large models are loaded one at a time. Machine: Mac M1 Pro, 16 GB unified memory, no NVIDIA GPU. Weights are float16. This is a device cast from bfloat16, not int8 or int4 quantization. No paid API.

## Models and parameters

Generator config: `configs/baseline_v2.yaml`. Evaluator config: `configs/evaluation_v2.yaml`.

| Role | Model | Revision | Settings |
| --- | --- | --- | --- |
| Retriever | `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | Query prefix exactly `Represent this sentence for searching relevant passages: ` (trailing space required). Passages have no prefix. CLS pooling, L2 normalize, FAISS `IndexFlatIP`, max length 512, Top-3, within-question search. Device MPS. Index: `indexes/dev`, 9769 vectors, dim 384. `labels_in_index` is false. |
| Generator | `Qwen/Qwen2.5-3B-Instruct` | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | Prompt `evidence-only-v2`. Greedy: `do_sample: false`. `max_new_tokens: 256`. `max_input_tokens: 2048`. float16, load on CPU then move to MPS. |
| Question aspects | same Qwen2.5 checkpoint | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | Prompt `question-aspects-v1`. Greedy, `max_new_tokens: 256`. Evaluation only. The model is not shown the evidence. |
| Claim and fact extraction | `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` | Prompts `claim-extraction-v2`, `supported-facts-v2`, `fact-match-v2`. Greedy, `max_new_tokens: 512`. Evaluation only. |
| Entailment | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | `6f5cf0a2b59cabb106aca4c287eed12e357e90eb` | CPU. Premise = evidence text. Hypothesis = claim or extracted fact. Decision = argmax. Labels: 0 entailment, 1 neutral, 2 contradiction. Low confidence if max probability < 0.5. |

Generator and judge are the same model family and different checkpoints. That reduces direct self-scoring, but it does not make the evaluation independent. `same_model_judge` is false.

Generator system prompt (`evidence-only-v2`):

> You answer a customer question about one product using only the evidence passages in the user message. Use only the supplied evidence. Do not use outside knowledge. If the question has several parts, answer every part the evidence supports. For a part the evidence does not address, say that it is not stated. Do not turn a missing fact into a negative answer. Absence of evidence is not evidence that the answer is no. Do not invent product properties, specifications, compatibility, prices, or policies.

## Sample and conditions

- Split: dev.
- Sample: `random.Random(42).sample(sorted unique dev qids, 30)`. Labels were not used to pick questions. Example ids are `baseline-01` through `baseline-30`. Manifest: `data/eval/baseline_dev_sample.json`.
- Condition reported here: standard RAG. Evidence is the saved Top-3 from `results/baseline_dev_retrieved.jsonl`.
- One answer per question. 30 of 30 answers were generated. No answer hit the 256-token cap. The longest output was 68 tokens. Output: `results/baseline_v2/answers.jsonl`.
- This version does not report a Gold Evidence QA condition. Gold is not part of the standard baseline.

Seven of the 30 questions have no `label=2` candidate in the full pool: `baseline-04`, `baseline-07`, `baseline-13`, `baseline-14`, `baseline-18`, `baseline-20`, `baseline-28`. Standard RAG still answered them from their Top-3. An empty evidence list returns `empty_evidence` and does not call the model. That case was smoke-tested and is not one of the 30.

## Metric definitions

Retrieval metrics are computed on all 977 dev questions. A question with no relevant candidate is undefined and is not counted as zero.

- Recall@3 for label 2: fraction of label-2 candidates placed in Top-3.
- Success@3 for label 2: 1 if Top-3 contains at least one label 2.
- MRR: reciprocal rank of the first relevant candidate.
- nDCG@3 gain = `2^label - 1` for labels 0, 1, and 2.

UCR uses atomic factual claims from the generated answer.

- A claim is supported only when at least one passage entails it and no passage contradicts it.
- Contradiction on any passage with no entailment makes the claim contradicted.
- Both entailment and contradiction make the claim uncertain (`passage_disagreement`).
- Neutral only makes the claim unsupported.
- A passage whose max probability is below 0.5 is treated as uncertain before aggregation.
- Micro UCR = (neutral unsupported + contradicted) / (supported + neutral unsupported + contradicted).
- Uncertain claims and evaluation errors are outside the primary UCR denominator.
- Abstentions and sentences that only say a fact is missing are `not_a_factual_claim` and are outside UCR.
- If an answer has no classifiable factual claim, its UCR is undefined, not zero.
- Macro UCR is the unweighted mean of per-answer UCR over answers where UCR is defined.

Answerable Coverage is evidence-conditioned. The denominator is required facts that the shown Top-3 evidence supports, after DeBERTa verifies the extracted fact against the aligned quote. The numerator is how many of those verified facts the answer actually expresses. Aspects from question decomposition are not the denominator. If no verified required fact exists, coverage is undefined, not zero. Coverage from two runs can be compared only when they use the same evidence scope.

## Retrieval results

Source: `results/retrieval_sanity.json`. Retriever was not rerun. 172 dev questions have no label 2. 56 have neither label 1 nor label 2.

| Metric | Value | Defined / undefined |
| --- | ---: | ---: |
| Recall@3, relevant = label 2 | 0.6373 | 805 / 172 |
| Success@3, at least one label 2 | 0.9093 | 805 / 172 |
| MRR, relevant = label 2 | 0.7858 | 805 / 172 |
| Recall@3, relevant = label 1 or 2 | 0.6108 | 921 / 56 |
| MRR, relevant = label 1 or 2 | 0.8650 | 921 / 56 |
| nDCG@3 | 0.7310 | 921 / 56 |

Top-3 usually contains at least one fully relevant passage when such a passage exists. It does not recover every label-2 passage: Recall@3 for label 2 is 0.6373.

## Generation and evaluation results

Source: `results/baseline_v2/summary.json`. Evaluator version `baseline-eval-v2`. `human_validated` is false. 30 standard-RAG answers.

| Metric | Micro | Numerator / denominator | Macro | Defined / undefined answers |
| --- | ---: | ---: | ---: | ---: |
| UCR | 0.4688 | 15 / 32 | 0.4375 | 20 / 10 |
| Answerable Coverage | 0.2800 | 7 / 25 | 0.2111 | 15 / 15 |

Claim counts across the 30 answers:

| Status | Count | How it enters UCR |
| --- | ---: | --- |
| supported | 17 | Denominator. Not in the numerator. |
| unsupported, neutral only | 6 | Numerator and denominator. |
| contradicted | 9 | Numerator and denominator. Also counted separately. |
| uncertain | 2 | Excluded from primary UCR. |
| evaluation error | 0 | Excluded. |
| not a factual claim | 17 | Excluded. |

The 15 in the UCR numerator is 6 neutral plus 9 contradicted. The 10 undefined UCR answers are 9 with no factual claim and 1 with only uncertain claims.

Fact counts: 25 verified, 2 rejected because the quote does not entail the extracted fact (`unsupported_by_citation`). The 15 undefined coverage answers all have reason `no_required_facts`. No alignment failure remained in the final fact statuses.

On the smoke questions, `baseline-02` still answered that the medium size would not be big enough, even though the prompt forbids treating missing evidence as a negative. That is observed model behavior on this prompt, not a separate metric.

## Files

- Answers: `results/baseline_v2/answers.jsonl`
- Aspects: `results/baseline_v2/aspects.jsonl`
- Judge records: `results/baseline_v2/judge.jsonl`
- Claim-level evaluation: `results/baseline_v2/evaluations.jsonl`
- Aggregate: `results/baseline_v2/summary.json`
- Run manifest: `results/baseline_v2/manifest.json`
- Previous generator outputs were not overwritten: `results/baseline_dev.jsonl`

There is no git commit for this run. The manifest stores SHA-256 hashes of the v2 config and scoring files.

## Not in this run

No human annotation. No Gold condition. No train or test index. No full-dev generation. No 100-question sample. No Top-K ablation. No reranker, Cross-Encoder, LoRA, or Phi-3.5. No sufficiency metric. No common-reference coverage that would use one fixed fact list for every condition.
