"""CPU NLI support checker.

Premise is the supplied evidence. Hypothesis is one atomic claim.
Class indices come from model.config.id2label at runtime.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evaluation.metrics import aggregate_passage_statuses, apply_confidence
from evaluation.schema import ClaimJudgment, Evidence


class NliVerifier:
    def __init__(self, config_path: Path):
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        nli = config["nli"]
        self.model_id = nli["primary_model_id"]
        self.revision = nli["revision"]
        local_dir = nli.get("local_dir")
        source = str((config_path.resolve().parents[1] / local_dir)) if local_dir else self.model_id
        load_kwargs = {} if local_dir else {"revision": self.revision}
        self.device = "cpu"
        self.max_length = 512
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(source, **load_kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(source, **load_kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        self.low_confidence_below = float(nli.get("low_confidence_below", 0.5))
        self.id2label = {int(key): value.lower() for key, value in self.model.config.id2label.items()}
        required = {"entailment", "neutral", "contradiction"}
        if set(self.id2label.values()) != required:
            raise RuntimeError(f"unexpected id2label: {self.id2label}")

    def score_texts(self, premise: str, hypothesis: str) -> ClaimJudgment:
        untruncated = self.tokenizer(premise, hypothesis, truncation=False, add_special_tokens=True)
        token_count = len(untruncated["input_ids"])
        truncated = token_count > self.max_length
        encoded = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = self.model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        scores = {
            self.id2label[index]: round(float(probabilities[index]), 4)
            for index in range(probabilities.shape[0])
        }
        chosen = max(scores, key=scores.get)
        status = {
            "entailment": "supported",
            "contradiction": "contradicted",
            "neutral": "unsupported",
        }[chosen]
        judgment = ClaimJudgment(
            claim=hypothesis,
            status=status,
            nli_label=chosen,
            scores=scores,
            truncated=truncated,
            max_probability=scores[chosen],
        )
        judgment.token_count = token_count
        return judgment

    def score_claim(self, evidence: list[Evidence], claim: str) -> ClaimJudgment:
        passages = []
        for item in evidence:
            scored = self.score_texts(item.text, claim)
            passage_status = apply_confidence(
                scored.status, scored.max_probability, self.low_confidence_below
            )
            passages.append(
                {
                    "evidence_id": item.evidence_id,
                    "status": passage_status,
                    "nli_label": scored.nli_label,
                    "scores": scored.scores,
                    "max_probability": scored.max_probability,
                    "truncated": scored.truncated,
                    "token_count": scored.token_count,
                }
            )
        status, reason = aggregate_passage_statuses([item["status"] for item in passages])
        chosen = next((item for item in passages if item["status"] == status), passages[0] if passages else None)
        judgment = ClaimJudgment(
            claim=claim,
            status=status,
            nli_label=None if chosen is None else chosen["nli_label"],
            scores={} if chosen is None else chosen["scores"],
            evidence_ids=tuple(item.evidence_id for item in evidence),
            truncated=any(item["truncated"] for item in passages),
            max_probability=None if chosen is None else chosen["max_probability"],
        )
        judgment.passage_judgments = passages
        judgment.decision_reason = reason
        return judgment
