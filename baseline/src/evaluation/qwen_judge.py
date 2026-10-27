"""Local judge. Each instance loads only the model named by its config section."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.prompts import CLAIM_EXTRACTION_V2, FACT_MATCH_V2, QUESTION_ASPECTS_V1, SUPPORTED_FACTS_V2
from evaluation.schema import Claim, Evidence, Fact, FactMatch, SufficiencyResult
from evaluation.span_align import align_quote


def parse_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("JSON root is not an object")
    return value


class QwenJudge:
    def __init__(self, config_path: Path, section: str = "llm_judge"):
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        judge = config[section]
        self.model_id = judge["model_id"]
        self.revision = judge["revision"]
        self.max_new_tokens = int(judge["decoding"]["max_new_tokens"])
        if judge["decoding"]["do_sample"] is not False:
            raise ValueError("judge decoding must set do_sample: false")
        root = config_path.resolve().parents[1]
        cache_dir = root / judge["cache_dir"]
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available for the local judge.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision, cache_dir=cache_dir
        )
        # Same CPU cast used by the baseline: MPS bf16->fp16 cast segfaults on this M1.
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            cache_dir=cache_dir,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.to("mps")
        self.model.eval()
        self.model.generation_config.do_sample = False

    def close(self) -> None:
        del self.model
        del self.tokenizer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = {key: value.to("mps") for key, value in encoded.items()}
        input_tokens = int(encoded["input_ids"].shape[-1])
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        new_tokens = output[0, input_tokens:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text, input_tokens, int(new_tokens.shape[-1])

    def _json_call(self, system_prompt: str, user_prompt: str) -> tuple[dict, dict]:
        last_error = "no attempt"
        raw = ""
        tokens = {"input_tokens": None, "output_tokens": None}
        for attempt in range(2):
            prompt = user_prompt
            if attempt:
                prompt += "\n\nYour previous reply was not valid JSON. Reply with the JSON object only."
            raw, input_tokens, output_tokens = self.complete(system_prompt, prompt)
            tokens = {"input_tokens": input_tokens, "output_tokens": output_tokens, "raw": raw}
            try:
                return parse_json_object(raw), tokens
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
        raise ValueError(f"{last_error}; raw={raw[:500]}")

    def extract_aspects(self, question: str) -> tuple[list[str], dict]:
        payload, meta = self._json_call(QUESTION_ASPECTS_V1, f"Question:\n{question.strip()}")
        aspects = []
        for item in payload.get("aspects", []):
            text = item if isinstance(item, str) else item.get("aspect", "")
            text = str(text).strip()
            if text:
                aspects.append(text)
        return aspects, meta

    def extract_answer_claims(self, question: str, answer: str) -> tuple[list[Claim], dict]:
        payload, meta = self._json_call(
            CLAIM_EXTRACTION_V2,
            f"Question:\n{question.strip()}\n\nAnswer:\n{answer.strip()}",
        )
        claims = []
        for item in payload.get("claims", []):
            kind = item.get("kind")
            text = str(item.get("claim", "")).strip()
            if kind not in {"factual", "not_a_factual_claim"} or not text:
                continue
            claims.append(Claim(original_span=str(item.get("span", "")).strip(), text=text, status=kind))
        return claims, meta

    def extract_supported_facts(
        self, question: str, evidence: list[Evidence]
    ) -> tuple[list[Fact], dict]:
        blocks = [f"[{item.evidence_id}]\n{item.text.strip()}" for item in evidence]
        payload, meta = self._json_call(
            SUPPORTED_FACTS_V2,
            "Question:\n" + question.strip() + "\n\nEvidence:\n" + "\n\n".join(blocks),
        )
        by_id = {item.evidence_id: item.text for item in evidence}
        facts: list[Fact] = []
        for item in payload.get("facts", []):
            fact = str(item.get("fact", "")).strip()
            evidence_id = str(item.get("evidence_id", "")).strip()
            quote = str(item.get("quote", "")).strip()
            if evidence_id not in by_id:
                alignment = {
                    "evidence_id": evidence_id,
                    "evidence_text": None,
                    "quote": quote,
                    "aligned_span": None,
                    "alignment_status": "alignment_failure",
                    "alignment_method": None,
                    "reason": "unknown_evidence_id",
                }
            else:
                alignment = align_quote(evidence_id, by_id[evidence_id], quote)
            if not fact:
                alignment["alignment_status"] = "alignment_failure"
                alignment["reason"] = "missing_fact"
            facts.append(
                Fact(
                    text=fact,
                    evidence_ids=(evidence_id,) if evidence_id else (),
                    quote=quote,
                    aligned_span=alignment["aligned_span"],
                    alignment_status=alignment["alignment_status"],
                    alignment_method=alignment["alignment_method"],
                    alignment_reason=alignment["reason"],
                    evidence_text=alignment["evidence_text"],
                )
            )
        return facts, meta

    def match_answer_to_facts(
        self, question: str, answer: str, facts: list[Fact]
    ) -> tuple[list[FactMatch], dict]:
        fact_lines = "\n".join(f"{index}. {fact.text}" for index, fact in enumerate(facts, start=1))
        payload, meta = self._json_call(
            FACT_MATCH_V2,
            f"Question:\n{question.strip()}\n\nRequired facts:\n{fact_lines}\n\nAnswer:\n{answer.strip()}",
        )
        raw_matches = payload.get("matches", [])
        matches: list[FactMatch] = []
        for index, fact in enumerate(facts):
            item = raw_matches[index] if index < len(raw_matches) else {}
            expressed = item.get("expressed")
            span = item.get("answer_span")
            span_text = span.strip() if isinstance(span, str) else None
            if expressed is True:
                if not span_text or span_text.casefold() not in answer.casefold():
                    expressed = None
                    reason = "answer_span_not_in_answer"
                else:
                    reason = str(item.get("reason", "")).strip()
            elif expressed is False:
                reason = str(item.get("reason", "")).strip()
                span_text = None
            else:
                expressed = None
                reason = "missing_match"
            matches.append(
                FactMatch(fact=fact.text, expressed=expressed, answer_span=span_text, reason=reason)
            )
        return matches, meta

    def assess_sufficiency(self, question: str, evidence: list[Evidence]) -> SufficiencyResult:
        raise RuntimeError("sufficiency-v1 is not enabled in this phase")


