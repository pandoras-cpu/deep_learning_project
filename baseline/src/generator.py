"""Frozen evidence-only generator.

The model sees the question and evidence text only. Labels and reference
answers must not be passed in.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import yaml
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    text: str


@dataclass(frozen=True)
class GenerationResult:
    answer: str | None
    messages: list[dict[str, str]]
    prompt_version: str
    input_tokens: int | None
    output_tokens: int | None
    error: str | None


class EvidenceGenerator:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        with config_path.open(encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)
        self.prompt_version = self.config["prompt_version"]
        self.system_prompt = self.config["system_prompt"].strip()
        self.max_input_tokens = int(self.config["max_input_tokens"])
        self.max_new_tokens = int(self.config["decoding"]["max_new_tokens"])
        if self.config["decoding"]["do_sample"] is not False:
            raise ValueError("baseline decoding must set do_sample: false")

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available. Stop before choosing a fallback device.")
        if self.config["torch_dtype"] != "float16":
            raise ValueError("baseline config on this machine expects torch_dtype float16")

        cache_dir = config_path.resolve().parents[1] / self.config["cache_dir"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_id = self.config["model_id"]
        revision = self.config["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
        )
        # Cast bfloat16 weights to float16 on CPU. Casting inside the MPS
        # Metal kernel segfaults on this M1 (EXC_BAD_ACCESS in copy_cast_kernel_mps).
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.model.to("mps")
        self.model.eval()
        self.model.generation_config.do_sample = False

    def build_messages(self, question: str, evidence: list[Evidence]) -> list[dict[str, str]]:
        blocks = []
        for item in evidence:
            text = item.text.strip()
            if not text:
                raise ValueError(f"empty evidence text for {item.evidence_id}")
            blocks.append(f"[{item.evidence_id}]\n{text}")
        user = "Evidence:\n" + "\n\n".join(blocks) + f"\n\nQuestion:\n{question.strip()}"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

    def complete(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
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

    def generate(self, question: str, evidence: list[Evidence]) -> GenerationResult:
        if not evidence:
            return GenerationResult(
                answer=None,
                messages=[],
                prompt_version=self.prompt_version,
                input_tokens=None,
                output_tokens=None,
                error="empty_evidence",
            )
        messages = self.build_messages(question, evidence)
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        encoded = {key: value.to("mps") for key, value in encoded.items()}
        input_tokens = int(encoded["input_ids"].shape[-1])
        if input_tokens > self.max_input_tokens:
            return GenerationResult(
                answer=None,
                messages=messages,
                prompt_version=self.prompt_version,
                input_tokens=input_tokens,
                output_tokens=None,
                error="input_exceeds_max_input_tokens",
            )
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        new_tokens = output[0, input_tokens:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return GenerationResult(
            answer=answer,
            messages=messages,
            prompt_version=self.prompt_version,
            input_tokens=input_tokens,
            output_tokens=int(new_tokens.shape[-1]),
            error=None,
        )
