"""Evaluation records. Labels and reference answers are not fields here."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    text: str


@dataclass(frozen=True)
class Claim:
    original_span: str
    text: str
    status: str | None = None


@dataclass(frozen=True)
class Fact:
    text: str
    evidence_ids: tuple[str, ...]
    quote: str
    aligned_span: str | None = None
    alignment_status: str = "alignment_failure"
    alignment_method: str | None = None
    alignment_reason: str | None = None
    evidence_text: str | None = None


@dataclass(frozen=True)
class FactMatch:
    fact: str
    expressed: bool | None
    answer_span: str | None
    reason: str


@dataclass(frozen=True)
class SufficiencyResult:
    status: str
    evidence_ids: tuple[str, ...]
    reason: str


@dataclass
class ClaimJudgment:
    claim: str
    status: str
    nli_label: str | None = None
    scores: dict[str, float] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()
    truncated: bool = False
    max_probability: float | None = None
