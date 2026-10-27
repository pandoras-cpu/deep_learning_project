"""Independent judge interface.

No provider is configured. Coverage fact extraction must not call the
generator and must not reuse the preliminary 14/38 or 24/38 counts.
"""

from __future__ import annotations

from typing import Protocol

from evaluation.schema import Claim, Evidence, Fact, FactMatch, SufficiencyResult


class JudgeClient(Protocol):
    def extract_answer_claims(self, question: str, answer: str) -> list[Claim]:
        """Split a generated answer into atomic claims."""

    def extract_supported_facts(self, question: str, evidence: list[Evidence]) -> list[Fact]:
        """List question-relevant facts supported by this evidence set."""

    def match_answer_to_facts(
        self, question: str, answer: str, facts: list[Fact]
    ) -> list[FactMatch]:
        """Decide which required facts the answer expresses."""

    def assess_sufficiency(self, question: str, evidence: list[Evidence]) -> SufficiencyResult:
        """Judge whether this evidence set can answer the question."""


class UnconfiguredJudge:
    """Placeholder until the user names an independent judge provider."""

    def __init__(self) -> None:
        self.provider = None
        self.model_id = None

    def _refuse(self, method: str) -> None:
        raise RuntimeError(
            f"{method} is not available: no independent LLM judge is configured. "
            "Do not call the generator and do not reuse the preliminary coverage counts."
        )

    def extract_answer_claims(self, question: str, answer: str) -> list[Claim]:
        self._refuse("extract_answer_claims")

    def extract_supported_facts(self, question: str, evidence: list[Evidence]) -> list[Fact]:
        self._refuse("extract_supported_facts")

    def match_answer_to_facts(
        self, question: str, answer: str, facts: list[Fact]
    ) -> list[FactMatch]:
        self._refuse("match_answer_to_facts")

    def assess_sufficiency(self, question: str, evidence: list[Evidence]) -> SufficiencyResult:
        self._refuse("assess_sufficiency")
