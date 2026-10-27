"""Align a judge quote to one evidence passage and keep the original span."""

from __future__ import annotations

_DROPPED = set("\"'`|“”‘’")


def align_quote(evidence_id: str, evidence_text: str, quote: str) -> dict:
    """Align within the cited evidence only. Failures stay visible."""
    record = {
        "evidence_id": evidence_id,
        "evidence_text": evidence_text,
        "quote": quote,
        "aligned_span": None,
        "alignment_status": "alignment_failure",
        "alignment_method": None,
        "reason": None,
    }
    if not evidence_text:
        record["reason"] = "missing_evidence_text"
        return record
    if not quote or not quote.strip():
        record["reason"] = "empty_quote"
        return record
    exact = _find_casefold(evidence_text, quote.strip())
    if exact is not None:
        start, end = exact
        record.update(
            {
                "aligned_span": evidence_text[start:end],
                "alignment_status": "aligned",
                "alignment_method": "exact",
                "start": start,
                "end": end,
            }
        )
        return record
    normalized = _find_normalized(evidence_text, quote.strip())
    if normalized is None:
        record["reason"] = "quote_not_aligned"
        return record
    start, end = normalized
    record.update(
        {
            "aligned_span": evidence_text[start:end],
            "alignment_status": "aligned",
            "alignment_method": "normalized_span",
            "start": start,
            "end": end,
        }
    )
    return record


def _find_casefold(text: str, quote: str) -> tuple[int, int] | None:
    haystack = text.casefold()
    needle = quote.casefold()
    if len(haystack) != len(text) or len(needle) != len(quote):
        return None
    start = haystack.find(needle)
    if start < 0:
        return None
    return start, start + len(quote)


def _find_normalized(text: str, quote: str) -> tuple[int, int] | None:
    text_norm, text_map = _normalize(text)
    quote_norm, _ = _normalize(quote)
    if len(quote_norm) < 8:
        return None
    start = text_norm.find(quote_norm)
    if start < 0:
        return None
    end = start + len(quote_norm) - 1
    return text_map[start], text_map[end] + 1


def _normalize(text: str) -> tuple[str, list[int]]:
    characters = []
    indexes = []
    for index, char in enumerate(text):
        if char in _DROPPED or char.isspace():
            continue
        folded = char.casefold()
        if len(folded) != 1:
            continue
        characters.append(folded)
        indexes.append(index)
    return "".join(characters), indexes
