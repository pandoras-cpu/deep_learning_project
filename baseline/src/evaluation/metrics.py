"""Aggregate UCR and evidence-conditional coverage. Does not invent missing scores."""

from __future__ import annotations


def apply_confidence(status: str, max_probability: float | None, threshold: float) -> str:
    if max_probability is None:
        return status
    if status in {"supported", "unsupported", "contradicted"} and max_probability < threshold:
        return "uncertain"
    return status


def aggregate_passage_statuses(statuses: list[str]) -> tuple[str, str]:
    if not statuses:
        return "evaluation_error", "no_evidence"
    supported = "supported" in statuses
    contradicted = "contradicted" in statuses
    uncertain = "uncertain" in statuses
    if supported and contradicted:
        return "uncertain", "passage_disagreement"
    if supported:
        return "supported", "passage_entailment"
    if contradicted and uncertain:
        return "uncertain", "contradiction_with_low_confidence"
    if contradicted:
        return "contradicted", "passage_contradiction"
    if uncertain:
        return "uncertain", "low_confidence"
    return "unsupported", "no_passage_entailment"


def score_ucr(claim_statuses: list[str]) -> dict:
    supported = sum(status == "supported" for status in claim_statuses)
    unsupported = sum(status in {"unsupported", "contradicted"} for status in claim_statuses)
    uncertain = sum(status == "uncertain" for status in claim_statuses)
    contradicted = sum(status == "contradicted" for status in claim_statuses)
    evaluation_error = sum(status == "evaluation_error" for status in claim_statuses)
    classified = supported + unsupported
    conservative_denominator = classified + uncertain
    if classified:
        undefined_reason = None
    elif evaluation_error and not uncertain:
        undefined_reason = "evaluation_error"
    elif uncertain and not evaluation_error:
        undefined_reason = "only_uncertain"
    elif uncertain and evaluation_error:
        undefined_reason = "no_confident_factual_claim"
    else:
        undefined_reason = "no_factual_claim"
    return {
        "supported": supported,
        "unsupported": unsupported,
        "contradicted": contradicted,
        "uncertain": uncertain,
        "evaluation_error": evaluation_error,
        "ucr": None if classified == 0 else unsupported / classified,
        "ucr_conservative": None
        if conservative_denominator == 0
        else (unsupported + uncertain) / conservative_denominator,
        "undefined": classified == 0,
        "undefined_reason": undefined_reason,
    }


def score_coverage(expressed_flags: list[bool | None]) -> dict:
    if not expressed_flags:
        return {
            "numerator": 0,
            "denominator": 0,
            "coverage": None,
            "undefined": True,
            "undefined_reason": "no_required_facts",
        }
    if any(flag is None for flag in expressed_flags):
        return {
            "numerator": None,
            "denominator": len(expressed_flags),
            "coverage": None,
            "undefined": True,
            "undefined_reason": "match_incomplete",
        }
    numerator = sum(bool(flag) for flag in expressed_flags)
    return {
        "numerator": numerator,
        "denominator": len(expressed_flags),
        "coverage": numerator / len(expressed_flags),
        "undefined": False,
        "undefined_reason": None,
    }


def aggregate(records: list[dict], condition: str) -> dict:
    chosen = [record for record in records if record["condition"] == condition]
    supported = unsupported = uncertain = evaluation_error = 0
    defined_ucr = []
    undefined_ucr = 0
    coverage_numer = coverage_denom = 0
    defined_coverage = []
    undefined_coverage = 0
    for record in chosen:
        ucr = record["ucr"]
        supported += ucr["supported"]
        unsupported += ucr["unsupported"]
        uncertain += ucr.get("uncertain", 0)
        evaluation_error += ucr.get("evaluation_error", 0)
        if ucr["undefined"]:
            undefined_ucr += 1
        else:
            defined_ucr.append(ucr["ucr"])
        coverage = record["coverage_evidence_conditional"]
        if coverage["undefined"]:
            undefined_coverage += 1
        else:
            coverage_numer += coverage["numerator"]
            coverage_denom += coverage["denominator"]
            defined_coverage.append(coverage["coverage"])
    classified = supported + unsupported
    return {
        "condition": condition,
        "answers": len(chosen),
        "ucr_micro": None if classified == 0 else round(unsupported / classified, 4),
        "ucr_macro": None if not defined_ucr else round(sum(defined_ucr) / len(defined_ucr), 4),
        "ucr_defined_answers": len(defined_ucr),
        "ucr_undefined_answers": undefined_ucr,
        "supported": supported,
        "unsupported": unsupported,
        "uncertain": uncertain,
        "evaluation_error": evaluation_error,
        "coverage_micro": None if coverage_denom == 0 else round(coverage_numer / coverage_denom, 4),
        "coverage_macro": None
        if not defined_coverage
        else round(sum(defined_coverage) / len(defined_coverage), 4),
        "coverage_numerator": coverage_numer,
        "coverage_denominator": coverage_denom,
        "coverage_defined_answers": len(defined_coverage),
        "coverage_undefined_answers": undefined_coverage,
    }
