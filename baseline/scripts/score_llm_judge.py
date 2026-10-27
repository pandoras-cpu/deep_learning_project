"""Score saved LLM-judge labels. Does not call a model and does not edit the human template."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "llm_judge_annotations.json"
CLAIMS = ROOT / "results" / "llm_judge_claims.csv"
FACTS = ROOT / "results" / "llm_judge_facts.csv"
METRICS = ROOT / "results" / "llm_judge_metrics.json"


def main():
    data = json.loads(SOURCE.read_text())
    items = data["items"]
    claim_rows = []
    fact_rows = []
    metrics = {}
    for condition in ("rag", "gold"):
        supported = unsupported = uncertain = nonclaim = 0
        defined = undefined = 0
        abstain_yes = abstain_no = 0
        expressed = required = 0
        questions_undefined = 0
        for item in items:
            counts = {"supported": 0, "unsupported": 0, "uncertain": 0, "not_a_factual_claim": 0}
            for text, label in item[condition]["claims"]:
                counts[label] += 1
                claim_rows.append(
                    {
                        "example_id": item["id"],
                        "condition": "standard_rag" if condition == "rag" else "gold_evidence_qa",
                        "claim": text,
                        "label": label,
                        "no_generation": item[condition].get("no_generation", False),
                    }
                )
            factual = counts["supported"] + counts["unsupported"]
            if factual:
                defined += 1
            else:
                undefined += 1
            supported += counts["supported"]
            unsupported += counts["unsupported"]
            uncertain += counts["uncertain"]
            nonclaim += counts["not_a_factual_claim"]
            if item[condition]["abstention"] == "yes":
                abstain_yes += 1
            elif item[condition]["abstention"] == "no":
                abstain_no += 1
            if condition == "rag":
                for index, fact in enumerate(item["facts"]):
                    fact_rows.append(
                        {
                            "example_id": item["id"],
                            "fact": fact,
                            "standard_rag_expressed": "yes" if item["rag"]["expressed"][index] else "no",
                            "gold_expressed": "yes" if item["gold"]["expressed"][index] else "no",
                        }
                    )
            if item["facts"]:
                required += len(item["facts"])
                expressed += sum(item[condition]["expressed"])
            else:
                questions_undefined += 1
        metrics[condition] = {
            "ucr": round(unsupported / (supported + unsupported), 4),
            "supported": supported,
            "unsupported": unsupported,
            "uncertain_excluded": uncertain,
            "nonclaim_excluded": nonclaim,
            "answers_with_factual_claims": defined,
            "answers_without_factual_claims": undefined,
            "abstention_appropriate": abstain_yes,
            "abstention_inappropriate": abstain_no,
            "answerable_coverage": round(expressed / required, 4),
            "facts_expressed": expressed,
            "facts_required": required,
            "questions_with_answerable_facts": len(items) - questions_undefined,
            "questions_without_answerable_facts": questions_undefined,
        }
    with CLAIMS.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            ["example_id", "condition", "claim", "label", "no_generation"],
        )
        writer.writeheader()
        writer.writerows(claim_rows)
    with FACTS.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            ["example_id", "fact", "standard_rag_expressed", "gold_expressed"],
        )
        writer.writeheader()
        writer.writerows(fact_rows)
    METRICS.write_text(json.dumps({"judge": data["judge"], "metrics": metrics}, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
