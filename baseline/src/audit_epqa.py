"""Audit the official ePQA CSVs and record a retrieval-corpus decision.

This script only reads the released files. It does not use labels or reference
answers to rank evidence or to build a generation prompt.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "external" / "contextual-product-qa" / "ePQA"
RESULTS_DIR = ROOT / "results"
SPLITS = ("train", "dev", "test")
OBSERVED_COLUMNS = [
    "qid",
    "question",
    "ASIN",
    "candidate",
    "source",
    "qa_pair_id",
    "title",
    "label",
    "answer",
]
README_COLUMNS = [
    "ASIN",
    "question",
    "qid",
    "candidate",
    "qa_pair_id",
    "source",
    "context",
    "label",
    "answer",
]
REPO_COMMIT = "cec976cc2f2218aa76e562ecc9c7d79f1f3271bc"
MAIN_SOURCES = ("attribute", "bullet", "cqa", "description", "review")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(split: str) -> pd.DataFrame:
    path = DATA_DIR / f"{split}.csv"
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    missing = [column for column in OBSERVED_COLUMNS if column not in frame.columns]
    extra = [column for column in frame.columns if column not in OBSERVED_COLUMNS]
    if missing or extra or list(frame.columns) != OBSERVED_COLUMNS:
        raise SystemExit(
            f"{split}.csv schema mismatch: missing={missing} extra={extra} "
            f"columns={list(frame.columns)}"
        )
    for column in OBSERVED_COLUMNS:
        frame[column] = frame[column].str.strip()
    if not frame["label"].isin({"0", "1", "2"}).all():
        unexpected = sorted(set(frame["label"]) - {"0", "1", "2"})
        raise SystemExit(f"{split}.csv has unexpected labels: {unexpected}")
    return frame


def question_label_patterns(frame: pd.DataFrame) -> dict[str, int]:
    grouped = frame.groupby("qid")["label"].agg(lambda values: frozenset(values))
    patterns = Counter()
    for labels in grouped:
        has_full = "2" in labels
        has_partial = "1" in labels
        has_irrelevant = "0" in labels
        if has_full and has_partial:
            patterns["has_label_2_and_1"] += 1
        elif has_full:
            patterns["has_label_2_only"] += 1
        elif has_partial:
            patterns["has_label_1_but_no_label_2"] += 1
        elif has_irrelevant:
            patterns["all_label_0"] += 1
        else:
            patterns["other"] += 1
    return dict(patterns)


def split_summary(split: str, frame: pd.DataFrame) -> dict:
    per_question = frame.groupby("qid").size()
    per_asin = frame.groupby("ASIN").size()
    asin_per_question = frame.groupby("qid")["ASIN"].nunique()
    question_per_qid = frame.groupby("qid")["question"].nunique()
    title_per_asin = frame.groupby("ASIN")["title"].nunique()
    duplicate_candidate_questions = int(
        (frame.groupby("qid")["candidate"].nunique() < frame.groupby("qid").size()).sum()
    )
    answer_present = frame["answer"].ne("")
    empty_candidates = frame["candidate"].eq("")
    sentinel = frame["qa_pair_id"].eq("-1")
    source_counts = frame["source"].value_counts().to_dict()
    return {
        "rows": int(len(frame)),
        "unique_qid": int(frame["qid"].nunique()),
        "unique_question_text": int(frame["question"].nunique()),
        "unique_ASIN": int(frame["ASIN"].nunique()),
        "unique_qa_pair_id": int(frame["qa_pair_id"].nunique()),
        "duplicate_qa_pair_id_rows": int(frame["qa_pair_id"].duplicated().sum()),
        "duplicate_full_rows": int(frame.duplicated().sum()),
        "questions_with_multiple_ASIN": int((asin_per_question > 1).sum()),
        "qids_with_multiple_question_texts": int((question_per_qid > 1).sum()),
        "questions_with_duplicate_candidate_text": duplicate_candidate_questions,
        "candidates_per_question": {
            "min": int(per_question.min()),
            "max": int(per_question.max()),
            "histogram": {str(key): int(value) for key, value in per_question.value_counts().sort_index().items()},
        },
        "candidates_per_ASIN": {
            "min": int(per_asin.min()),
            "median": float(per_asin.median()),
            "max": int(per_asin.max()),
        },
        "titles_per_ASIN_max": int(title_per_asin.max()),
        "asins_with_multiple_titles": int((title_per_asin > 1).sum()),
        "label_counts": frame["label"].value_counts().sort_index().astype(int).to_dict(),
        "answer_nonempty_by_label": (
            frame.loc[answer_present, "label"].value_counts().sort_index().astype(int).to_dict()
        ),
        "answer_empty_by_label": (
            frame.loc[~answer_present, "label"].value_counts().sort_index().astype(int).to_dict()
        ),
        "question_label_patterns": question_label_patterns(frame),
        "source_counts": {str(key): int(value) for key, value in source_counts.items()},
        "sources_outside_main_five": {
            str(key): int(value)
            for key, value in source_counts.items()
            if key not in MAIN_SOURCES
        },
        "qa_pair_id_minus_one_rows": int(sentinel.sum()),
        "qa_pair_id_minus_one_qids": int(frame.loc[sentinel, "qid"].nunique()),
        "empty_candidate_rows": int(empty_candidates.sum()),
        "empty_candidate_by_label": (
            frame.loc[empty_candidates, "label"].value_counts().sort_index().astype(int).to_dict()
        ),
        "empty_candidate_by_source": (
            frame.loc[empty_candidates, "source"].value_counts().astype(int).to_dict()
        ),
        "blank_field_counts": {
            column: int(frame[column].eq("").sum()) for column in OBSERVED_COLUMNS
        },
    }


def overlap(frames: dict[str, pd.DataFrame]) -> dict:
    fields = {
        "qid": "qid",
        "question_text": "question",
        "ASIN": "ASIN",
        "qa_pair_id": "qa_pair_id",
        "candidate_text": "candidate",
    }
    report = {}
    for name, column in fields.items():
        sets = {split: set(frame[column]) for split, frame in frames.items()}
        report[name] = {
            "train_dev": len(sets["train"] & sets["dev"]),
            "train_test": len(sets["train"] & sets["test"]),
            "dev_test": len(sets["dev"] & sets["test"]),
            "all_three": len(sets["train"] & sets["dev"] & sets["test"]),
        }
    shared_questions = {}
    for left, right, key in (
        ("train", "dev", "train_dev"),
        ("train", "test", "train_test"),
        ("dev", "test", "dev_test"),
    ):
        shared = sorted(set(frames[left]["question"]) & set(frames[right]["question"]))
        examples = []
        for text in shared:
            left_rows = frames[left].loc[frames[left]["question"].eq(text), ["qid", "ASIN"]].drop_duplicates()
            right_rows = frames[right].loc[frames[right]["question"].eq(text), ["qid", "ASIN"]].drop_duplicates()
            examples.append(
                {
                    "question": text,
                    left: left_rows.to_dict(orient="records"),
                    right: right_rows.to_dict(orient="records"),
                    "same_qid": bool(set(left_rows["qid"]) & set(right_rows["qid"])),
                    "same_ASIN": bool(set(left_rows["ASIN"]) & set(right_rows["ASIN"])),
                }
            )
        shared_questions[key] = examples
    report["shared_question_texts"] = shared_questions
    return report


def sample_rows(frame: pd.DataFrame, limit: int = 2) -> list[dict]:
    samples = []
    for record in frame.head(limit).to_dict(orient="records"):
        item = {}
        for key, value in record.items():
            text = value.replace("\n", " ")
            item[key] = text if len(text) <= 180 else f"{text[:180]}..."
            item[f"{key}_chars"] = len(value)
        samples.append(item)
    return samples


def build_markdown(audit: dict) -> str:
    decision = audit["corpus_decision"]
    lines = [
        "# ePQA data audit",
        "",
        "官方仓库内的 ePQA CSV 审计。数字只来自本次读入的文件。",
        "",
        f"- Repo commit: `{audit['repo_commit']}`（shallow clone）",
        f"- Python: `{audit['python']}`",
        f"- pandas: `{audit['pandas']}`",
        "- 分隔符：逗号。`read_csv(dtype=str, keep_default_na=False)`。",
        "",
        "## Schema",
        "",
        "观察到的列：`" + "`, `".join(OBSERVED_COLUMNS) + "`。",
        "",
        "README 写有 `context`（候选周围句子）。三个 CSV 都没有这一列，同一位置是产品标题 `title`。",
        "",
        "README 对 `label` 的定义：`2` fully answering，`1` helpful but not fully answering，`0` irrelevant。",
        "观察到 `answer` 非空的行正好是 `label` 为 1 或 2 的行；`label=0` 的 `answer` 为空。",
        "因此 `answer` 是否存在不能用来判断候选是不是 fully answering。",
        "",
        "## Splits",
        "",
        "| split | rows | qid | ASIN | qa_pair_id | label 0 | label 1 | label 2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        item = audit["splits"][split]
        labels = item["label_counts"]
        lines.append(
            f"| {split} | {item['rows']} | {item['unique_qid']} | {item['unique_ASIN']} | "
            f"{item['unique_qa_pair_id']} | {labels.get('0', 0)} | {labels.get('1', 0)} | {labels.get('2', 0)} |"
        )
    lines.extend(
        [
            "",
            "每个 `qid` 只对应一个 `ASIN`、一段 question text。train 和 dev 的每个问题都是 10 条候选。",
            "test 有 1,852 个问题是 10 条，145 个是 11 条，3 个是 9 条。",
            "train 和 dev 的 `qa_pair_id` 在 split 内唯一，跨 split 也没有相同的 `qa_pair_id`。",
            "test 有 150 行的 `qa_pair_id` 都是 `-1`，分布在 30 个问题里，每题 5 条；这些行的 `label` 都是 0，`answer` 和 `title` 为空。",
            "因此证据 ID 不用 `qa_pair_id`，而用官方 CSV 的 `split:file_line`（含表头的文件行号）。",
            "",
            "问题级标签形态：",
            "",
            "| split | 同时有 2 和 1 | 只有 2 | 有 1 但没有 2 | 全部是 0 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in SPLITS:
        patterns = audit["splits"][split]["question_label_patterns"]
        lines.append(
            f"| {split} | {patterns.get('has_label_2_and_1', 0)} | "
            f"{patterns.get('has_label_2_only', 0)} | "
            f"{patterns.get('has_label_1_but_no_label_2', 0)} | "
            f"{patterns.get('all_label_0', 0)} |"
        )
    overlap_report = audit["overlap"]
    lines.extend(
        [
            "",
            "## Cross-split overlap",
            "",
            "| field | train∩dev | train∩test | dev∩test | 三者共有 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for field, counts in overlap_report.items():
        if field == "shared_question_texts":
            continue
        lines.append(
            f"| {field} | {counts['train_dev']} | {counts['train_test']} | "
            f"{counts['dev_test']} | {counts['all_three']} |"
        )
    lines.extend(
        [
            "",
            "`qid` 没有跨 split 重叠。question 文本有少量重叠，但对应的 `qid` 和 `ASIN` 都不同，重叠的是泛问句，不是同一道题。",
            "`ASIN` 和部分 `candidate` 文本有重叠：同一产品会带着不同问题出现在不同 split。",
            "标准 baseline 只在当前问题自己的候选池里检索，不把其他问题或其他 split 的候选并进索引。",
            "",
            "## Corpus decision",
            "",
            decision["summary_zh"],
            "",
            f"- Query ID: `{decision['query_id']}`",
            f"- Evidence ID: `{decision['evidence_id']}`",
            f"- Product ID: `{decision['product_id']}`",
            f"- Retrieval text: `{decision['retrieval_text']}`",
            f"- Initial Top-K: {decision['top_k']}",
            "",
            "不把 `label` 和 `answer` 放进检索或标准 RAG 的生成提示。",
            "检索指标以后要分开报告：相关集合只含 `label=2`，以及相关集合含 `label` 为 1 或 2。",
            "Gold Evidence QA 使用 `label=2` 的候选，并且单独标注，不进入标准 RAG。",
            "",
            "## Caveats",
            "",
            "- 发布文件没有周围句子，不能做 candidate 与 candidate+context 的对比。`title` 是产品标题。",
            "- `label=1` 只表示该候选被标成 helpful but not fully answering，不能单独推出整个问题部分可答。",
            "- test 里有少量主来源以外的 `source` 字符串，见 JSON。",
            "- 空 `candidate`：train 137 行、dev 1 行、test 14 行，按 `label` 和 `source` 的计数见 JSON。",
            "- 同一问题里有重复 `candidate` 文本：train 64 题、dev 5 题、test 11 题。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    frames = {split: load_split(split) for split in SPLITS}
    files = {}
    for split in SPLITS:
        path = DATA_DIR / f"{split}.csv"
        files[split] = {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "columns": list(frames[split].columns),
        }
    audit = {
        "dataset": "ePQA",
        "source_repo": "https://github.com/amazon-science/contextual-product-qa",
        "repo_commit": REPO_COMMIT,
        "clone": "shallow",
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "delimiter": "comma",
        "readme_columns": README_COLUMNS,
        "observed_columns": OBSERVED_COLUMNS,
        "context_column_present": False,
        "label_definition_from_readme": {
            "2": "fully answering",
            "1": "helpful but not fully answering",
            "0": "irrelevant",
        },
        "files": files,
        "splits": {split: split_summary(split, frames[split]) for split in SPLITS},
        "overlap": overlap(frames),
        "dev_samples_truncated": sample_rows(frames["dev"]),
        "corpus_decision": {
            "task": "question-specific candidate ranking",
            "open_corpus_product_retrieval": False,
            "reason": (
                "Each qid has one ASIN and a preassembled pool of about 10 candidates. "
                "The released files do not contain a product-page corpus independent of these pools."
            ),
            "summary_zh": (
                "Baseline 是问题级候选排序：给定一个问题和它自带的约 10 条候选，检索其中的证据再生成答案。"
                "这不是产品页开放语料检索。"
            ),
            "query_id": "qid",
            "evidence_id": "split:file_line",
            "evidence_id_note": (
                "file_line is the 1-based line number in the official CSV, including the header. "
                "qa_pair_id is unique in train and dev and does not overlap across splits, "
                "but 150 test rows share qa_pair_id -1, so qa_pair_id is not an evidence ID."
            ),
            "product_id": "ASIN",
            "retrieval_text": "candidate",
            "top_k": 3,
            "top_k_reason": "Every question pool has 9 to 11 candidates, so Top-K=3 selects a proper subset.",
            "excluded_from_model_input": ["label", "answer"],
            "context_comparison_supported": False,
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "data_audit.json"
    md_path = RESULTS_DIR / "data_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(build_markdown(audit), encoding="utf-8")
    print(f"wrote {json_path.relative_to(ROOT)}")
    print(f"wrote {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
