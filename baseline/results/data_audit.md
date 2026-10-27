# ePQA data audit

官方仓库内的 ePQA CSV 审计。数字只来自本次读入的文件。

- Repo commit: `cec976cc2f2218aa76e562ecc9c7d79f1f3271bc`（shallow clone）
- Python: `3.13.5`
- pandas: `3.1.0`
- 分隔符：逗号。`read_csv(dtype=str, keep_default_na=False)`。

## Schema

观察到的列：`qid`, `question`, `ASIN`, `candidate`, `source`, `qa_pair_id`, `title`, `label`, `answer`。

README 写有 `context`（候选周围句子）。三个 CSV 都没有这一列，同一位置是产品标题 `title`。

README 对 `label` 的定义：`2` fully answering，`1` helpful but not fully answering，`0` irrelevant。
观察到 `answer` 非空的行正好是 `label` 为 1 或 2 的行；`label=0` 的 `answer` 为空。
因此 `answer` 是否存在不能用来判断候选是不是 fully answering。

## Splits

| split | rows | qid | ASIN | qa_pair_id | label 0 | label 1 | label 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 121750 | 12175 | 5393 | 121750 | 80704 | 18431 | 22615 |
| dev | 9770 | 977 | 888 | 9770 | 6415 | 1042 | 2313 |
| test | 20142 | 2000 | 1632 | 19993 | 13505 | 2246 | 4391 |

每个 `qid` 只对应一个 `ASIN`、一段 question text。train 和 dev 的每个问题都是 10 条候选。
test 有 1,852 个问题是 10 条，145 个是 11 条，3 个是 9 条。
train 和 dev 的 `qa_pair_id` 在 split 内唯一，跨 split 也没有相同的 `qa_pair_id`。
test 有 150 行的 `qa_pair_id` 都是 `-1`，分布在 30 个问题里，每题 5 条；这些行的 `label` 都是 0，`answer` 和 `title` 为空。
因此证据 ID 不用 `qa_pair_id`，而用官方 CSV 的 `split:file_line`（含表头的文件行号）。

问题级标签形态：

| split | 同时有 2 和 1 | 只有 2 | 有 1 但没有 2 | 全部是 0 |
| --- | ---: | ---: | ---: | ---: |
| train | 5094 | 3313 | 2954 | 814 |
| dev | 423 | 382 | 116 | 56 |
| test | 905 | 776 | 222 | 97 |

## Cross-split overlap

| field | train∩dev | train∩test | dev∩test | 三者共有 |
| --- | ---: | ---: | ---: | ---: |
| qid | 0 | 0 | 0 | 0 |
| question_text | 5 | 7 | 0 | 0 |
| ASIN | 743 | 1368 | 324 | 219 |
| qa_pair_id | 0 | 0 | 0 | 0 |
| candidate_text | 1134 | 1830 | 503 | 335 |

`qid` 没有跨 split 重叠。question 文本有少量重叠，但对应的 `qid` 和 `ASIN` 都不同，重叠的是泛问句，不是同一道题。
`ASIN` 和部分 `candidate` 文本有重叠：同一产品会带着不同问题出现在不同 split。
标准 baseline 只在当前问题自己的候选池里检索，不把其他问题或其他 split 的候选并进索引。

## Corpus decision

Baseline 是问题级候选排序：给定一个问题和它自带的约 10 条候选，检索其中的证据再生成答案。这不是产品页开放语料检索。

- Query ID: `qid`
- Evidence ID: `split:file_line`
- Product ID: `ASIN`
- Retrieval text: `candidate`
- Initial Top-K: 3

不把 `label` 和 `answer` 放进检索或标准 RAG 的生成提示。
检索指标以后要分开报告：相关集合只含 `label=2`，以及相关集合含 `label` 为 1 或 2。
Gold Evidence QA 使用 `label=2` 的候选，并且单独标注，不进入标准 RAG。

## Caveats

- 发布文件没有周围句子，不能做 candidate 与 candidate+context 的对比。`title` 是产品标题。
- `label=1` 只表示该候选被标成 helpful but not fully answering，不能单独推出整个问题部分可答。
- test 里有少量主来源以外的 `source` 字符串，见 JSON。
- 空 `candidate`：train 137 行、dev 1 行、test 14 行，按 `label` 和 `source` 的计数见 JSON。
- 同一问题里有重复 `candidate` 文本：train 64 题、dev 5 题、test 11 题。
