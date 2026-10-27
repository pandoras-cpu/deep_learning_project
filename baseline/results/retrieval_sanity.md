# ePQA dev retrieval sanity check

排序只在当前问题自己的候选池内进行。`label` 和 `answer` 不在索引里，指标是检索之后才接合的。

- Encoder: `BAAI/bge-small-en-v1.5` `5c38ec7c405e`
- Query prefix: `Represent this sentence for searching relevant passages: `
- Passages: `candidate` text, CLS pooling, L2 normalized, inner product
- Vectors: 9769；排除空 candidate 1 行
- Truncated at 512 tokens: 2

没有相关候选的问题不计入均值，也不记成 0。

| metric | mean | defined questions | undefined |
| --- | ---: | ---: | ---: |
| Recall@3, relevant = label 2 | 0.6373 | 805 | 172 |
| Success@3, at least one label 2 | 0.9093 | 805 | 172 |
| MRR, relevant = label 2 | 0.7858 | 805 | 172 |
| Recall@3, relevant = label 1 or 2 | 0.6108 | 921 | 56 |
| MRR, relevant = label 1 or 2 | 0.8650 | 921 | 56 |
| nDCG@3, gain = 2^label - 1 | 0.7310 | 921 | 56 |

## Why the pool is restricted

问题 `36` 的全索引 Top-3 不是 baseline。

- `dev:25` qid=36 score=0.7639
- `dev:3591` qid=3243 score=0.6821
- `dev:9544` qid=8588 score=0.6749

## Manual top-3

### qid 18

Top-3 labels after ranking: 1, 1, 0

- `dev:6` label=1 source=cqa score=0.8358: i am assuming it will fit. Question: my frigidare model # is frt18is6jbo, will part # 240337103 fit it?
- `dev:4` label=1 source=cqa score=0.8135: im not sure but i believe you can go on google and ask this same question and they will tell you what models that piece fits on Question: will this fit in a frigidaire model #frt18
- `dev:5` label=0 source=cqa score=0.7022: frigidaire oem part #240337103 yes this crisper drawer matches your model Question: will this fit kenmore refrigerator 25367889506?

### qid 31

Top-3 labels after ranking: 0, 0, 0

- `dev:15` label=0 source=cqa score=0.6642: the im3 kit also includes the water valve, water lines and ice tray.check the freezer but you should have a plug in the side to plug the icemaker into Question: my fridge call for 
- `dev:13` label=0 source=cqa score=0.6609: i am not really sure but if the old ice maker hangs on the left side of the freezer on two bolts that are roughly 5 inches apart it probably would work. Question: i have a ge fridg
- `dev:21` label=0 source=cqa score=0.6531: yes, this will work on your model Question: will this replace a model tbx22ma icemaker?

### qid 36

Top-3 labels after ranking: 2, 2, 2

- `dev:25` label=2 source=review score=0.7639: the visible handle part of the latch is black on this unit, but the white version doesn't seem to be available anymore.
- `dev:23` label=2 source=description score=0.6433: the door latch is only available in black (the white and biscuit colors have been discontinued).
- `dev:22` label=2 source=attribute score=0.6022: color:  { value:"white" }

### qid 62

Top-3 labels after ranking: 0, 2, 2

- `dev:44` label=0 source=cqa score=0.6929: the ice is around 30 degrees so need to put it in a freezer pretty fast or it will melt.live in south texas so... Question: does it make nugget ice ( sonic ice )?
- `dev:45` label=2 source=cqa score=0.658: dah, it does not make ice i said. Question: mine stopped making ice.  how can i fix it?
- `dev:43` label=2 source=review score=0.6339: this machine would not make ice, the lights for add water and full of ice kept going on and off, it is not worth the price.
