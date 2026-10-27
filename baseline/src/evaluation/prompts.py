"""Frozen judge prompts. These prompts do not see labels or reference answers."""

QUESTION_ASPECTS_V1 = """You list the parts of a question that an answer would need to address. Reply with one JSON object and no markdown.
Schema: {"aspects": [string]}
Rules:
- List requested aspects only. Do not guess the factual answer.
- Do not use product knowledge or evidence. You are not shown evidence.
- Keep each aspect short.
- If the question asks one thing, return one aspect.
Format example, not a product question:
Question: Does it support 4K, and what is its weight?
{"aspects": ["whether it supports 4K", "its weight"]}"""



CLAIM_EXTRACTION_V2 = """You split one answer into atomic claims. Reply with one JSON object and no markdown.
Schema: {"claims": [{"span": string, "claim": string, "kind": "factual" | "not_a_factual_claim"}]}
Rules:
- A factual claim states a product property, compatibility, quantity, instruction, or other checkable fact.
- Split an answer that states two facts into two claims.
- Keep negation. If the answer says "does not" or "will not", the claim must stay negative.
- Keep conditionals. "If X, then Y" must not become "Y" or "X is possible" unless the answer explicitly says that.
- Keep product-specific qualifiers in the claim: model numbers, sizes, "medium", "after receipt", and similar limits.
- Do not strengthen a qualified statement into an unconditional one.
- Politeness, "not stated", "the evidence does not say", and statements that only say a requested fact is missing are not_a_factual_claim.
- Do not add anything the answer does not say.
- Do not use outside knowledge.
Format example, not the product you are judging:
Answer: Yes, it supports 4K. The evidence does not state the weight.
{"claims": [
  {"span": "it supports 4K", "claim": "The product supports 4K.", "kind": "factual"},
  {"span": "The evidence does not state the weight.", "claim": "The weight is not stated.", "kind": "not_a_factual_claim"}
]}"""

SUPPORTED_FACTS_V2 = """You list atomic facts that the question asks for and that the evidence explicitly supports. Reply with one JSON object and no markdown.
Schema: {"facts": [{"fact": string, "evidence_id": string, "quote": string}]}
Rules:
- Include a fact only when the question requires it and the quote states that same fact.
- The fact must not add a conclusion, product name, or qualifier that the quote does not state.
- Keep negation and conditionals. A quote beginning with "if" does not become an unconditional fact.
- Copy quote from the cited evidence and copy evidence_id exactly.
- Do not include unrelated product details.
- Deduplicate facts.
- If the evidence supports none of the requested facts, return {"facts": []}.
- You are not shown labels or a reference answer. Do not invent them.
Format example, not the product you are judging:
Question: Does it support 4K?
Evidence:
[ex:1]
The camera supports 4K recording.
{"facts": [{"fact": "The camera supports 4K recording.", "evidence_id": "ex:1", "quote": "The camera supports 4K recording."}]}"""

FACT_MATCH_V2 = """You decide whether an answer expresses each required fact. Reply with one JSON object and no markdown.
Schema: {"matches": [{"fact": string, "expressed": true or false, "answer_span": string or null, "reason": string}]}
Rules:
- expressed is true only when the answer states that same fact, including its negation and qualifiers.
- A fact with "if", "not", or a model/size limit is not expressed by a sentence that drops that condition.
- When expressed is true, answer_span must be copied verbatim from the answer. Otherwise answer_span is null.
- A denial or an abstention does not express the fact.
- Return one match for every fact, in the same order.
- Do not use outside knowledge.
Format example, not the product you are judging:
Fact: The camera supports 4K recording.
Answer: Yes, it records in 4K.
{"matches": [{"fact": "The camera supports 4K recording.", "expressed": true, "answer_span": "it records in 4K", "reason": "The answer states 4K recording."}]}"""
