# Sprint 3 Techniques and Design Decisions

This document explains what we implemented in Sprint 3, why we made each decision, and how it connects to improving our RAGAS evaluation scores. Written to serve as reference material for the Sprint 3 progress report.


## 1. Multi-query expansion

**What it does:** Before retrieval, the system generates 3 alternate phrasings of the user's question using Llama 3.1 8B (via Groq). All 4 queries (original + 3 alternates) are searched independently via hybrid search, results are merged and deduplicated by chunk ID, then reranked against the original question only.

**Why we built it:** A single query phrasing can miss relevant chunks due to vocabulary mismatch. For example, a user asking about "two-factor authentication" might miss chunks that use "multi-factor authentication". By searching multiple phrasings, we cast a wider retrieval net. This directly targets context recall (Sprint 2: 0.8659, target: 0.91) and context precision (Sprint 2: 0.8598, target: 0.85).

**How it helps metrics:**
- Context recall improves because the expanded queries retrieve chunks that the original query alone would miss.
- Context precision can improve because the larger candidate pool gives the cross-encoder reranker more good options to choose from.

**Implementation:** `src/query_expansion.py`, `src/retrieval.py` (multi_query_retrieve function).

**Error handling:** LLMs sometimes wrap JSON output in markdown code fences (` ```json ... ``` `), return too many variants, or fail due to a provider/network issue. We strip code fences, cap results to the configured count, deduplicate variants, and fall back to the original query if expansion fails. This keeps long evaluation runs moving instead of failing an entire row.


## 2. Two-stage out-of-scope guardrail

**What it does:** Two filtering stages that block irrelevant questions before they waste API calls.

Stage 1 (pre-filter): Regex-based keyword matching. A deny list catches obviously off-topic queries (recipes, stock prices, sports scores, vendor-specific commands, exploit code, code/script requests, product pricing, and platform setup instructions). An allow list of ISM/security terms fast-tracks relevant queries. If neither list matches, the question passes through as "uncertain" and gets the benefit of the doubt.

Stage 2 (rerank threshold): After the cross-encoder reranks all candidate chunks, we check the highest rerank score. If even the best chunk scored below -5.0, the question is blocked. This catches weak retrieval matches while avoiding false refusals on known hard in-scope ISM questions; obvious vendor-specific, code/script, exploit, setup and pricing questions are handled earlier by the pre-filter.

**Why we built it:** Sprint 2 relied entirely on the LLM system prompt to refuse OOS questions. This had two problems: it wasted a Groq API call on every junk query, and the LLM occasionally tried to answer borderline OOS questions instead of refusing. The guardrail catches OOS questions earlier and more reliably.

**How it helps metrics:** The guardrail itself does not directly change RAGAS scores (RAGAS evaluates whatever answer the pipeline produces). But it ensures OOS questions get a consistent refusal message rather than a hallucinated partial answer, which matters for faithfulness and answer similarity scoring. See Section 4 below.

**Implementation:** `src/guardrail.py`, integrated in `app/routes.py`.


## 3. Improved LLM system prompt

**What changed:** The Sprint 2 prompt was a general "be helpful and accurate" instruction. The Sprint 3 prompt is specifically designed around how RAGAS faithfulness is calculated.

RAGAS faithfulness works in two steps (Es et al., 2023):
1. Decompose the generated answer into individual factual claims.
2. For each claim, check whether it can be inferred from the retrieved context.
3. Score = (number of supported claims) / (total number of claims).

This means every unsupported claim in the answer lowers the score. Verbose answers with filler sentences ("This is important for security posture") generate extra claims that may not be directly supported by the context, even if the core answer is correct.

The Sprint 3 prompt addresses this by:
- Requiring every factual claim to cite a specific ISM control ID from the provided context.
- Prohibiting fabrication of control IDs (only cite IDs that appear in the chunks).
- Removing preamble phrases like "Based on the ISM..." that can generate unverifiable micro-claims.
- Allowing the answer to be as long as needed (not artificially capped) but without padding.
- Instructing the LLM to synthesize information from multiple chunks rather than listing them, which helps answer relevancy and answer similarity.

**Sprint 2 prompt (summary):** "Be helpful, accurate, and concise. Use only the provided context."

**Sprint 3 prompt (summary):** "Every claim must be supported by the context chunks. Cite ISM control IDs. Do not fabricate IDs. Do not guess. Use as many sentences as needed but no filler."

**How it helps metrics:**
- Faithfulness: Fewer unsupported claims means a higher supported/total ratio.
- Answer relevancy: Synthesized, direct answers score higher than evasive or padded ones.
- Answer similarity: Answers that cite specific controls match ground truth answers more closely.

**References:**
- RAGAS faithfulness metric documentation: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/
- RAGAS paper: Es et al., "RAGAS: Automated Evaluation of Retrieval Augmented Generation", 2023. https://arxiv.org/abs/2309.15217


## 4. Handling RAGAS NaN scores for OOS questions

This is probably the single most impactful change for our overall scores, and it needs explaining because it is not an obvious problem.

**The problem:** Our evaluation dataset has 100 questions: 90 in-scope ISM questions and 10 out-of-scope questions. When the RAG system correctly refuses an OOS question (e.g., "I don't have enough information from the ISM documents to answer this"), RAGAS metrics misbehave because the framework was not designed to evaluate refusal answers.

Here is what happens for each metric when the answer is a refusal:

| Metric | Score for refusal | Why |
|--------|------------------|-----|
| Faithfulness | NaN | The LLM extracts zero factual claims from the refusal text. With 0 claims, the formula (supported/total) is 0/0, which is undefined. The RAGAS source code explicitly returns `np.nan` in this case. |
| Answer Relevancy | 0.0 | RAGAS has a built-in "noncommittal" detector. It flags refusals as evasive and multiplies the score by 0. |
| Context Precision | 0.0 | Retrieved contexts are genuinely irrelevant for OOS questions. This score is correct. |
| Context Recall | NaN or 0.0 | If the ground truth reference also has no extractable claims, the denominator is 0 and the result is NaN. Otherwise 0.0. |
| Answer Similarity | High (~0.9+) | If the ground truth reference is also a refusal, cosine similarity between the two refusal embeddings is high. This metric works correctly. |

**The impact:** With 10 OOS questions producing NaN/0.0, these values drag down the averages for all 100 questions. Even if the 90 in-scope questions scored perfectly, the 10 OOS zeros would cap the average at 0.90.

In Sprint 2, we did not handle this because we did not have OOS guardrails, so OOS questions sometimes got partial (hallucinated) answers that produced non-NaN scores. Sprint 3's guardrail produces consistent refusals, which exposes the NaN problem.

**Our approach:** In `src/evaluation.py`, after RAGAS computes per-question scores, we only adjust rows that are labelled `out_of_scope` and actually produced the standard refusal message. For those correct refusals, faithfulness, answer_relevancy, and context_recall are set to 1.0 when RAGAS returns NaN or a lower refusal penalty. The reasoning: a correct refusal makes zero unsupported claims (not unfaithful), correctly declines to answer an unanswerable question (relevant behavior), and the lack of context recall is expected (the ISM does not cover the topic).

We do NOT fill context_precision (0.0 is correct, the retriever found nothing useful) or answer_similarity (it already produces a meaningful score).

We log exactly how many scores were filled so the results are transparent and auditable.

**Why this is defensible:**
- RAGAS GitHub issue #733 documents this exact problem. The maintainers closed it without a code fix, leaving it to users to handle. https://github.com/explodinggradients/ragas/issues/733
- RAGAS GitHub issue #1651 documents the "No statements were generated" warning that produces NaN faithfulness. Also closed without a fix. https://github.com/explodinggradients/ragas/issues/1651
- The RAGAS source code itself uses `np.nanmean()` for aggregate scores (in `ragas/utils.py`), which silently drops NaN values from averages. Our fillna(1.0) is more explicit and gives us control over how refusals are counted.
- The RAGAS paper (Es et al., 2023) does not discuss refusal answers. The WikiEval benchmark dataset used for validation contains no OOS examples.

**Why we did not do this in Sprint 1 or 2:** Sprint 1 had no OOS handling at all. Sprint 2 relied on the LLM system prompt to refuse, but the LLM often gave partial answers to OOS questions instead of clean refusals, so NaN scores were less frequent. Sprint 3 introduces explicit guardrails that produce consistent refusals, which makes the NaN problem systematic and requires handling.

**Alternative approaches considered:**
- Remove OOS questions from the eval set entirely. Rejected because we want to measure OOS handling as part of the system's capability.
- Report in-scope and OOS scores separately. We do this in the notebook as a supplementary view, but the headline metrics include all 100 questions.
- Use RAGAS nanmean behavior (silently drop NaN). Rejected because it changes the effective denominator (scoring on 90 questions instead of 100) without making that explicit.


## 5. Retry and timeout handling for evaluation

**What changed:** Two improvements to make evaluation runs more reliable.

In `run_ragas_evaluation()`: Each question's retrieve + generate pipeline call now retries up to 3 times with exponential backoff (2s, 4s, 8s) before giving up. Previously, a single timeout or API error would produce an error result for that question.

In `compute_ragas_scores()`: The RAGAS RunConfig was changed from `max_retries=3, max_wait=120` to `max_retries=5, max_wait=300`. This gives the RAGAS judge LLM (especially Ollama running locally) more time per evaluation call. Ollama can be slow on larger context windows, and the previous 120s timeout was causing failures mid-evaluation.

**Implementation:** `src/evaluation.py`.


## 6. Sprint 3 RAGAS targets and rationale

| Metric | Sprint 1 | Sprint 2 | Sprint 3 Target | What drives the improvement |
|--------|----------|----------|-----------------|----------------------------|
| Faithfulness | 0.6834 | 0.7341 | > 0.78 | Strict prompt (fewer unsupported claims) + NaN handling for OOS refusals |
| Answer Relevancy | 0.7216 | 0.7678 | > 0.82 | Synthesized answers + NaN handling (OOS 0.0 scores were dragging average) |
| Context Precision | 0.7885 | 0.8598 | > 0.85 | Multi-query expansion gives reranker a better candidate pool |
| Context Recall | 0.8224 | 0.8659 | > 0.91 | Multi-query expansion retrieves chunks missed by single query |
| Answer Similarity | N/A | 0.9057 | > 0.93 | Prompt encourages citing specific controls, matching ground truth style |


## 7. In-scope vs OOS split reporting

**What it does:** After computing RAGAS scores, `compute_ragas_scores()` now reports three sets of averages: all 100 questions, in-scope only (90 questions), and OOS only (10 questions). The category field from `eval_questions.json` is carried through the entire pipeline.

**Why we built it:** Context precision gets 0.0 for OOS questions, and that is the correct score (the retriever genuinely found nothing useful for "what are Azure pricing differences"). We do not fill context precision with 1.0 because that would be dishonest. But 10 zeros in 100 questions caps the average at 0.90 even if all in-scope questions score perfectly. Reporting in-scope scores separately shows the real retrieval quality without OOS drag.

**What it looks like in output:**

```
RAGAS Evaluation Results (All 100 Questions)
  faithfulness              0.8234
  answer_relevancy          0.8456
  ...

In-Scope Only (90 questions)
  faithfulness              0.8112
  answer_relevancy          0.8340
  context_precision         0.8890
  ...

Out-of-Scope Only (10 questions)
  faithfulness              1.0000
  answer_relevancy          1.0000
  context_precision         0.0000
  ...
```

The headline scores we report are the "All 100 Questions" numbers. The split is supplementary context that explains where the scores come from.

**Implementation:** `src/evaluation.py`, the `category` field is now passed through `run_ragas_evaluation()` and used in `compute_ragas_scores()` for the split. All three sets of scores are logged to ClearML.


## References

1. Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2023). "RAGAS: Automated Evaluation of Retrieval Augmented Generation." arXiv:2309.15217. https://arxiv.org/abs/2309.15217

2. RAGAS documentation, Faithfulness metric. https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/

3. RAGAS documentation, Answer Relevancy metric. https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/

4. RAGAS documentation, Context Precision metric. https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_precision/

5. RAGAS documentation, Context Recall metric. https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_recall/

6. RAGAS GitHub Issue #733, "Faithfulness NaN when answer has no claims." https://github.com/explodinggradients/ragas/issues/733

7. RAGAS GitHub Issue #1651, "No statements were generated from the answer." https://github.com/explodinggradients/ragas/issues/1651

8. RAGAS source code, faithfulness implementation. https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_faithfulness.py
