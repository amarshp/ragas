# RAGAS Interview Prep — Focused Study Guide

> Only what matters for interviews. Concepts first, implementation details second.

---

## Part 1: The 4 Core Metrics (Know Deeply)

### 1. Faithfulness

> *"Is the response grounded in the retrieved context?"*

**Formula**:

$$\text{Faithfulness} = \frac{\text{claims in response supported by context}}{\text{total claims in response}}$$

**How it works** (2-step LLM pipeline):
1. **Extract** atomic claims from the response
2. **Verify** each claim against retrieved context via NLI → supported (1) or not (0)

**Example**:

| Claim | Context says… | Verdict |
|---|---|---|
| "Einstein was born in Germany" | "born in Ulm, Germany" | 1 ✓ |
| "He was born March 20, 1879" | "born 14 March 1879" | 0 ✗ |

Score = 1/2 = **0.5**

**Key interview points**:
- Measures *groundedness*, NOT quality — a response that copies context verbatim scores 1.0 but may be useless
- Reference-free → usable in production
- Costs 2 LLM calls per sample
- `FaithfulnesswithHHEM` variant uses a T5 model instead of LLM → deterministic, zero LLM cost

---

### 2. Context Precision

> *"Does the retriever rank relevant chunks higher than irrelevant ones?"*

**Formula** (Average Precision):

$$\text{Context Precision@K} = \frac{\sum_{k=1}^{K} \text{Precision@k} \times v_k}{\text{number of relevant items in top K}}$$

where $v_k = 1$ if chunk at rank $k$ is relevant, else 0.

**How it differs from standard Precision@K**:

| | Standard Precision@K | Context Precision (RAGAS) |
|---|---|---|
| **"Relevant" means** | Useful for the *question* | Useful in arriving at the *answer* |
| **Ranking reward** | No — flat ratio | Yes — early relevant chunks score higher |
| **Formula** | relevant/K | Weighted average precision |

**Algorithm**:
```
For each chunk at position k:
  LLM asks: "Was this useful in arriving at the answer?"
  → verdict = 1 or 0

cumsum = 0, numerator = 0
for i, verdict in enumerate(verdicts):
    cumsum += verdict
    if verdict == 1:
        numerator += cumsum / (i + 1)
score = numerator / (cumsum + epsilon)
```

**Key interview points**:
- Answer-oriented, not question-oriented
- Order matters — burying a relevant chunk at rank 5 hurts the score
- Has a reference-free variant (`WithoutReference`) → usable in production
- Costs K LLM calls (one per chunk)

---

### 3. Context Recall

> *"Did the retriever find all the information needed for the correct answer?"*

**Formula**:

$$\text{Context Recall} = \frac{\text{reference claims attributable to retrieved context}}{\text{total reference claims}}$$

**How it differs from standard Recall@K**:

| | Standard Recall@K | Context Recall (RAGAS) |
|---|---|---|
| **"Relevant" means** | Document is useful for the *question* | Reference claim is covered by the context |
| **Unit** | Whole documents | Individual claims |
| **Denominator** | Total relevant docs in corpus | Total claims in reference answer |

**Algorithm**:
```
1. Break reference answer into claims
2. For each claim, LLM asks: "Can this be attributed to any retrieved chunk?"
3. Score = attributed claims / total claims
```

**Key interview points**:
- Answer-oriented — measures coverage of the *reference answer*, not retrieval of question-relevant docs
- Always requires a reference → **offline only**, never in production
- Only two numbers matter: claims covered and total claims. Number of chunks, which chunk, relevance to question — all distractors
- That's why it's more useful than standard Recall@K for RAG — you don't care if the retriever found "relevant documents" in the abstract. You care if it found the documents that contain the information needed to produce the correct answer.
- Costs 1 LLM call per sample

---

### 4. Answer Relevancy

> *"Does the response actually address the question?"*

**Formula**:

$$\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \cos(E_{g_i}, E_o)$$

where $E_{g_i}$ = embedding of generated question $i$, $E_o$ = embedding of original question.

**How it works** (reverse-engineering):
1. Generate N questions the response could answer
2. Embed those questions + the original question
3. Average cosine similarity
4. If response is non-committal ("I don't know") → score forced to 0

**Why this works**: If the response truly answers the question, reverse-engineered questions should be semantically similar to the original.

**Intuition by example**:

*Good response* — Original question: *"What is the capital of France?"*
- Response: *"Paris is the capital of France, located on the Seine river."*
- Generated questions from response: *"What is France's capital?"*, *"Which city is the capital of France?"*, *"What is the capital city of France?"*
- These are very similar to the original → high cosine similarity → **high score**

*Bad response (off-topic)* — Original question: *"What is the capital of France?"*
- Response: *"France has a population of 67 million and is known for wine."*
- Generated questions: *"What is France known for?"*, *"What is the population of France?"*
- These are **not similar** to the original → low cosine similarity → **low score**

**Why not just compare the response to the question directly?** A good answer rarely *looks like* the question textually. "Paris" doesn't resemble "What is the capital of France?" But a question reverse-engineered from "Paris is the capital..." *does* resemble the original. The reverse-engineering step creates a comparable representation.

**The blind spot**: A confident but *wrong* answer like *"Lyon is the capital of France"* generates questions like *"What is the capital of France?"* → high similarity → high score. It measures **topicality**, not **correctness**. That's why you need Factual Correctness separately.

**Key interview points**:
- Reference-free → usable in production
- Cannot detect factual errors — a confidently wrong answer scores high
- Uses both LLM (generate questions) and embeddings (cosine similarity)
- Costs N LLM calls + 1 embedding call (default N=3)

---

## Part 2: Know Conceptually (Tier 2)

### 5. Factual Correctness

- Decomposes both response and reference into claims
- Uses NLI to classify: TP (response claim matches reference), FP (doesn't), FN (reference claim missing from response)
- Returns precision, recall, or F1
- **Distinct from Faithfulness**: Faithfulness checks against *context*, Factual Correctness checks against *reference*
- Requires reference → offline only

### 6. Answer Correctness

- Composite score: $w \cdot F1_{\text{factual}} + (1-w) \cdot \text{SemanticSimilarity}$
- Default weights: $w = 0.75$ factuality, $0.25$ semantic similarity
- **Semantic Similarity**: embeds the full response and full reference strings, then computes cosine similarity (L2-normalize both vectors, dot product). Uses whatever embedding model is configured (typically OpenAI `text-embedding-ada-002`).
- Combines factual accuracy with meaning closeness
- Requires reference → offline only

### 7. Aspect Critic

- Custom binary judge: you provide a definition, LLM returns 0 or 1
- Runs 3 times with majority vote for robustness
- The go-to tool for custom production checks (safety, tone, format)
- Reference-free → usable in production

### 8. Noise Sensitivity

- **Lower is better** (unlike all other metrics)
- Tests robustness when retriever returns noisy/misleading chunks
- Requires reference → offline only

**How incorrect claims are measured** (3 NLI passes):
1. Decompose both response and reference into atomic claims
2. Check each response claim against the **reference** via NLI → not inferable = **incorrect**
3. Check each response claim against each **retrieved context** → determines which context sourced the claim
4. Check each ground truth claim against each **context** → a context is "relevant" if any ground truth claim can be inferred from it

**Score** = fraction of response claims that are both **(a) incorrect** (not supported by reference) AND **(b) attributable to a retrieved context**

**Two variants**:
- **Relevant** (default): incorrect claims sourced from contexts that *do* contain some ground truth info — the context was topically on-point but still led the model astray
- **Irrelevant**: incorrect claims sourced from contexts with *no* ground truth support — pure noise that polluted the answer

Both range 0–1. Diagnostic: tells you *where* the errors come from (misleading relevant contexts vs. pure noise).

---

### 9. Classic Baseline Metrics (for contrast)

These are not RAGAS-specific but you need crisp definitions to contrast with RAGAS metrics.

**Standard Precision@K**:
$$\text{Precision@K} = \frac{\text{relevant documents in top K}}{K}$$
"Of the K documents I retrieved, how many are relevant to the *question*?" Flat ratio — no ranking reward, question-oriented.

**Standard Recall@K**:
$$\text{Recall@K} = \frac{\text{relevant documents in top K}}{\text{total relevant documents in corpus}}$$
"Of all relevant documents in the corpus, how many did I find?" Requires knowing the full set of relevant docs — only usable offline.

**BLEU** (Bilingual Evaluation Understudy):
N-gram precision of response vs reference. Counts overlapping word sequences (unigrams through 4-grams) with a brevity penalty. Fast, deterministic, zero cost. Misses paraphrases ("quick" vs "fast" score 0 even though semantically identical).

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation):
N-gram recall of response vs reference. ROUGE-1 = unigram overlap, ROUGE-L = longest common subsequence. Same pros/cons as BLEU — cheap and deterministic but surface-level.

**Why RAGAS improves on all of these**:
- Context Precision/Recall are *answer-oriented* (vs question-oriented) and *claim-level* — they measure what actually matters for RAG quality
- Faithfulness and factual metrics use NLI rather than surface n-gram overlap → catch semantically equivalent paraphrases and semantic errors BLEU/ROUGE miss

---

## Part 3: Cross-Cutting Concepts

### Reference-Free vs Reference-Required

| Can use in production (no reference) | Offline only (needs reference) |
|---|---|
| Faithfulness | Context Recall |
| Answer Relevancy | Factual Correctness |
| Context Precision (w/o ref) | Answer Correctness |
| Aspect Critic | Noise Sensitivity |
| | BLEU / ROUGE / Exact Match |

**Interview phrasing**: *"In production you don't have ground truth answers. That limits you to Faithfulness, Answer Relevancy, and Aspect Critic for monitoring. Context Recall and Factual Correctness can only run offline against curated test sets."*

### Answer-Oriented vs Question-Oriented

| Metric | Evaluates against… |
|---|---|
| Standard Precision@K / Recall@K | The *question* |
| Context Precision (RAGAS) | The *answer* (was this chunk useful in arriving at the answer?) |
| Context Recall (RAGAS) | The *answer* (can this reference claim be attributed to context?) |

This is the single most important conceptual distinction in RAGAS.

### LLM vs Non-LLM Tradeoff

| | LLM-based | Non-LLM |
|---|---|---|
| Accuracy | Higher (closer to human) | Lower |
| Cost | $$ (API calls) | Free |
| Speed | Slow | Fast |
| Deterministic | No | Yes |
| Scale | Sample at 10K+ | Run on everything |

**Interview phrasing**: *"At scale, I'd use BLEU/ROUGE as CI gates and sample 500-1000 for LLM metrics in nightly runs."*

### Cost Math

| Metric | LLM Calls / Sample |
|---|---|
| Faithfulness | 2 |
| Context Precision | K (per chunk) |
| Context Recall | 1 |
| Answer Relevancy | 3 + embed |
| Aspect Critic | 3 |
| BLEU / ROUGE | 0 |

**Quick calc**: 1,000 samples × (Faithfulness + Context Recall + Answer Relevancy) = 6,000 LLM calls ≈ $20 at GPT-4o pricing.

### Meta-Evaluation

> "How do you know your metrics are trustworthy?"

1. Take 50-100 samples
2. Have human annotators score them
3. Compute correlation between human and RAGAS scores
4. \> 0.8 = trust it. 0.6-0.8 = tune prompts. < 0.6 = switch model or fall back to non-LLM

Also check inter-annotator agreement (Cohen's κ) — if humans disagree, no metric will be reliable.

### The Circularity Problem

LLM evaluating LLM is circular. GPT-4 evaluating GPT-4 can have self-preference bias.

**Mitigations**: Use different model family as evaluator, calibrate against human annotations, use non-LLM metrics as cross-check.

---

## Part 4: Interview Questions & Answers

### Foundational

**Q1: What is RAGAS?**

Open-source Python framework for evaluating LLM applications (especially RAG). Provides objective metrics for retrieval quality, generation faithfulness, and response quality using a combination of LLM-based and traditional metrics.

---

**Q2: Name the 4 core RAG metrics and what each measures.**

1. **Faithfulness** — is the response grounded in the retrieved context? (hallucination detection)
2. **Context Precision** — are relevant chunks ranked higher? (retrieval ranking quality)
3. **Context Recall** — did we retrieve everything needed for the correct answer? (retrieval completeness)
4. **Answer Relevancy** — does the response actually address the question? (response quality)

---

**Q3: How does Faithfulness work internally?**

Two-step LLM pipeline: (1) extract atomic claims from the response, (2) verify each claim against context via NLI. Score = supported claims / total claims. A score of 1.0 means zero hallucination relative to context.

---

**Q4: What's the difference between Faithfulness and Factual Correctness?**

- **Faithfulness**: checks response claims against *retrieved context* — "is it grounded?"
- **Factual Correctness**: checks response claims against *reference answer* — "is it correct?"

A response can be faithful (everything comes from context) but factually incorrect (the context itself was wrong). Conversely, a response can be factually correct but unfaithful (hallucinated the right answer without context support).

---

**Q5: How does RAGAS Context Precision differ from standard Precision@K?**

Three ways: (1) relevance is judged against the *answer*, not the *question*. (2) It uses Average Precision, so ranking order matters — relevant chunks placed earlier score higher. (3) The LLM prompt asks "was this useful in arriving at the answer?" not "is this relevant to the query?"

---

**Q6: How does RAGAS Context Recall differ from standard Recall@K?**

Standard Recall@K counts relevant *documents* retrieved vs. total relevant in corpus. RAGAS Context Recall decomposes the *reference answer* into claims and checks how many can be attributed to any retrieved chunk. It's answer-oriented and claim-level, not document-level.

---

**Q7: Explain how Answer Relevancy uses reverse-engineering.**

Instead of comparing response to question directly, it: (1) generates N questions the response could answer, (2) embeds those + the original question, (3) computes average cosine similarity. If the response truly answers the question, reverse-engineered questions should be similar to the original. This works without a reference answer.

---

### Practical

**Q8: Which metrics can you use in production without reference answers?**

Faithfulness (grounded in context?), Answer Relevancy (addresses the question?), Context Precision without reference (relevant chunks?), and Aspect Critic (custom safety/tone checks). Everything else requires a reference and is offline-only.

---

**Q9: How many LLM calls does evaluating 1,000 samples with Faithfulness + Context Recall + Answer Relevancy take? What does it cost?**

Faithfulness: 2,000. Context Recall: 1,000. Answer Relevancy: 3,000 + 1,000 embed. Total: ~6,000 LLM calls. At GPT-4o pricing: ~$20. At scale (>10K), sample for LLM metrics and use BLEU/ROUGE for full coverage.

---

**Q10: How would you set up RAG evaluation in production?**

1. Log queries, responses, and contexts
2. Sample strategically (not everything)
3. Run Faithfulness + Answer Relevancy periodically (no reference needed)
4. Use Aspect Critic for safety/tone checks
5. Alert on score degradation (e.g., faithfulness < 0.8)
6. Feed flagged cases back into offline test set
7. Run Context Recall + Factual Correctness offline against curated dataset

---

**Q11: How do you know your evaluation metrics are trustworthy?**

Meta-evaluation: score 50-100 samples with both RAGAS and human annotators. Compute correlation. >0.8 = trust. 0.6-0.8 = tune prompts. <0.6 = switch evaluator model or use non-LLM metrics. Also check inter-annotator agreement — if humans disagree (κ < 0.6), the task is inherently subjective.

---

**Q12: What are the main limitations of RAGAS?**

1. **Circularity** — LLM judging LLM can share blind spots. Mitigate with different model families.
2. **Faithfulness gaming** — copying context verbatim scores 1.0 but is useless. Always pair with Answer Relevancy.
3. **Answer Relevancy misses factual errors** — a wrong but on-topic answer scores high. Need Factual Correctness for actual accuracy.
4. **Reference dependency** — most useful metrics (Recall, Factual Correctness) need ground truth, unavailable in production.
5. **Non-determinism** — same eval can yield different scores. Report confidence intervals.
6. **Domain sensitivity** — default prompts tuned for general knowledge; specialized domains need prompt adaptation.

---

**Q13: Faithfulness is 1.0 but the answer is clearly bad. How is that possible?**

The response only parrots the context verbatim. Every claim is trivially "supported" because it's copied. Faithfulness measures groundedness, not quality. The response could be irrelevant, incomplete, or poorly structured. This is why you always pair Faithfulness with Answer Relevancy:

| Response Type | Faithfulness | Answer Relevancy | Quality |
|---|---|---|---|
| Parrot (copies context) | 1.0 | Low | Bad |
| Hallucinated but fluent | Low | High | Bad |
| Grounded + relevant | High | High | Good |

---

**Q14: Answer Relevancy is 0.9 but the answer is factually wrong. How?**

Answer Relevancy only checks if the response *addresses* the question, not if it's *correct*. "Python was created in 2005" scores high for relevancy against "When was Python created?" because it's on-topic. You need Factual Correctness or Answer Correctness (both require a reference) to catch factual errors.

---

### Scenario-Based

**Q15: 10 retrieved chunks, 7 relevant to the question, reference has 4 claims, only 2 attributable to any chunk. What's Context Recall?**

**0.5** (= 2/4). The 10 chunks, 7 relevant-to-question — all distractors. Context Recall only cares: how many reference claims are covered (2) out of total reference claims (4). It's answer-oriented, not question-oriented.

---

**Q16: You need to evaluate 50,000 samples. How do you approach this?**

Tiered strategy:
- **Full coverage** (50K): BLEU, ROUGE, Semantic Similarity — free, deterministic, fast
- **Sampled** (500-1,000): Faithfulness, Answer Relevancy, Factual Correctness — ~$20, ~20 min
- **CI gate**: ROUGE as automated regression check on every commit
- **Nightly**: LLM metrics on stratified sample
- Cost reduction: use GPT-4o-mini as evaluator (verify correlation first), enable DiskCacheBackend

---

**Q17: Your team wants to evaluate a customer service chatbot. Which metrics and how?**

- **Faithfulness** — is it making stuff up? (production, no reference needed)
- **Answer Relevancy** — is it actually answering the customer's question? (production)
- **Aspect Critic** with custom definitions:
  - "Is the response polite and professional?" (tone)
  - "Does the response avoid discussing competitor products?" (topic boundary)
  - "Does the response include a resolution or next step?" (completeness)
- **Context Recall** — offline against curated Q&A pairs from support docs
- **Topic Adherence** — if agent-based, ensure it doesn't go off-script

---

**Q18: You see Faithfulness scores dropping from 0.85 to 0.72 over 2 weeks. Diagnosis?**

Possible causes:
1. **Retriever degradation** — index drift, stale embeddings, new docs not indexed → check Context Precision
2. **Prompt change** — someone modified the generation prompt to be more creative → check recent commits
3. **Model update** — provider silently updated the model → check if using pinned model version
4. **Data distribution shift** — new types of queries the retriever wasn't designed for → slice by query category
5. **Evaluator instability** — if evaluator model was also updated, scores may drift without real quality change → run meta-evaluation on held-out set

---

**Q19: Context Precision is high but Context Recall is low. What does that mean?**

The retriever is precise but incomplete. It returns *good* chunks (relevant ones are ranked well) but *not enough* of them. The retrieved context covers some reference claims but misses others.

Fix: increase top-K, add query expansion, use hybrid search (dense + sparse), or improve document coverage in the index.

---

**Q20: Context Recall is high but Context Precision is low. What does that mean?**

The retriever is complete but noisy. It finds everything needed (all reference claims are covered) but also retrieves a lot of irrelevant chunks. This wastes context window tokens and can confuse the generator (see Noise Sensitivity).

Fix: reduce top-K, improve retriever re-ranking, add a filtering step, or use a cross-encoder reranker.

---

**Q21: Why would you use a different LLM as evaluator than the one generating responses?**

To avoid self-preference bias. GPT-4 evaluating GPT-4 outputs tends to rate its own style higher and share the same blind spots — it won't catch hallucinations the generation model also "believes." Using Claude to evaluate GPT-4 (or vice versa) reduces this circularity.

---

**Q22: What's the point of non-LLM metrics like BLEU/ROUGE when LLM metrics are more accurate?**

- **Free** — zero API cost at any scale
- **Deterministic** — same input always gives same output
- **Fast** — milliseconds vs seconds per sample
- **Scalable** — run on 100K samples trivially
- **CI-friendly** — regression gate on every commit
- Use them as cheap filters; reserve expensive LLM metrics for sampled deep evaluation

---

## Quick Reference Card

```
PRODUCTION MONITORING (no reference):
  Faithfulness      → hallucination detection      (2 LLM calls)
  Answer Relevancy  → off-topic detection           (3 LLM + embed)
  Aspect Critic     → custom safety/tone checks     (3 LLM calls)

OFFLINE EVALUATION (with reference):
  Context Recall    → retriever completeness         (1 LLM call)
  Context Precision → retriever ranking quality      (K LLM calls)
  Factual Correctness → end-to-end accuracy          (3 LLM calls)
  BLEU/ROUGE        → cheap regression check         (0 LLM calls)
```

**The one sentence for each:**
- Faithfulness = "claims supported by context / total claims"
- Context Precision = "average precision, answer-oriented, ranking matters"
- Context Recall = "reference claims covered / total reference claims"
- Answer Relevancy = "cosine similarity of reverse-engineered questions"
- Factual Correctness = "TP/FP/FN claim comparison against reference"
- Aspect Critic = "custom binary LLM judge with majority vote"

---

*Distilled from the full course. Concepts are durable; class names change every release.*
