# RAGAS: Complete Interview-Level Course

> A comprehensive study guide for mastering the RAGAS evaluation framework — from fundamentals to advanced internals — built directly from the source repository.

---

## Table of Contents

1. [What is RAGAS?](#1-what-is-ragas)
2. [Why Evaluation Matters for LLM Applications](#2-why-evaluation-matters-for-llm-applications)
3. [Core Architecture & Concepts](#3-core-architecture--concepts)
4. [Evaluation Datasets & Schemas](#4-evaluation-datasets--schemas)
5. [Metrics — The Heart of RAGAS](#5-metrics--the-heart-of-ragas)
6. [Deep Dive: RAG Metrics](#6-deep-dive-rag-metrics)
7. [Deep Dive: Agent & Tool Use Metrics](#7-deep-dive-agent--tool-use-metrics)
8. [Deep Dive: NLP Comparison Metrics](#8-deep-dive-nlp-comparison-metrics)
9. [Deep Dive: General-Purpose Metrics](#9-deep-dive-general-purpose-metrics)
10. [Test Data Generation](#10-test-data-generation)
11. [The Experiment Framework](#11-the-experiment-framework)
12. [Custom Metrics — Building Your Own](#12-custom-metrics--building-your-own)
13. [Prompt Engineering in RAGAS](#13-prompt-engineering-in-ragas)
14. [Integrations Ecosystem](#14-integrations-ecosystem)
15. [Feedback Loops & Continuous Improvement](#15-feedback-loops--continuous-improvement)
16. [Code Walkthrough — Key Source Modules](#16-code-walkthrough--key-source-modules)
17. [Hands-On: End-to-End RAG Evaluation](#17-hands-on-end-to-end-rag-evaluation)
18. [Hands-On: Agent Evaluation](#18-hands-on-agent-evaluation)
19. [Interview Questions & Answers](#19-interview-questions--answers)
20. [Quick Reference Cheat Sheet](#20-quick-reference-cheat-sheet)

---

## 1. What is RAGAS?

**RAGAS** (Retrieval Augmented Generation Assessment) is an open-source Python framework for objectively evaluating LLM-powered applications — particularly RAG pipelines — using a combination of LLM-based and traditional metrics.

### Key Value Propositions

| Capability | Description |
|---|---|
| **Objective Metrics** | Precision-grade evaluation using both LLM-based and traditional metrics |
| **Test Data Generation** | Automatic creation of comprehensive test datasets via knowledge graphs |
| **Seamless Integrations** | Works with LangChain, LlamaIndex, Haystack, Amazon Bedrock, and more |
| **Feedback Loops** | Leverages production data for continuous application improvement |
| **Experimentation** | Structured experiment tracking with CSV-based result storage |

### Installation

```bash
# PyPI (recommended)
pip install ragas

# From source
pip install git+https://github.com/explodinggradients/ragas

# With LangChain support
pip install -U "langchain-core>=0.2,<0.3" "langchain-openai>=0.1,<0.2" openai
```

### CLI Quickstart

```bash
# Scaffold a new evaluation project
uvx ragas quickstart rag_eval
cd rag_eval
uv sync
export OPENAI_API_KEY="your-key"
uv run python evals.py
```

This generates:
```
rag_eval/
├── README.md              # Project documentation
├── pyproject.toml         # Project configuration
├── rag.py                 # Your RAG application
├── evals.py               # Evaluation workflow
├── __init__.py
└── evals/
    ├── datasets/          # Test data (CSV)
    ├── experiments/       # Evaluation results
    └── logs/              # Execution logs
```

---

## 2. Why Evaluation Matters for LLM Applications

### The Core Problem

LLM applications (especially RAG pipelines) are non-deterministic. Unlike traditional software where unit tests give binary pass/fail, LLM outputs require **nuanced, multi-dimensional assessment**.

### What Metrics Enable

| Function | Example |
|---|---|
| **Component Selection** | Compare Pinecone vs. Weaviate retriever with your own data |
| **Error Diagnosis** | Identify whether retrieval or generation is causing bad answers |
| **Continuous Monitoring** | Track faithfulness drift after production deployment |
| **A/B Testing** | Measure impact of prompt changes quantitatively |

### The Golden Rule

> *"You can't improve what you don't measure."*
> Metrics are the feedback loop that makes iteration possible.

---

## 3. Core Architecture & Concepts

### High-Level API Surface

```python
from ragas import (
    # Dataset handling
    EvaluationDataset, SingleTurnSample, MultiTurnSample, Dataset, DataTable,
    # Evaluation
    evaluate, aevaluate,
    # Experimentation
    experiment, Experiment, version_experiment,
    # Configuration
    RunConfig,
    # Caching
    CacheInterface, DiskCacheBackend, cacher,
    # Tokenizers
    BaseTokenizer, HuggingFaceTokenizer, TiktokenWrapper, get_tokenizer,
)
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      User Application                        │
│                  (RAG Pipeline / Agent / LLM)                │
└─────────────────┬───────────────────────────────────────────┘
                  │ Responses + Contexts
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   EvaluationDataset                           │
│          SingleTurnSample / MultiTurnSample                  │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    evaluate() / aevaluate()                   │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │Faithfulness│  │Precision │  │ Recall   │  │Relevancy │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                              │
│              Executor (async, batched, parallel)             │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   EvaluationResult                            │
│          Scores per metric + per sample breakdown            │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Async-first**: All scoring is `async` — enables parallel LLM calls
2. **Pydantic schemas**: Input/output validation via Pydantic models
3. **Prompt-as-object**: Prompts are composable objects with instruction + examples + I/O models
4. **Backend-agnostic storage**: CSV, JSONL, Google Drive, in-memory
5. **LLM-agnostic**: Works with OpenAI, Anthropic, Google, Ollama, or any OpenAI-compatible endpoint

---

## 4. Evaluation Datasets & Schemas

### Sample Types

#### `SingleTurnSample` — For single Q&A interactions

```python
from ragas.dataset_schema import SingleTurnSample

sample = SingleTurnSample(
    user_input="What is photosynthesis?",
    response="Photosynthesis is the process by which plants convert sunlight to energy.",
    reference="Photosynthesis is the process used by plants to convert light energy into chemical energy.",
    retrieved_contexts=[
        "Plants use sunlight, water, and CO2 to produce glucose and oxygen.",
        "The process occurs primarily in the leaves of plants."
    ],
    reference_contexts=["Photosynthesis converts light energy into chemical energy in plants."],
)
```

**Available fields:**

| Field | Type | Purpose |
|---|---|---|
| `user_input` | `str` | The user's question or query |
| `response` | `str` | The system's generated answer |
| `reference` | `str` | Ground truth / expected answer |
| `retrieved_contexts` | `list[str]` | Contexts retrieved by the system |
| `reference_contexts` | `list[str]` | Ideal contexts (ground truth) |
| `rubric` | `dict` | Scoring rubric for evaluation |
| `persona_name` | `str` | Persona used for generation |
| `query_style` | `str` | Style of query (web search, chat) |
| `query_length` | `str` | Length category (short, medium, long) |

#### `MultiTurnSample` — For conversational / agent interactions

```python
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall

sample = MultiTurnSample(
    user_input=[
        HumanMessage(content="What's the weather in NYC?"),
        AIMessage(content="Let me check.", tool_calls=[
            ToolCall(name="get_weather", args={"city": "NYC"})
        ]),
        ToolMessage(content='{"temp": "72°F", "condition": "sunny"}'),
        AIMessage(content="It's 72°F and sunny in NYC!"),
    ],
    reference="The weather in NYC is 72°F and sunny.",
    reference_tool_calls=[
        ToolCall(name="get_weather", args={"city": "NYC"})
    ],
)
```

**Message sequence validation**: Must follow `HumanMessage → AIMessage → ToolMessage → ...` pattern.

### `EvaluationDataset`

```python
from ragas import EvaluationDataset

dataset = EvaluationDataset(samples=[sample1, sample2, sample3])

# Properties
dataset.is_multi_turn()       # True if contains MultiTurnSamples
dataset.get_sample_type()     # Returns sample class type
dataset.features()            # Returns available feature names

# Interoperability
df = dataset.to_pandas()
hf = dataset.to_hf_dataset()
dataset.to_csv("eval_data.csv")
dataset.to_jsonl("eval_data.jsonl")

# Reconstruct from various sources
dataset = EvaluationDataset.from_pandas(df)
dataset = EvaluationDataset.from_hf_dataset(hf_dataset)
dataset = EvaluationDataset.from_jsonl("eval_data.jsonl")
```

### Best Practices for Datasets

1. **Representative samples** — Mirror real-world query distribution
2. **Balanced distribution** — Cover easy, medium, hard cases and edge cases
3. **Quality over quantity** — 50 great samples > 500 noisy ones
4. **Metadata-rich** — Enable slicing analysis by topic, difficulty, source
5. **Version-controlled** — Track dataset evolution over time

---

## 5. Metrics — The Heart of RAGAS

### Taxonomy of Metrics

```
                         ┌──────────────────┐
                         │     Metric       │  (Abstract Base)
                         └───────┬──────────┘
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
            ┌──────────┐ ┌──────────────┐ ┌──────────────────┐
            │  Metric  │ │MetricWithLLM │ │MetricWithEmbeddings│
            │(non-LLM) │ │+ PromptMixin │ │                    │
            └──────────┘ └──────────────┘ └──────────────────┘
                    │            │
           ┌───────┼───────┐    │
           ▼                ▼   ▼
    ┌──────────────┐  ┌──────────────┐
    │SingleTurnMetric│  │MultiTurnMetric│
    │single_turn_    │  │multi_turn_    │
    │  ascore()      │  │  ascore()     │
    └──────────────┘  └──────────────┘
```

### Classification by Implementation

| Type | Base Class | Deterministic? | Accuracy | Example |
|---|---|---|---|---|
| **LLM-based** | `MetricWithLLM` | No | Higher (closer to human) | Faithfulness, Context Recall |
| **Non-LLM** | `Metric` | Yes | Lower | BLEU, ROUGE, Exact Match |
| **Embedding-based** | `MetricWithEmbeddings` | Yes | Medium | Answer Relevancy, Semantic Similarity |

### Classification by Output Type

| Output Type | Class | Values | Use Case |
|---|---|---|---|
| **Discrete** | `DiscreteMetric` | Categorical (pass/fail, good/bad) | Classification tasks |
| **Numeric** | `NumericMetric` | Float/int in range (e.g., 0–1) | Continuous scoring |
| **Ranking** | `RankingMetric` | Ordered list | Comparing multiple outputs |

### Classification by Interaction Pattern

| Pattern | Class | Method | Input |
|---|---|---|---|
| **Single-Turn** | `SingleTurnMetric` | `single_turn_ascore()` | `SingleTurnSample` |
| **Multi-Turn** | `MultiTurnMetric` | `multi_turn_ascore()` | `MultiTurnSample` |

### Required Columns System

Each metric declares which `SingleTurnSample`/`MultiTurnSample` fields it needs:

```python
_required_columns: Dict[MetricType, Set[str]] = {
    MetricType.SINGLE_TURN: {
        "user_input",
        "response",
        "retrieved_contexts",          # required
        "reference:optional",          # optional (suffix marker)
    }
}
```

**Valid columns**: `user_input`, `retrieved_contexts`, `reference_contexts`, `response`, `reference`, `rubric`

### Choosing the Right Metrics — Decision Framework

1. **Prioritize end-to-end metrics** — They reflect real user satisfaction
2. **Ensure interpretability** — The entire team should understand what a score means
3. **Prefer objective over subjective** — Target ≥80% inter-rater agreement
4. **Few strong signals > many weak signals** — 3-4 carefully chosen metrics beat 10 noisy ones

---

## 6. Deep Dive: RAG Metrics

### 6.1 Faithfulness

> *"Is the response factually grounded in the retrieved context?"*

**Score Range**: 0.0 to 1.0

**Formula**:

$$\text{Faithfulness} = \frac{|\text{claims in response supported by context}|}{|\text{total claims in response}|}$$

**Algorithm** (2-step LLM pipeline):

```
Step 1: Statement Extraction
  Input:  (user_input, response)
  Output: List of atomic factual statements
  Example: "Einstein was born in Germany and studied physics"
           → ["Einstein was born in Germany.", "Einstein studied physics."]

Step 2: Natural Language Inference (NLI)
  Input:  (retrieved_contexts, list of statements)
  Output: For each statement → verdict (1=supported, 0=not supported) + reason
```

**Implementation skeleton**:

```python
@dataclass
class Faithfulness(MetricWithLLM, SingleTurnMetric):
    name: str = "faithfulness"
    _required_columns = {
        MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
    }

    async def _ascore(self, row, callbacks) -> float:
        statements = await self._create_statements(row, callbacks)   # Step 1
        verdicts = await self._create_verdicts(row, statements, callbacks)  # Step 2
        supported = sum(v.verdict for v in verdicts)
        return supported / len(verdicts) if verdicts else 0.0
```

**Variant — `FaithfulnesswithHHEM`**: Uses Vectara's T5-based hallucination evaluation model instead of LLM NLI for faster, cheaper, deterministic scoring.

**Example walkthrough**:

| Statement | Context Support | Verdict |
|---|---|---|
| "Einstein was born in Germany" | "Albert Einstein was born in Ulm, Germany" | 1 ✓ |
| "Einstein was born on March 20, 1879" | "born on 14 March 1879" | 0 ✗ |

Score = 1/2 = **0.5**

---

### 6.2 Context Precision

> *"Does the retriever rank relevant chunks higher than irrelevant ones?"*

**Score Range**: 0.0 to 1.0

**Formula (Average Precision@K)**:

$$\text{Context Precision@K} = \frac{\sum_{k=1}^{K} \left( \text{Precision@k} \times v_k \right)}{|\text{relevant items in top K}|}$$

where $v_k = 1$ if item at rank $k$ is relevant, else 0.

**Algorithm**:

```
For each retrieved context chunk at position k:
  Ask LLM: "Was this context useful in arriving at the reference answer?"
  → verdict = 1 (yes) or 0 (no)

Compute average precision:
  cumsum = 0, numerator = 0
  for i, verdict in enumerate(verdicts):
      cumsum += verdict
      if verdict == 1:
          numerator += cumsum / (i + 1)
  score = numerator / (cumsum + epsilon)
```

**Why order matters**: A relevant chunk at position 1 contributes more to precision than one at position 5.

**Implementations**:

| Variant | Reference Needed | Method |
|---|---|---|
| `LLMContextPrecisionWithReference` | Yes (reference answer) | LLM judges relevance against reference |
| `LLMContextPrecisionWithoutReference` | No (uses response) | LLM judges relevance against generated response |
| `NonLLMContextPrecisionWithReference` | Yes (reference contexts) | String similarity (Levenshtein-based) |
| `IDBasedContextPrecision` | Yes (reference context IDs) | Direct ID set intersection |

---

### 6.3 Context Recall

> *"Did the retriever find all the relevant information?"*

**Score Range**: 0.0 to 1.0

**Formula**:

$$\text{Context Recall} = \frac{|\text{reference claims attributable to retrieved context}|}{|\text{total reference claims}|}$$

**Algorithm**:

```
1. Break reference answer into individual sentences/claims
2. For each claim, ask LLM: "Can this claim be attributed to the retrieved context?"
   → verdict = 1 (attributed) or 0 (not attributed)
3. Score = sum(verdicts) / len(verdicts)
```

**Key distinction from Precision**: Recall always requires a reference (ground truth). It measures *completeness* — how much of what *should* have been retrieved actually was.

**Implementations**:

| Variant | Method |
|---|---|
| `LLMContextRecall` | LLM classifies each reference claim as attributable or not |
| `NonLLMContextRecall` | Max similarity between each reference context and retrieved contexts |
| `IDBasedContextRecall` | Set intersection of reference and retrieved context IDs |

---

### 6.4 Answer Relevancy (Response Relevance)

> *"Is the response actually answering the question that was asked?"*

**Score Range**: 0.0 to 1.0

**Formula**:

$$\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \cos\left(E_{g_i},\ E_o\right)$$

where $E_{g_i}$ = embedding of the $i$-th generated question, $E_o$ = embedding of original user input, $N$ = `strictness` parameter (default: 3).

**Algorithm (reverse-engineering approach)**:

```
1. Generate N questions that the response could answer
   LLM Input:  response text
   LLM Output: N questions + "non-committal" flag for each

2. Embed all generated questions and the original question

3. Compute cosine similarity between each generated question and original

4. If ALL generated questions flagged as non-committal → score = 0
   Else → score = mean(cosine_similarities)
```

**Why this works**: If the response truly answers the original question, then questions reverse-engineered from the response should be semantically similar to the original question.

**Non-committal detection**: Catches evasive responses like "I don't know" or "I'm not sure about that" → penalized to 0.

```python
@dataclass
class ResponseRelevancy(MetricWithLLM, MetricWithEmbeddings, SingleTurnMetric):
    strictness: int = 3  # Number of questions to reverse-generate

    _required_columns = {
        MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts:optional"}
    }
```

---

### 6.5 Context Entities Recall

> *"Did the retriever capture the key named entities from the reference?"*

**Score Range**: 0.0 to 1.0

**Formula**:

$$\text{Context Entity Recall} = \frac{|CE_{\text{retrieved}} \cap CE_{\text{reference}}|}{|CE_{\text{reference}}|}$$

where $CE$ = set of named entities extracted from the text.

**Algorithm**:

```
1. Extract named entities from reference answer → reference_entities
2. Extract named entities from retrieved contexts → context_entities
3. Compute set intersection
4. Score = |intersection| / |reference_entities|
```

**Required Columns**: `reference`, `retrieved_contexts`

**Type**: LLM-based (uses LLM for entity extraction)

**Use case**: Ensures retriever captures documents containing the key entities. Useful when entity coverage is more important than full-text coverage.

---

### 6.6 Noise Sensitivity

> *"How much does irrelevant context degrade the response?"*

**Score Range**: 0.0 to 1.0 (**lower is better** — measures error rate)

**Formula**:

$$\text{Noise Sensitivity} = \frac{|\text{incorrect claims in response}|}{|\text{total claims in response}|}$$

**Algorithm**:

```
1. Decompose response into atomic factual claims
2. Verify each claim against the reference answer
3. Count claims that contradict or are unsupported by the reference
4. Score = incorrect_claims / total_claims
```

**Modes**: `relevant` (noise from relevant docs) and `irrelevant` (noise from irrelevant docs)

**Required Columns**: `user_input`, `response`, `reference`, `retrieved_contexts`

**Type**: LLM-based

**Use case**: Measures how robust the generation model is when the retriever returns noisy or misleading chunks alongside relevant ones.

---

### 6.7 Multimodal Faithfulness

> *"Is the response factually consistent with both visual and textual context?"*

**Score Range**: Binary (0 or 1)

**Algorithm**: Same claim-extraction + NLI pipeline as Faithfulness, but the context can include images (paths/URLs). Requires a vision-capable LLM (e.g., GPT-4o).

**Required Columns**: `response`, `retrieved_contexts` (may include image references)

**Type**: LLM-based (vision model)

---

### 6.8 Multimodal Relevance

> *"Is the response relevant to the multimodal context provided?"*

**Score Range**: Binary (0 or 1)

**Required Columns**: `user_input`, `response`, `retrieved_contexts`

**Type**: LLM-based (vision model)

**Use case**: RAG systems that retrieve images alongside text (e.g., product catalogs, medical imaging).

---

## 7. Deep Dive: Agent & Tool Use Metrics

### 7.1 Tool Call Accuracy

> *"Did the agent call the right tools with the right arguments in the right order?"*

**Score Range**: 0.0 to 1.0

**Algorithm**:

```
1. Extract tool calls made by the agent from conversation history
2. Compare against reference_tool_calls:
   a. Match tool names (exact string match)
   b. Match tool arguments (key-value comparison)
3. In strict_order mode (default): sequence alignment matters
   In flexible mode: order-independent matching
4. Score = correctly matched calls / total reference calls
```

**Required Columns**: `user_input` (multi-turn messages), `reference_tool_calls`

**Type**: Non-LLM (exact matching — deterministic)

**Parameters**: `strict_order` (default: `True`)

---

### 7.2 Tool Call F1

> *"Balancing tool call precision and recall."*

**Score Range**: 0.0 to 1.0

**Formula**:

$$\text{Precision} = \frac{|\text{correct tool calls}|}{|\text{predicted tool calls}|}$$

$$\text{Recall} = \frac{|\text{correct tool calls}|}{|\text{reference tool calls}|}$$

$$F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Required Columns**: `user_input`, `reference_tool_calls`

**Type**: Non-LLM (set-based comparison — deterministic)

**Use case**: When the agent might call extra tools (hurts precision) or miss required tools (hurts recall), and you want a balanced view.

---

### 7.3 Agent Goal Accuracy

> *"Did the agent ultimately accomplish what the user wanted?"*

**Score Range**: Binary (0 or 1) or 0.0 to 1.0

**Variants**:

| Class | Reference Needed | Method |
|---|---|---|
| `AgentGoalAccuracyWithReference` | Yes | LLM compares outcome against reference |
| `AgentGoalAccuracyWithoutReference` | No | LLM infers goal from conversation and judges completion |

**Algorithm**:

```
1. Extract the user's goal from conversation history
2. Analyze the agent's final outcome
3. LLM judges: "Did the agent achieve the stated goal?"
4. Returns binary verdict + reasoning
```

**Required Columns**: `user_input` (multi-turn), `reference` (optional)

**Type**: LLM-based

---

### 7.4 Topic Adherence

> *"Did the agent stay within its designated topic boundaries?"*

**Score Range**: 0.0 to 1.0

**Modes**: `precision`, `recall`, `f1`

**Formulas**:

$$\text{Precision} = \frac{|\text{response topics} \cap \text{reference topics}|}{|\text{response topics}|}$$

$$\text{Recall} = \frac{|\text{response topics} \cap \text{reference topics}|}{|\text{reference topics}|}$$

**Algorithm**:

```
1. Extract topics discussed in the agent's responses (LLM-based classification)
2. Compare against reference_topics (predefined allowed topic set)
3. Compute precision/recall/F1 of topic coverage
```

**Required Columns**: `user_input` (multi-turn messages), `reference_topics`

**Type**: LLM-based

**Use case**: Ensuring customer service bots don't go off-script or discuss unauthorized topics.

---

## 8. Deep Dive: NLP Comparison Metrics

### 8.1 Factual Correctness

> *"Are the facts in the response correct compared to the reference?"*

**Score Range**: 0.0 to 1.0

**Modes**: `f1` (default), `precision`, `recall`

**Formulas**:

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN} \qquad F1 = \frac{2 \cdot P \cdot R}{P + R}$$

where TP = response claims supported by reference, FP = response claims not in reference, FN = reference claims missing from response.

**Algorithm**:

```
1. Decompose response into atomic claims
2. Decompose reference into atomic claims
3. Use NLI to classify each response claim as:
   - TP (supported by reference)
   - FP (contradicts or absent from reference)
4. Use NLI to find reference claims missing from response (FN)
5. Compute precision/recall/F1
```

**Parameters**: `atomicity` (high/low — granularity of claim decomposition), `coverage` (high/low)

**Required Columns**: `response`, `reference`

**Type**: LLM-based

---

### 8.2 Answer Correctness

> *"Combining semantic meaning and factual accuracy into a single score."*

**Score Range**: 0.0 to 1.0

**Formula**:

$$\text{Answer Correctness} = w \cdot F1_{\text{factual}} + (1 - w) \cdot \text{Semantic Similarity}$$

where $w$ is a configurable weight (default: 0.5).

**Required Columns**: `user_input`, `response`, `reference`

**Type**: Hybrid (LLM for factual F1 + embeddings for semantic similarity)

---

### 8.3 Semantic Similarity

> *"How close in meaning are the response and reference?"*

**Score Range**: 0.0 to 1.0

**Formula**:

$$\text{Semantic Similarity} = \cos(\mathbf{E}_{\text{response}},\ \mathbf{E}_{\text{reference}})$$

**Required Columns**: `response`, `reference`

**Type**: Embedding-based (deterministic)

**Notes**: Can optionally use cross-encoder models for higher accuracy.

---

### 8.4 Non-LLM String Similarity

> *"Traditional edit-distance comparison without any LLM."*

**Score Range**: 0.0 to 1.0

**Available Distance Measures**:

| Measure | Description |
|---|---|
| **Levenshtein** (default) | Minimum edit operations (insert, delete, substitute) |
| **Hamming** | Number of differing positions (same-length strings) |
| **Jaro** | Matching characters and transpositions |
| **Jaro-Winkler** | Jaro + prefix bonus for common beginnings |

**Required Columns**: `response`, `reference`

**Type**: Non-LLM (deterministic)

---

### 8.5 BLEU Score

> *"N-gram precision for machine-translation-style evaluation."*

**Score Range**: 0.0 to 1.0

**Algorithm**:

```
1. Compute modified n-gram precision for n = 1, 2, 3, 4
   (clipped counts — each reference n-gram used at most once)
2. Compute brevity penalty: BP = exp(1 - ref_len/response_len) if response < reference
3. BLEU = BP × exp(weighted average of log precisions)
```

**Required Columns**: `response`, `reference`

**Type**: Non-LLM (statistical, deterministic)

**Limitation**: Doesn't understand meaning — "The cat sat on the mat" and "The mat sat on the cat" score similarly.

---

### 8.6 ROUGE Score

> *"Recall-oriented n-gram evaluation — the standard for summarization."*

**Score Range**: 0.0 to 1.0

**Variants**:

| Variant | What It Measures |
|---|---|
| **ROUGE-1** | Unigram overlap |
| **ROUGE-L** (default) | Longest Common Subsequence (LCS) |

**Modes**: `f1measure` (default), `precision`, `recall`

**Formula (ROUGE-L)**:

$$R = \frac{|LCS|}{|\text{reference}|} \qquad P = \frac{|LCS|}{|\text{response}|} \qquad F1 = \frac{2PR}{P + R}$$

**Required Columns**: `response`, `reference`

**Type**: Non-LLM (statistical, deterministic)

---

### 8.7 CHRF Score

> *"Character-level F-score — robust to morphological variation."*

**Score Range**: 0.0 to 1.0

**Algorithm**: Computes character-level n-gram precision and recall, then calculates F-score. Better than BLEU for morphologically rich languages (German, Finnish, etc.).

**Required Columns**: `response`, `reference`

**Type**: Non-LLM (statistical, deterministic)

---

### 8.8 Exact Match

> *"Does the response exactly equal the reference?"*

**Score**: Binary — 1 if `response == reference`, else 0.

**Required Columns**: `response`, `reference`

**Type**: Non-LLM (literal comparison)

---

### 8.9 String Presence

> *"Does the response contain the reference string?"*

**Score**: Binary — 1 if `reference in response`, else 0.

**Required Columns**: `response`, `reference`

**Type**: Non-LLM (substring check)

**Use case**: Verifying that a specific keyword, entity, or phrase appears in the output.

---

## 9. Deep Dive: General-Purpose Metrics

### 9.1 Aspect Critic

> *"A single-aspect binary judge — does the response meet this one criterion?"*

**Score**: Binary (0 or 1)

**Algorithm**:

```
1. Construct prompt with the user-defined aspect definition
2. Ask LLM: "Does this response align with the criterion: {definition}?"
3. Repeat 3 times (majority vote for robustness)
4. Return majority verdict (0 or 1)
```

**Required Columns**: `user_input`, `response` (optional: `retrieved_contexts`, `reference`)

**Type**: LLM-based (3 LLM calls with majority vote)

```python
from ragas.metrics import AspectCritic

politeness = AspectCritic(
    name="politeness",
    definition="Is the response polite and professional?",
    llm=evaluator_llm,
)

conciseness = AspectCritic(
    name="conciseness",
    definition="Is the response concise without unnecessary verbosity?",
    llm=evaluator_llm,
)
```

**Use case**: Quick custom binary checks — safety, tone, format compliance, etc.

---

### 9.2 Simple Criteria Score

> *"Flexible scoring on a custom integer or categorical scale."*

**Score**: Integer or categorical (configurable)

**Algorithm**: Similar to Aspect Critic but returns a score from a range instead of binary. The LLM is asked to score and provide reasoning.

**Required Columns**: `user_input`, `response`

**Type**: LLM-based

**Use case**: When you need more granularity than binary but don't want a full rubric.

---

### 9.3 Rubrics-Based Scoring (Domain-Specific Rubrics)

> *"Consistent multi-level rubric applied to all samples."*

**Score**: 1–5 (default) with descriptive level definitions

**Default Rubric**:

| Score | Meaning |
|---|---|
| 1 | Poor — Completely misses the mark |
| 2 | Below Average — Partially addresses the task |
| 3 | Average — Adequately addresses the task |
| 4 | Good — Well addresses with minor gaps |
| 5 | Excellent — Fully addresses with no gaps |

**Algorithm**:

```
1. Present the rubric + sample to LLM
2. LLM assigns a score (1-5) with reasoning
3. Return score + explanation
```

**Required Columns**: `user_input`, `response` (optional: `retrieved_contexts`, `reference`)

**Type**: LLM-based

**Use case**: Standardized quality assessment across a dataset with consistent criteria.

---

### 9.4 Instance-Specific Rubrics

> *"Each sample gets its own custom rubric."*

**Score**: 1–5 (per-sample rubric)

**Key difference from 9.3**: The rubric is read from the `rubric` field of each `SingleTurnSample`, allowing different scoring criteria per question.

**Required Columns**: `rubrics`, `user_input`, `response`

**Type**: LLM-based

**Use case**: Evaluating diverse question types where a single rubric doesn't fit all — e.g., coding questions vs. creative writing vs. factual recall.

---

### 9.5 Summarization Score

> *"How good is this summary? Balancing information coverage with conciseness."*

**Score Range**: 0.0 to 1.0

**Formula**:

$$\text{Summarization Score} = c \cdot QA_{\text{score}} + (1 - c) \cdot \text{Conciseness}$$

where $c$ = coefficient (default: 0.5).

**Sub-scores**:

$$QA_{\text{score}} = \frac{|\text{questions answered correctly from summary}|}{|\text{total questions generated from context}|}$$

$$\text{Conciseness} = 1 - \frac{\min(|\text{summary}|,\ |\text{context}|)}{|\text{context}|}$$

**Algorithm**:

```
1. Extract key phrases from the source context
2. Generate questions from those key phrases
3. Try to answer each question using ONLY the summary
4. QA Score = fraction of questions answerable from summary
5. Conciseness = how much shorter is the summary vs. source
6. Final = weighted combination
```

**Required Columns**: `response` (the summary), `reference_contexts` (source text)

**Type**: LLM-based (keyphrase extraction + QA generation)

---

### 9.6 SQL Metrics

#### DataCompy Score (Execution-Based)

> *"Run both SQL queries and compare the result tables."*

**Score Range**: 0.0 to 1.0

**Modes**: `rows` (row-wise comparison) or `columns`

**Formulas (Row mode)**:

$$\text{Precision} = \frac{|\text{matching rows}|}{|\text{response rows}|} \qquad \text{Recall} = \frac{|\text{matching rows}|}{|\text{reference rows}|}$$

**Required Columns**: `response` (CSV output), `reference` (CSV output)

**Type**: Non-LLM (execution + DataFrame comparison using `datacompy` library)

#### LLM SQL Equivalence

> *"Are these two SQL queries logically equivalent?"*

**Score**: Binary (0 or 1)

**Algorithm**: LLM analyzes both SQL queries + the database schema and determines if they would produce identical results for all possible inputs.

**Required Columns**: `response`, `reference`, `reference_contexts` (database schema)

**Type**: LLM-based

---

### 9.7 NVIDIA Metrics

Three metrics following NVIDIA's evaluation methodology:

#### Answer Accuracy

> *"Does the response match the reference ground truth?"*

**Score Range**: 0.0 to 1.0

**Algorithm**: Uses **dual independent LLM judges** for robustness. Each judge rates on a 3-point scale (0 = no match, 2 = partial, 4 = exact match). Final score = average of both judges, normalized to [0, 1].

**Required Columns**: `user_input`, `response`, `reference`

#### Context Relevance

> *"Are the retrieved contexts relevant to the query?"*

**Required Columns**: `user_input`, `retrieved_contexts`

#### Response Groundedness

> *"Is the response grounded in the provided context?"*

**Required Columns**: `response`, `retrieved_contexts`

All NVIDIA metrics are LLM-based and follow NVIDIA's evaluation prompt templates.

---

## 10. Test Data Generation

### The Problem

Manually creating evaluation datasets is expensive, slow, and biased. RAGAS automates this via **knowledge graph–based test set generation**.

### Characteristics of an Ideal Test Dataset

1. **High quality** data samples
2. **Wide variety** of scenarios reflecting real-world usage
3. **Sufficient size** for statistically significant conclusions
4. **Continually updated** to prevent data drift

### Query Types Generated

```
                    ┌──────────────┐
                    │  Query Types  │
                    └──────┬───────┘
               ┌──────────┴──────────┐
               ▼                      ▼
        ┌─────────────┐        ┌─────────────┐
        │  Single-Hop  │        │  Multi-Hop   │
        └──────┬──────┘        └──────┬──────┘
          ┌────┴────┐            ┌────┴────┐
          ▼         ▼            ▼         ▼
      Specific  Abstract     Specific  Abstract
```

- **Single-Hop Specific**: "What year was the theory of relativity published?"
- **Single-Hop Abstract**: "How did Einstein's theory change our understanding of time?"
- **Multi-Hop Specific**: "Who influenced Einstein, and what theory did they propose?"
- **Multi-Hop Abstract**: "How have theories on relativity evolved since Einstein?"

### Knowledge Graph Pipeline

```
Documents → Splitter → Nodes → Extractors → Enriched Nodes → Relationship Builders → Knowledge Graph
```

**Components**:

| Component | Purpose | Types |
|---|---|---|
| **Document Splitter** | Chunks documents hierarchically | Custom or built-in |
| **Extractors** | Extract entities, summaries, themes from nodes | `LLMBasedExtractor`, `Extractor` (rule-based) |
| **Relationship Builders** | Link related nodes | Jaccard Similarity, Cosine Similarity |
| **Transforms** | Orchestrate pipeline in sequence | Handles parallel processing |

### Generation Workflow

```python
from ragas.testset import TestsetGenerator

# 1. Create generator
generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

# 2. Build knowledge graph from documents
generator.generate_with_langchain_docs(documents, testset_size=10)

# 3. Or granular control:
#    a) Build KG
kg = generator.knowledge_graph
kg.save("knowledge_graph.json")

#    b) Define query distribution
distributions = {
    single_hop_specific: 0.5,
    multi_hop_abstract: 0.3,
    single_hop_abstract: 0.2,
}

#    c) Generate
testset = generator.generate(distributions=distributions, testset_size=20)
testset.to_pandas()
```

### Scenario Composition

Each test scenario combines:

| Dimension | Options |
|---|---|
| **Nodes** | Which knowledge graph nodes to use |
| **Query Length** | Short, medium, long |
| **Query Style** | Web search, chat, formal |
| **Persona** | Senior engineer, junior engineer, etc. |

---

## 11. The Experiment Framework

### Philosophy

An **experiment** is a deliberate, measurable change to your application:

```
Make Change → Run Evaluations → Observe Results → Hypothesize Next Change → Repeat
```

### Principles of Good Experiments

1. **Define measurable metrics** before running
2. **Store results systematically** for comparison
3. **Isolate changes** — one variable at a time
4. **Iterate** — use results to inform next experiment

### The `@experiment()` Decorator

```python
from ragas import Dataset, experiment
from ragas.metrics import discrete_metric, MetricResult

@discrete_metric(name="accuracy", allowed_values=["pass", "fail"])
def accuracy_score(response: str, expected: str):
    result = "pass" if expected.strip().lower() == response.strip().lower() else "fail"
    return MetricResult(value=result, reason=f"Match: {result == 'pass'}")

@experiment()
async def run_experiment(row):
    response = my_app(query=row["query"])
    score = accuracy_score.score(response=response, expected=row["expected_output"])
    return {**row, "response": response, "accuracy": score.value}

# Execute
import asyncio
dataset = Dataset(name="test_data", backend="local/csv", root_dir=".")
results = asyncio.run(run_experiment.arun(dataset, name="v1_baseline"))
```

### Experiment Storage

Results auto-save to CSV:

```
evals/
├── datasets/
│   └── test_data.csv           # Input dataset
└── experiments/
    └── v1_baseline.csv          # Timestamped results
```

### `ExperimentWrapper` API

| Method | Purpose |
|---|---|
| `__call__(row)` | Call original function |
| `arun(dataset, name)` | Run experiment across entire dataset |
| Returns `Experiment` object | Access results, metadata |

---

## 12. Custom Metrics — Building Your Own

### Discrete Metric (Categorical Output)

```python
from ragas.metrics import discrete_metric, MetricResult

@discrete_metric(name="tone_check", allowed_values=["professional", "casual", "rude"])
def tone_check(response: str):
    if any(word in response.lower() for word in ["please", "thank you", "appreciate"]):
        return MetricResult(value="professional", reason="Contains polite language")
    return MetricResult(value="casual", reason="Neutral tone")
```

### Numeric Metric (Continuous Output)

```python
from ragas.metrics import numeric_metric, MetricResult

@numeric_metric(name="length_score", allowed_values=(0, 1))
def length_score(response: str):
    ideal_length = 200
    actual = len(response.split())
    score = max(0, 1 - abs(actual - ideal_length) / ideal_length)
    return MetricResult(value=score, reason=f"Word count: {actual}")
```

### Subclassing for Advanced Metrics

```python
from dataclasses import dataclass
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric, MetricType

@dataclass
class MyCustomMetric(MetricWithLLM, SingleTurnMetric):
    name: str = "my_metric"
    _required_columns = {
        MetricType.SINGLE_TURN: {"user_input", "response"}
    }

    async def _single_turn_ascore(self, sample, callbacks) -> float:
        # Your custom LLM-based evaluation logic
        prompt_result = await self.llm.generate(...)
        return computed_score
```

---

## 13. Prompt Engineering in RAGAS

### Prompt Object Structure

Every RAGAS prompt is a **`PydanticPrompt`** with four components:

```python
class MyPrompt(PydanticPrompt):
    instruction = "Evaluate whether the response is faithful to the context."
    input_model = FaithfulnessInput      # Pydantic model
    output_model = FaithfulnessOutput    # Pydantic model
    examples = [                          # Few-shot examples (3-5 ideal)
        (FaithfulnessInput(...), FaithfulnessOutput(...)),
    ]
```

### Modifying Built-in Prompts

```python
from ragas.metrics import Faithfulness

scorer = Faithfulness(llm=my_llm)

# Access the prompt
print(scorer.nli_statements_prompt.instruction)

# Override with custom prompt
scorer.nli_statements_prompt.instruction = "Be very strict. Only mark supported if directly stated."
scorer.nli_statements_prompt.examples = [...]  # domain-specific examples
```

### Language Adaptation

```python
# Translate few-shot examples to Hindi
scorer.statement_generator_prompt = await scorer.statement_generator_prompt.adapt(
    target_language="hindi",
    llm=translation_llm,
)
```

### Prompt Optimization Workflow

```python
# 1. Create metric with custom prompt
# 2. Run experiment
# 3. Measure performance
# 4. Modify prompt
# 5. Re-run experiment
# 6. Compare results in experiments/ CSV files
```

---

## 14. Integrations Ecosystem

### LLM Frameworks

| Framework | Support Level |
|---|---|
| **LangChain** | First-class (built-in wrappers) |
| **LlamaIndex** | First-class (RAG + Agents) |
| **Haystack** | Supported |
| **Griptape** | Supported |
| **LlamaStack** | Supported |
| **Swarm** | Supported |

### LLM Providers

| Provider | Setup |
|---|---|
| **OpenAI** | `export OPENAI_API_KEY=...` (default) |
| **Anthropic Claude** | `export ANTHROPIC_API_KEY=...` |
| **Google Gemini** | `export GOOGLE_API_KEY=...` |
| **Amazon Bedrock** | AWS credentials |
| **OCI Gen AI** | OCI config |
| **Ollama (local)** | Custom `base_url` |
| **Any OpenAI-compatible** | Custom `base_url` + `api_key` |

### Observability / Tracing

| Tool | Purpose |
|---|---|
| **Arize Phoenix** | LLM observability and tracing |
| **LangSmith** | LangChain tracing and evaluation |

### Data Backends

| Backend | Entry Point |
|---|---|
| Local CSV | `local/csv` |
| Local JSONL | `local/jsonl` |
| In-Memory | `inmemory` |
| Google Drive | `gdrive` (experimental) |

---

## 15. Feedback Loops & Continuous Improvement

### The Iteration Cycle

```
       ┌──────────────────────────────────┐
       │        Production System          │
       └──────────────┬───────────────────┘
                      │ User feedback (thumbs up/down, corrections)
                      ▼
       ┌──────────────────────────────────┐
       │      Feedback Collection          │
       │   (noisy signals from users)      │
       └──────────────┬───────────────────┘
                      │ Extract patterns
                      ▼
       ┌──────────────────────────────────┐
       │       Analysis & Diagnosis        │
       │  Which component is failing?      │
       └──────────────┬───────────────────┘
                      │ Targeted fix
                      ▼
       ┌──────────────────────────────────┐
       │     Experiment & Evaluate         │
       │   Measure impact of changes       │
       └──────────────┬───────────────────┘
                      │ Deploy improved version
                      ▼
```

### Key Practices

1. **Treat user feedback as noisy signals** — aggregate before acting
2. **Extract patterns** — recurring failure modes reveal systemic issues
3. **Detect specific pipeline issues** — use component-level metrics to localize problems
4. **Prevent recurring errors** — add failing cases to evaluation dataset
5. **Continuous monitoring** — track metric trends over time in production

---

## 16. Code Walkthrough — Key Source Modules

### `src/ragas/evaluation.py` — The `evaluate()` Function

**Signature**:
```python
def evaluate(
    dataset: Union[Dataset, EvaluationDataset],
    metrics: Optional[List[Metric]] = None,
    llm: Optional[Union[BaseRagasLLM, LangchainLLM]] = None,
    embeddings: Optional[Union[BaseRagasEmbeddings, LangchainEmbeddings]] = None,
    experiment_name: Optional[str] = None,
    callbacks: Optional[Callbacks] = None,
    run_config: Optional[RunConfig] = None,
    raise_exceptions: bool = False,
    column_map: Optional[Dict[str, str]] = None,
    show_progress: bool = True,
    batch_size: Optional[int] = None,
) -> EvaluationResult:
```

**Internal flow**:
1. Validate dataset columns against metric requirements
2. Initialize metrics (set LLM, embeddings, run_config)
3. Create `Executor` for async parallel execution
4. Submit all (sample × metric) scoring jobs
5. Collect results → build `EvaluationResult`

### `src/ragas/executor.py` — Async Execution Engine

```python
class Executor:
    desc: str
    show_progress: bool
    raise_exceptions: bool
    batch_size: Optional[int]
    run_config: RunConfig

    def submit(self, callable, *args, **kwargs)  # Queue a job
    def results(self) -> List                     # Run all (sync)
    async def aresults(self) -> List              # Run all (async)
    def cancel(self)                              # Cancel all jobs
```

**Design**: Uses `asyncio` with configurable batching and progress bars (via `tqdm`). Handles error collection vs. raising.

### `src/ragas/dataset_schema.py` — Data Models

Key class hierarchy:
```
BaseSample
├── SingleTurnSample
└── MultiTurnSample

RagasDataset (Abstract)
└── EvaluationDataset
```

### `src/ragas/metrics/base.py` — Metric Base Classes

```
Metric (ABC)
├── MetricWithLLM (+ PromptMixin)
├── MetricWithEmbeddings
├── SingleTurnMetric
└── MultiTurnMetric
```

### `src/ragas/messages.py` — Message Types

```
HumanMessage(content: str)
AIMessage(content: str, tool_calls: List[ToolCall])
ToolMessage(content: str)
ToolCall(name: str, args: dict)
```

---

## 17. Hands-On: End-to-End RAG Evaluation

### Complete Working Example

```python
from langchain_openai import ChatOpenAI
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, LLMContextRecall, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
import openai

# 1. Setup evaluator LLM
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

# 2. Prepare evaluation data
samples = [
    SingleTurnSample(
        user_input="What is photosynthesis?",
        response="Photosynthesis is how plants make food from sunlight, water, and CO2.",
        reference="Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose.",
        retrieved_contexts=[
            "Photosynthesis occurs in chloroplasts. Plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
            "The process has two stages: light-dependent reactions and the Calvin cycle.",
        ],
    ),
    # ... more samples
]
dataset = EvaluationDataset(samples=samples)

# 3. Define metrics
metrics = [
    Faithfulness(),          # Is response grounded in context?
    LLMContextRecall(),      # Did retriever find relevant info?
    FactualCorrectness(),    # Is response factually correct?
]

# 4. Run evaluation
result = evaluate(
    dataset=dataset,
    metrics=metrics,
    llm=evaluator_llm,
    show_progress=True,
)

# 5. Analyze results
print(result)                     # Overall scores
df = result.to_pandas()           # Per-sample breakdown
print(df[["faithfulness", "context_recall", "factual_correctness"]])
```

---

## 18. Hands-On: Agent Evaluation

### Evaluating a Multi-Turn Agent

```python
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
from ragas.metrics import AgentGoalAccuracy, ToolCallAccuracy

samples = [
    MultiTurnSample(
        user_input=[
            HumanMessage(content="What's 25 * 48?"),
            AIMessage(content="Let me calculate that.", tool_calls=[
                ToolCall(name="multiply", args={"a": 25, "b": 48})
            ]),
            ToolMessage(content="1200"),
            AIMessage(content="25 × 48 = 1,200"),
        ],
        reference="1200",
        reference_tool_calls=[
            ToolCall(name="multiply", args={"a": 25, "b": 48})
        ],
    ),
]

dataset = EvaluationDataset(samples=samples)
result = evaluate(
    dataset=dataset,
    metrics=[AgentGoalAccuracy(), ToolCallAccuracy()],
    llm=evaluator_llm,
)
```

---

## 19. Interview Questions & Answers

### Fundamentals

**Q1: What is RAGAS and why is it needed?**

RAGAS is an open-source evaluation framework for LLM applications, especially RAG pipelines. It provides objective, reproducible metrics to assess retrieval quality, generation faithfulness, and overall response quality. It's needed because LLM outputs are non-deterministic and can't be evaluated with simple unit tests — you need multi-dimensional, nuanced assessment.

---

**Q2: What are the four core RAG metrics in RAGAS?**

1. **Faithfulness** — Is the response factually consistent with the retrieved context? (measures hallucination)
2. **Context Precision** — Are relevant chunks ranked higher than irrelevant ones? (measures retrieval ranking)
3. **Context Recall** — Did the retriever find all relevant information? (measures retrieval completeness)
4. **Answer Relevancy** — Does the response actually answer the user's question? (measures response quality)

---

**Q3: How does the Faithfulness metric work internally?**

It uses a 2-step LLM pipeline:
1. **Statement Extraction**: Decompose the response into atomic factual claims
2. **NLI Verification**: For each claim, use LLM to check if it's supported by the retrieved context

Score = (supported claims) / (total claims). A score of 1.0 means zero hallucination.

---

**Q4: Explain the difference between Context Precision and Context Recall.**

- **Precision** = "Of the chunks I retrieved, how many are actually relevant?" — focuses on ranking quality and avoiding noise
- **Recall** = "Of all relevant information that exists, how much did I retrieve?" — focuses on completeness

An analogy: Precision asks "is my retriever precise?"; Recall asks "did my retriever miss anything important?"

---

**Q5: How does Answer Relevancy use a reverse-engineering approach?**

Instead of directly comparing the response to the question, it:
1. Generates N questions that the response *could* answer
2. Embeds these generated questions and the original question
3. Computes cosine similarity between them

If the response truly answers the original question, the reverse-engineered questions should be semantically similar to the original. This approach is robust to paraphrasing and doesn't require a reference answer.

---

### Architecture & Design

**Q6: What is the difference between LLM-based and non-LLM-based metrics?**

| Aspect | LLM-Based | Non-LLM |
|---|---|---|
| Base class | `MetricWithLLM` | `Metric` |
| Deterministic | No | Yes |
| Human correlation | Higher | Lower |
| Cost | Higher (LLM API calls) | Free |
| Speed | Slower | Faster |
| Examples | Faithfulness, Context Recall | BLEU, ROUGE, Exact Match |

---

**Q7: How does RAGAS handle async execution and parallelism?**

RAGAS uses an `Executor` class that:
- Queues all (sample × metric) pairs as async jobs
- Runs them in parallel with configurable `batch_size`
- Shows progress via `tqdm` progress bars
- Handles errors gracefully (log vs. raise based on `raise_exceptions`)
- Uses `asyncio` with `nest_asyncio` for nested event loop compatibility

---

**Q8: What is the `PydanticPrompt` pattern and why does RAGAS use it?**

Each prompt in RAGAS is a structured object with:
1. **Instruction**: Natural language task description
2. **Input Model**: Pydantic model defining input schema
3. **Output Model**: Pydantic model defining output schema
4. **Few-shot Examples**: Input/output pairs for in-context learning

This pattern enables: type-safe I/O, easy prompt modification, language adaptation, and consistent prompt versioning across metric implementations.

---

**Q9: How does RAGAS generate test data using knowledge graphs?**

1. **Document Splitting**: Chunk documents hierarchically
2. **Entity Extraction**: Extract named entities and key information from each chunk
3. **Relationship Building**: Link related chunks using Jaccard similarity or cosine similarity
4. **Knowledge Graph**: Build a graph where nodes = chunks, edges = relationships
5. **Scenario Generation**: Combine nodes + query type + length + style to create diverse scenarios
6. **Query Synthesis**: Use LLM to generate realistic questions and reference answers from scenarios

This produces diverse, representative test sets covering single-hop, multi-hop, specific, and abstract queries.

---

**Q10: How would you evaluate a multi-turn agent vs. a single-turn RAG system?**

| Aspect | Single-Turn RAG | Multi-Turn Agent |
|---|---|---|
| Sample type | `SingleTurnSample` | `MultiTurnSample` |
| Metric base | `SingleTurnMetric` | `MultiTurnMetric` |
| Key metrics | Faithfulness, Precision, Recall | Tool Call Accuracy, Goal Accuracy |
| Input format | Single query + response | Conversation history with tool calls |
| Evaluation focus | Retrieval + generation quality | Tool usage + goal completion |

---

### Practical & Advanced

**Q11: How would you customize a RAGAS metric for a domain-specific use case?**

Three approaches:
1. **Modify prompts**: Change instruction and few-shot examples on existing metrics to match domain terminology
2. **Use Aspect Critic**: Create a custom single-aspect evaluator with your own definition
3. **Subclass**: Create a new metric class inheriting from `MetricWithLLM` + `SingleTurnMetric`, implementing `_single_turn_ascore()`

---

**Q12: What are the tradeoffs between using `evaluate()` vs. the `@experiment()` decorator?**

- **`evaluate()`**: One-shot evaluation of a dataset against metrics. Returns `EvaluationResult`. Best for: quick assessments, CI/CD pipelines.
- **`@experiment()`**: Wraps your application logic, runs it across a dataset, and auto-saves results. Best for: iterative development, A/B testing, tracking changes over time.

---

**Q13: How would you set up continuous RAG evaluation in production?**

1. **Collect production data**: Log queries, responses, and retrieved contexts
2. **Sample representative cases**: Don't evaluate everything — sample strategically
3. **Run periodic evaluations**: Use `evaluate()` with Faithfulness + Answer Relevancy (no reference needed)
4. **Track trends**: Compare metric scores over time from experiments/ CSV files
5. **Alert on degradation**: Set thresholds (e.g., faithfulness < 0.8 triggers review)
6. **Feed back failures**: Add flagged cases to evaluation dataset for regression testing

---

**Q14: What is the `RunConfig` and when would you customize it?**

`RunConfig` controls execution parameters:
- Timeout settings for LLM calls
- Retry logic for transient failures
- Max concurrent workers
- Seed for reproducibility

Customize when: dealing with rate limits, slow LLM endpoints, or needing deterministic results for benchmarking.

---

**Q15: How does RAGAS handle the non-determinism of LLM-based metrics?**

1. **Multiple evaluations**: Some metrics (like Answer Relevancy) generate N samples and average
2. **Structured outputs**: Uses Pydantic models to constrain LLM output format
3. **Few-shot examples**: Provides in-context examples to guide consistent behavior
4. **NLI decomposition**: Breaks evaluation into binary (supported/not supported) sub-tasks that are more reliable than holistic scoring

---

**Q16: Compare RAGAS to other evaluation frameworks (e.g., DeepEval, TruLens).**

| Feature | RAGAS | DeepEval | TruLens |
|---|---|---|---|
| Focus | RAG evaluation | General LLM eval | RAG + tracing |
| Test generation | ✓ (KG-based) | ✓ | ✗ |
| Metric transparency | Open (see prompt code) | Open | Partially open |
| Async execution | ✓ | ✓ | ✓ |
| Framework integrations | LangChain, LlamaIndex, Haystack | LangChain | LangChain, LlamaIndex |
| Experimentation | ✓ (`@experiment`) | ✓ | ✓ (dashboard) |
| Unique strength | Knowledge graph test gen | Conversational metrics | Feedback functions |

---

**Q17: What is the Faithfulness with HHEM variant and when would you use it?**

`FaithfulnesswithHHEM` replaces the LLM-based NLI step with Vectara's T5-based Hallucination Evaluation Model. Use it when:
- You need **deterministic** faithfulness scoring
- You want to **reduce costs** (no LLM API calls for NLI)
- You need **faster** evaluation at scale
- Trade-off: May be less accurate than GPT-4 based NLI for complex claims

---

**Q18: Walk through what happens when you call `evaluate()` with 100 samples and 3 metrics.**

1. **Validation**: Check that dataset columns satisfy each metric's `_required_columns`
2. **Initialization**: Set LLM and embeddings on each `MetricWithLLM`/`MetricWithEmbeddings` instance
3. **Job creation**: 100 samples × 3 metrics = 300 async scoring jobs submitted to `Executor`
4. **Parallel execution**: Jobs run in batches (configurable `batch_size`), with progress bar
5. **Result collection**: 300 scores collected into a matrix (100 rows × 3 metric columns)
6. **Aggregation**: Compute mean score per metric across all samples
7. **Return**: `EvaluationResult` with aggregate scores + per-sample DataFrame

---

## 20. Quick Reference Cheat Sheet

### Essential Imports

```python
# Data
from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample, MultiTurnSample
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall

# Evaluation
from ragas import evaluate, aevaluate

# Experiments
from ragas import experiment, Dataset

# Metrics — RAG
from ragas.metrics import Faithfulness, FaithfulnesswithHHEM
from ragas.metrics import LLMContextRecall, NonLLMContextRecall, IDBasedContextRecall
from ragas.metrics import LLMContextPrecisionWithReference, LLMContextPrecisionWithoutReference
from ragas.metrics import ResponseRelevancy, ContextEntityRecall, NoiseSensitivity
from ragas.metrics import MultiModalFaithfulness, MultiModalRelevance

# Metrics — NL Comparison
from ragas.metrics import FactualCorrectness, AnswerCorrectness
from ragas.metrics import SemanticSimilarity, NonLLMStringSimilarity
from ragas.metrics import BleuScore, RougeScore, ChrfScore, ExactMatch, StringPresence

# Metrics — Agent
from ragas.metrics import ToolCallAccuracy, ToolCallF1, AgentGoalAccuracy, TopicAdherenceScore

# Metrics — General Purpose
from ragas.metrics import AspectCritic, SimpleCriteriaScore, RubricsScore, InstanceRubrics
from ragas.metrics import SummarizationScore

# Metrics — SQL
from ragas.metrics import DataCompyScore, LLMSQLEquivalence

# Metrics — NVIDIA
from ragas.metrics import AnswerAccuracy, ContextRelevance, ResponseGroundedness

# Metrics — Custom
from ragas.metrics import discrete_metric, numeric_metric, ranking_metric, MetricResult

# Test Generation
from ragas.testset import TestsetGenerator
```

### Metric → Required Columns Quick Map

| Metric | `user_input` | `response` | `retrieved_contexts` | `reference` | `reference_contexts` | Other |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Faithfulness | ✓ | ✓ | ✓ | | | |
| Context Precision (w/ ref) | ✓ | | ✓ | ✓ | | |
| Context Precision (w/o ref) | ✓ | ✓ | ✓ | | | |
| Context Recall | ✓ | | ✓ | ✓ | | |
| Context Entities Recall | | | ✓ | ✓ | | |
| Noise Sensitivity | ✓ | ✓ | ✓ | ✓ | | |
| Answer Relevancy | ✓ | ✓ | ○ | | | |
| Factual Correctness | | ✓ | | ✓ | | |
| Answer Correctness | ✓ | ✓ | | ✓ | | |
| Semantic Similarity | | ✓ | | ✓ | | |
| Non-LLM String Similarity | | ✓ | | ✓ | | |
| BLEU / ROUGE / CHRF | | ✓ | | ✓ | | |
| Exact Match / String Presence | | ✓ | | ✓ | | |
| Summarization Score | | ✓ | | | ✓ | |
| Aspect Critic | ✓ | ✓ | ○ | ○ | | |
| Rubrics Score | ✓ | ✓ | ○ | ○ | | |
| Instance Rubrics | ✓ | ✓ | | | | `rubrics` |
| Tool Call Accuracy | | | | | | `user_input` (multi), `ref_tool_calls` |
| Tool Call F1 | | | | | | `user_input` (multi), `ref_tool_calls` |
| Agent Goal Accuracy | | | | | | `user_input` (multi), `reference` (opt) |
| Topic Adherence | | | | | | `user_input` (multi), `ref_topics` |
| DataCompy Score | | ✓ | | ✓ | | |
| LLM SQL Equivalence | | ✓ | | ✓ | ✓ | |
| NVIDIA Answer Accuracy | ✓ | ✓ | | ✓ | | |
| NVIDIA Context Relevance | ✓ | | ✓ | | | |
| NVIDIA Response Groundedness | | ✓ | ✓ | | | |
| Multimodal Faithfulness | | ✓ | ✓ | | | |
| Multimodal Relevance | ✓ | ✓ | ✓ | | | |

✓ = required, ○ = optional

### Key Formulas

$$\text{Faithfulness} = \frac{\text{Supported Claims}}{\text{Total Claims}}$$

$$\text{Context Precision@K} = \frac{\sum_{k=1}^{K} \text{Precision@k} \cdot v_k}{\sum_{k=1}^{K} v_k}$$

$$\text{Context Recall} = \frac{\text{Attributed Reference Claims}}{\text{Total Reference Claims}}$$

$$\text{Answer Relevancy} = \frac{1}{N}\sum_{i=1}^{N} \cos(E_{g_i}, E_o)$$

$$\text{Factual Correctness (F1)} = \frac{2 \cdot P \cdot R}{P + R} \quad \text{where } P = \frac{TP}{TP+FP},\ R = \frac{TP}{TP+FN}$$

$$\text{Answer Correctness} = w \cdot F1_{\text{factual}} + (1-w) \cdot \text{SemanticSim}$$

$$\text{Summarization} = c \cdot QA_{\text{score}} + (1-c) \cdot \text{Conciseness}$$

$$\text{Context Entity Recall} = \frac{|CE_{\text{retrieved}} \cap CE_{\text{reference}}|}{|CE_{\text{reference}}|}$$

$$\text{Noise Sensitivity} = \frac{\text{Incorrect Claims}}{\text{Total Claims}} \quad \text{(lower is better)}$$

$$\text{Tool Call F1} = \frac{2 \cdot P_{\text{tools}} \cdot R_{\text{tools}}}{P_{\text{tools}} + R_{\text{tools}}}$$

### Decision Tree: Which Metric Do I Need?

```
Is the problem with RETRIEVAL or GENERATION?

RETRIEVAL:
  ├── Are relevant docs being found?        → Context Recall
  ├── Are key entities being captured?       → Context Entities Recall
  ├── Are irrelevant docs sneaking in?       → Context Precision
  ├── Is ranking order wrong?                → Context Precision
  ├── Is noise hurting answers?              → Noise Sensitivity
  └── Is context even relevant? (NVIDIA)     → Context Relevance

GENERATION:
  ├── Is the answer hallucinating?           → Faithfulness / Response Groundedness
  ├── Is the answer off-topic?               → Answer Relevancy
  ├── Is the answer factually wrong?         → Factual Correctness
  ├── Does it match ground truth?            → Answer Correctness / Answer Accuracy
  ├── Is the meaning similar?                → Semantic Similarity
  └── Is the summary good?                   → Summarization Score

AGENT:
  ├── Are correct tools being called?        → Tool Call Accuracy / F1
  ├── Is the goal being achieved?            → Agent Goal Accuracy
  └── Is the agent staying on topic?         → Topic Adherence

SQL:
  ├── Do query results match?                → DataCompy Score
  └── Are queries logically equivalent?      → LLM SQL Equivalence

MULTIMODAL:
  ├── Faithful to images + text?             → Multimodal Faithfulness
  └── Relevant to images + text?             → Multimodal Relevance

CUSTOM:
  ├── Binary pass/fail check                 → Aspect Critic
  ├── Numeric scale (same rubric)            → Rubrics Score / Simple Criteria
  └── Per-sample rubric                      → Instance-Specific Rubrics

TRADITIONAL (no LLM needed):
  ├── N-gram precision                       → BLEU Score
  ├── N-gram recall                          → ROUGE Score
  ├── Character-level (multilingual)         → CHRF Score
  ├── Edit distance                          → Non-LLM String Similarity
  ├── Exact string match                     → Exact Match
  └── Substring presence                     → String Presence
```

---

*Built from the [RAGAS repository](https://github.com/explodinggradients/ragas) — Apache 2.0 License*
