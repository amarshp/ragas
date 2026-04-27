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

Extracts named entities from both reference and retrieved contexts, then computes entity-level recall.

### 6.6 Noise Sensitivity

> *"How much does irrelevant context degrade the response?"*

Measures the system's robustness when noisy/irrelevant chunks are mixed into retrieved contexts.

---

## 7. Deep Dive: Agent & Tool Use Metrics

### 7.1 Tool Call Accuracy

Compares the tools actually called by the agent against reference tool calls — checks both tool name and arguments.

### 7.2 Tool Call F1

Computes F1 score between predicted and reference tool calls, balancing precision and recall of tool usage.

### 7.3 Agent Goal Accuracy

Evaluates whether a multi-turn agent conversation ultimately achieved the user's intended goal.

### 7.4 Topic Adherence

Measures whether the agent stays on-topic throughout a multi-turn conversation without drifting.

---

## 8. Deep Dive: NLP Comparison Metrics

### Traditional (Non-LLM) Metrics

| Metric | What It Measures | Deterministic |
|---|---|---|
| **BLEU Score** | N-gram precision (machine translation heritage) | ✓ |
| **ROUGE Score** | N-gram recall (summarization heritage) | ✓ |
| **CHRF Score** | Character-level F-score | ✓ |
| **Exact Match** | Binary: response == reference | ✓ |
| **String Presence** | Whether specific strings appear in response | ✓ |

### LLM-Based Comparison Metrics

| Metric | What It Measures | Deterministic |
|---|---|---|
| **Factual Correctness** | Whether response facts match reference facts (claim-level) | ✗ |
| **Semantic Similarity** | Embedding-based meaning comparison | ✓ |
| **Non-LLM String Similarity** | Edit distance / Levenshtein | ✓ |

---

## 9. Deep Dive: General-Purpose Metrics

### 9.1 Aspect Critic

A **single-aspect binary judge** — evaluates whether a response meets a specific criterion.

```python
from ragas.metrics import AspectCritic

politeness = AspectCritic(
    name="politeness",
    definition="Is the response polite and professional?",
    llm=evaluator_llm,
)
```

### 9.2 Simple Criteria Scoring

Like Aspect Critic but returns a **numeric score** instead of binary.

### 9.3 Rubrics-Based Scoring

Evaluates against a **multi-level rubric** (e.g., 1-5 scale with descriptions for each level).

### 9.4 Instance-Specific Rubrics

Each sample carries its own rubric in the `rubric` field — useful when different queries need different scoring criteria.

### 9.5 SQL Metrics

| Metric | Method |
|---|---|
| **Execution-Based Datacompy Score** | Execute both queries, compare result DataFrames |
| **SQL Query Semantic Equivalence** | LLM judges whether two SQL queries are logically equivalent |

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
from ragas.metrics import Faithfulness, LLMContextRecall, LLMContextPrecisionWithReference
from ragas.metrics import ResponseRelevancy, FactualCorrectness

# Metrics — Agent
from ragas.metrics import ToolCallAccuracy, AgentGoalAccuracy

# Metrics — General
from ragas.metrics import AspectCritic, SemanticSimilarity

# Metrics — Custom
from ragas.metrics import discrete_metric, numeric_metric, MetricResult

# Test Generation
from ragas.testset import TestsetGenerator
```

### Metric → Required Columns Quick Map

| Metric | `user_input` | `response` | `retrieved_contexts` | `reference` | `reference_contexts` |
|---|:---:|:---:|:---:|:---:|:---:|
| Faithfulness | ✓ | ✓ | ✓ | | |
| Context Precision (w/ ref) | ✓ | | ✓ | ✓ | |
| Context Recall | ✓ | | ✓ | ✓ | |
| Answer Relevancy | ✓ | ✓ | ○ | | |
| Factual Correctness | | ✓ | | ✓ | |
| Semantic Similarity | | ✓ | | ✓ | |

✓ = required, ○ = optional

### Key Formulas

$$\text{Faithfulness} = \frac{\text{Supported Claims}}{\text{Total Claims}}$$

$$\text{Context Precision@K} = \frac{\sum_{k=1}^{K} \text{Precision@k} \cdot v_k}{\sum_{k=1}^{K} v_k}$$

$$\text{Context Recall} = \frac{\text{Attributed Reference Claims}}{\text{Total Reference Claims}}$$

$$\text{Answer Relevancy} = \frac{1}{N}\sum_{i=1}^{N} \cos(E_{g_i}, E_o)$$

### Decision Tree: Which Metric Do I Need?

```
Is the problem with RETRIEVAL or GENERATION?

RETRIEVAL:
  ├── Are relevant docs being found?        → Context Recall
  ├── Are irrelevant docs sneaking in?       → Context Precision
  └── Is ranking order wrong?                → Context Precision

GENERATION:
  ├── Is the answer hallucinating?           → Faithfulness
  ├── Is the answer off-topic?               → Answer Relevancy
  ├── Is the answer factually wrong?         → Factual Correctness
  └── Is the answer semantically different?  → Semantic Similarity

AGENT:
  ├── Are correct tools being called?        → Tool Call Accuracy / F1
  ├── Is the goal being achieved?            → Agent Goal Accuracy
  └── Is the agent staying on topic?         → Topic Adherence

CUSTOM:
  └── Define your own                        → Aspect Critic / Rubrics
```

---

*Built from the [RAGAS repository](https://github.com/explodinggradients/ragas) — Apache 2.0 License*
