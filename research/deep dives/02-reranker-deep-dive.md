# Reranker Deep Dive: The Single Biggest Lever in RAG Optimization

> **TL;DR:** Adding a cross-encoder reranker gave us +5.7% precision improvement — the largest gain from any single change across 20+ experiments. This document explains what rerankers are, how we implemented one, and why it worked so well.

---

📖 **Part of [RAG Observatory](../../README.md)** | Previous: [← Metrics vs Quality Gap](01-metrics-vs-quality-gap.md)

---

## Table of Contents

- [The Problem with Bi-Encoders](#the-problem-with-bi-encoders)
- [How Rerankers Work](#how-rerankers-work)
- [Our Implementation](#our-implementation)
- [Results](#results)
- [Why It Works](#why-it-works)
- [Configuration Guide](#configuration-guide)
- [Trade-offs and Considerations](#trade-offs-and-considerations)
- [Recommendations](#recommendations)

---

## The Problem with Bi-Encoders

Standard RAG retrieval uses **bi-encoders** (like MPNet, BGE, OpenAI embeddings):

```
Query: "What's the return deadline for electronics?"
         │
         ▼
    ┌─────────┐
    │ Encoder │ ──► Query Vector [0.12, -0.45, 0.78, ...]
    └─────────┘
                            │
                            ▼
                     Cosine Similarity
                            │
                            ▼
    ┌─────────────────────────────────────────┐
    │ Document Vectors (pre-computed)         │
    │ Doc1: [0.11, -0.42, 0.80, ...]  → 0.94  │
    │ Doc2: [0.08, -0.39, 0.75, ...]  → 0.89  │
    │ Doc3: [0.15, -0.48, 0.72, ...]  → 0.87  │
    └─────────────────────────────────────────┘
```

**The limitation:** Query and documents are encoded *independently*. The model can't see how query terms relate to specific parts of each document.

This causes the **"meleset sedikit"** problem we documented — retrieving the right document but wrong subsection, because the overall document embedding is similar even if the specific content doesn't match.

---

## How Rerankers Work

**Cross-encoder rerankers** process query and document *together*:

```
    ┌─────────────────────────────────────┐
    │ Query + Doc1 (concatenated)         │
    │ "return deadline electronics" +     │
    │ "Electronics: 7 days after..."      │
    └─────────────────────────────────────┘
                    │
                    ▼
            ┌───────────────┐
            │ Cross-Encoder │
            │  (BERT-based) │
            └───────────────┘
                    │
                    ▼
            Relevance Score: 0.92
```

The model sees the **interaction** between query and document:
- Which query terms appear in the document?
- Are they in the right context?
- Does the document actually answer the question?

**The trade-off:** Cross-encoders are slower (can't pre-compute) but more accurate.

### Bi-Encoder vs Cross-Encoder

| Aspect | Bi-Encoder | Cross-Encoder |
|--------|------------|---------------|
| **Speed** | Fast (pre-computed vectors) | Slow (compute per query-doc pair) |
| **Accuracy** | Good for recall | Better for precision |
| **Scalability** | Handles millions of docs | Only feasible for top-k candidates |
| **Use Case** | Initial retrieval | Reranking top candidates |

**Best Practice:** Use both! Bi-encoder for fast initial retrieval, cross-encoder for accurate reranking.

---

## Our Implementation

### Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Stage 1: Bi-Encoder Retrieval (Fast, High Recall)   │
│                                                     │
│   MPNet Embedding ──► FAISS Search ──► Top 7 docs   │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: Cross-Encoder Reranking (Slow, Precise)    │
│                                                     │
│   BGE Reranker ──► Score each of 7 docs ──► Top 3   │
└─────────────────────────────────────────────────────┘
    │
    ▼
Final 3 Documents (High Precision)
```

### Code Implementation

Here's our actual reranker implementation:

```python
"""
BGE Reranker for RAG Observatory.
Cross-encoder reranking to improve retrieval accuracy.
"""

from typing import List, Tuple
from langchain.schema import Document


class BGEReranker:
    """Cross-encoder reranker using BAAI BGE models."""

    def __init__(
        self, 
        model_name: str = "BAAI/bge-reranker-base", 
        use_fp16: bool = True
    ):
        """
        Initialize BGE reranker.

        Args:
            model_name: HuggingFace model name
            use_fp16: Use half precision for faster inference
        """
        from FlagEmbedding import FlagReranker

        self.model_name = model_name
        self.use_fp16 = use_fp16

        print(f"Loading reranker model: {model_name} (fp16={use_fp16})...")
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16)
        print(f"✓ Reranker loaded: {model_name}")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3,
        return_scores: bool = False
    ):
        """
        Rerank documents using cross-encoder model.

        Args:
            query: User query
            documents: List of retrieved documents (from bi-encoder)
            top_k: Number of top documents to return after reranking
            return_scores: If True, return (document, score) tuples

        Returns:
            List of top-k reranked documents (optionally with scores)
        """
        if not documents:
            return []

        # Prepare (query, doc) pairs for reranker
        pairs = [[query, doc.page_content] for doc in documents]

        # Get relevance scores from cross-encoder
        scores = self.reranker.compute_score(pairs)

        # Handle single document case (score is float, not list)
        if not isinstance(scores, list):
            scores = [scores]

        # Sort documents by score (descending)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Add reranker scores to document metadata
        for doc, score in doc_score_pairs:
            doc.metadata['reranker_score'] = float(score)

        # Return top-k
        if return_scores:
            return doc_score_pairs[:top_k]
        else:
            return [doc for doc, score in doc_score_pairs[:top_k]]
```

### Integration with RAG Pipeline

```python
# In rag.py - simplified version

def retrieve_context(query: str, config: dict) -> List[Document]:
    # Stage 1: Bi-encoder retrieval
    retriever = get_faiss_retriever(config)
    candidates = retriever.get_relevant_documents(query)[:config['retrieval_k']]
    
    # Stage 2: Reranking (if enabled)
    if config.get('use_reranker', False):
        reranker = BGEReranker(
            model_name=config['reranker_model'],
            use_fp16=config.get('reranker_use_fp16', True)
        )
        documents = reranker.rerank(
            query=query,
            documents=candidates,
            top_k=config['reranker_top_k']
        )
    else:
        documents = candidates
    
    return documents
```

### Configuration

```yaml
# Final production config

# Embedding (Stage 1)
embedding_model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
chunk_size: 500
chunk_overlap: 50

# Retrieval (Stage 1)
retrieval_k: 7           # Retrieve more candidates for reranking

# Reranking (Stage 2)
use_reranker: true
reranker_model: BAAI/bge-reranker-base
reranker_top_k: 3        # Return fewer, better results
reranker_use_fp16: true  # Faster inference
```

---

## Results

### Quantitative Improvement

| Experiment | Reranker | Precision | Recall | F1 | MRR |
|------------|----------|-----------|--------|-----|-----|
| Exp6 (Baseline) | ❌ No | 0.783 | 0.917 | 0.795 | 0.900 |
| **Exp9a1** | ✅ Yes | **0.828** | 0.950 | 0.845 | 0.950 |
| Change | - | **+5.7%** | +3.6% | +6.3% | +5.6% |

**+5.7% precision** — the single largest improvement across all 20+ experiments.

### Impact by Difficulty

| Difficulty | Without Reranker | With Reranker | Change |
|------------|------------------|---------------|--------|
| Easy (19 queries) | 0.762 | 0.781 | +2.5% |
| Medium (9 queries) | 0.815 | 0.889 | +9.1% |
| Hard (2 queries) | 0.833 | **1.000** | +20.0% |

**Key insight:** Reranker helped most on medium and hard queries — exactly where bi-encoders struggle with nuance.

### Hard Queries: Perfect Precision

Both hard queries achieved **1.000 precision** with reranker:

```
Query: "Barang sampai rusak, seller bilang custom order jadi tidak 
        bisa return. Tapi di listing tidak ada informasi itu. 
        Apa hak saya sebagai pembeli?"

Without Reranker: Retrieved general return policy (partially relevant)
With Reranker:    Retrieved specific section about buyer rights + 
                  custom order disputes (exactly relevant)
```

The cross-encoder understood the nuanced relationship between "custom order" + "no listing info" + "buyer rights" — something bi-encoder similarity couldn't capture.

---

## Why It Works

### 1. Token-Level Attention

Cross-encoders use transformer attention to see how query tokens relate to document tokens:

```
Query:    "return deadline electronics"
                ↓↓↓         ↓↓↓
Document: "... **deadline**: **Electronics**: 7 days ..."
```

The model learns that "deadline" in query should match "deadline:" in document, and "electronics" should match "Electronics:" — not just similar vectors.

### 2. Negation and Context Understanding

Bi-encoders struggle with negation:
- "Products that **CAN** be returned" 
- "Products that **CANNOT** be returned"

These have very similar embeddings! But cross-encoders read both together and understand the difference.

### 3. Fixing "Meleset Sedikit"

Our main failure pattern was retrieving the right document but wrong subsection. Rerankers help because:

| Scenario | Bi-Encoder | Cross-Encoder |
|----------|------------|---------------|
| Query about HIGH priority | "priority" matches both HIGH and LOW sections | Sees "HIGH" in query doesn't match "LOW" in doc |
| Query about Step 1 | "return process" matches all steps | Sees query wants beginning, not Step 3 |

---

## Configuration Guide

### Choosing retrieval_k and reranker_top_k

```
retrieval_k = 7 (candidates from bi-encoder)
                    │
                    ▼
            ┌───────────────┐
            │   Reranker    │
            └───────────────┘
                    │
                    ▼
reranker_top_k = 3 (final results)
```

**Guidelines:**

| Setting | Recommendation | Why |
|---------|----------------|-----|
| `retrieval_k` | 2-3x your final k | Give reranker enough candidates to work with |
| `reranker_top_k` | 3-5 for most cases | Balances coverage and precision |

**What we tested:**

| retrieval_k | reranker_top_k | Precision | Notes |
|-------------|----------------|-----------|-------|
| 3 | 3 | 0.783 | No benefit (reranker sees same docs as final output) |
| 5 | 3 | 0.811 | Slight improvement |
| **7** | **3** | **0.828** | **Sweet spot** ✅ |
| 10 | 3 | 0.822 | Diminishing returns |

> **Key insight:** When `retrieval_k` equals `reranker_top_k`, there's no benefit — the reranker just re-orders the same documents you'd get anyway. The magic happens when you retrieve more candidates than you need, letting the reranker pick the best ones.

### Model Selection

We tested with `BAAI/bge-reranker-base`. Other options:

| Model | File Size | Loaded Memory | Speed | Quality | Notes |
|-------|-----------|---------------|-------|---------|-------|
| `bge-reranker-base` | 278MB | ~600MB | Fast | Good | **Our choice** — best balance |
| `bge-reranker-large` | 560MB | ~1.2GB | Medium | Better | +1-2% quality, 2x slower |
| `bge-reranker-v2-m3` | 568MB | ~1.2GB | Medium | Best | Multilingual optimized |

For Indonesian e-commerce, `bge-reranker-base` was sufficient. The larger models showed minimal improvement for our domain.

### Chunk Size Interaction

**Important finding:** Chunk size matters more with rerankers.

| Config | Chunk Size | Precision | Notes |
|--------|------------|-----------|-------|
| Exp9a1 | 500 | **0.828** | ✅ Winner |
| Exp9a2 | 700 | 0.778 | ❌ Below baseline! |

**Why?** Larger chunks contain more noise. The reranker still scores the *whole chunk*, so irrelevant content in a 700-char chunk can drag down the score of otherwise relevant content.

**Recommendation:** Use smaller, focused chunks (400-500 chars) when using rerankers.

---

## Trade-offs and Considerations

### Latency

Cross-encoders add latency because they can't pre-compute:

```
Without Reranker:
  Query → FAISS lookup (fast) → Results
  Total: ~50-100ms

With Reranker:
  Query → FAISS lookup → Rerank 7 docs → Results
  Total: ~150-300ms (estimated, not measured in this study)
```

> **Note:** Latency figures are estimates based on typical cross-encoder performance. Actual latency depends on hardware, model size, and document length. We recommend benchmarking in your specific environment.

**Mitigation strategies:**
- Use `fp16=True` for faster inference
- Keep `retrieval_k` reasonable (7-10, not 50)
- Use smaller reranker models for latency-sensitive applications

### Memory

Reranker models need to be loaded in memory:

| Model | File Size | Loaded Memory |
|-------|-----------|---------------|
| `bge-reranker-base` | 278MB | ~600MB |
| `bge-reranker-large` | 560MB | ~1.2GB |

**Tip:** Load reranker once at startup, not per-query.

### When NOT to Use Rerankers

| Scenario | Recommendation |
|----------|----------------|
| Latency-critical (<100ms required) | Skip reranker or use async |
| Very large k (>20 results needed) | Reranking cost too high |
| Simple keyword queries | Bi-encoder often sufficient |
| Resource-constrained environment | May not fit in memory |

---

## Recommendations

### 1. Start with Reranker Early

We tested reranker in Phase 9 — late in our research. In hindsight, we should have tried it earlier. It provided better ROI than:
- ❌ Embedding model experiments (5 hours, -14% precision)
- ❌ Splitter experiments (4 hours, -17% precision)
- ❌ Hybrid search experiments (4 hours, -3% precision)

**If you're optimizing RAG, try reranker before exotic approaches.**

### 2. Use the 7→3 Pattern

Our optimal configuration:
```yaml
retrieval_k: 7        # Cast a wide net
reranker_top_k: 3     # Narrow to best
```

This pattern works because:
- 7 candidates gives reranker enough options
- 3 results is usually sufficient for LLM context
- More than 7 candidates has diminishing returns

### 3. Pair with Smaller Chunks

Rerankers work better with focused chunks:
- ✅ 500 chars: Each chunk is one concept
- ❌ 1000 chars: Chunk mixes multiple concepts, harder to score

### 4. Monitor Reranker Scores

We store reranker scores in metadata:
```python
doc.metadata['reranker_score'] = float(score)
```

This enables:
- Threshold filtering (reject low-confidence results)
- Debugging (why was this doc ranked higher?)
- Analytics (score distribution over time)

### 5. Consider Score Thresholds

BGE reranker scores are roughly:
- `> 1.0`: High confidence relevant
- `0 to 1.0`: Moderate confidence
- `< 0`: Likely not relevant

You can add threshold filtering:
```python
# Only return docs above confidence threshold
filtered = [doc for doc, score in results if score > 0.5]
```

---

## Summary

### What We Learned

1. **Reranker = highest ROI optimization** — +5.7% precision, more than any other change
2. **Cross-encoders fix "meleset sedikit"** — they understand query-document interaction
3. **Configuration matters** — 7→3 pattern, smaller chunks, fp16 enabled
4. **Hard queries benefit most** — reranker took hard query precision from 0.833 to 1.000

### Quick Start

```python
# Install
pip install FlagEmbedding

# Use
from FlagEmbedding import FlagReranker

reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=True)
scores = reranker.compute_score([
    ["user query", "document 1 content"],
    ["user query", "document 2 content"],
])
# scores = [0.92, 0.45] — higher is more relevant
```

### Final Configuration

```yaml
embedding_model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
chunk_size: 500
chunk_overlap: 50
retrieval_k: 7

use_reranker: true
reranker_model: BAAI/bge-reranker-base
reranker_top_k: 3
reranker_use_fp16: true
```

---

<p align="center">
  <i>"The best optimization is often not a better algorithm, but a second opinion."</i>
</p>

---

📖 **[← Back to RAG Observatory](../../README.md)** | **[← Previous: Metrics vs Quality Gap](01-metrics-vs-quality-gap.md)**