# Phase 9B: Hybrid Search Experiments - Summary

**Status:** ❌ FAILED - All variants worse than reranker-only
**Date:** November 2025
**Goal:** Improve upon Exp9a1 (0.828 precision) with hybrid search (Semantic + BM25)
**Result:** **Hybrid search rejected** - Semantic + Reranker alone is optimal

---

## Executive Summary

### Hypothesis
**BM25 keyword matching + Semantic search** would fix "meleset sedikit" cases by catching exact keyword matches that embeddings miss.

### Result
**Hypothesis REJECTED** - All hybrid variants performed worse than reranker-only (Exp9a1).

### Key Finding
**For Indonesian e-commerce CS docs, simpler is better:**
- Semantic search + Reranker = **0.828 precision** ✅
- Hybrid search (all variants) = **0.794-0.811 precision** ❌
- BM25 adds more **noise than signal**

---

## Experiments Executed

| Experiment | Hybrid Weights | Embedding | Precision | Change vs 9a1 | Status |
|------------|----------------|-----------|-----------|---------------|--------|
| **Exp9a1** (Baseline) | N/A (Reranker only) | MPNet | **0.828** | Baseline | ✅ WINNER |
| **Exp9b1** | [0.5, 0.5] (50/50) | MPNet | 0.794 | **-3.4%** | ❌ Failed |
| **Exp9b2** | [0.5, 0.5] (50/50) | bge-m3 | Lower | **Worse** | ❌ Failed |
| **Exp9b3** | [0.7, 0.3] (70/30) | MPNet | 0.811 | **-1.7%** | ⚠️ Better than 50/50 but still below baseline |

---

## Detailed Analysis

### Exp9b1: 50/50 Weights (First Attempt)

**Config:**
```yaml
use_hybrid_search: true
hybrid_weights: [0.5, 0.5]  # Equal weight
retrieval_k: 7
use_reranker: true
```

**Results:**
- Precision: **0.794** (-3.4% vs 9a1) ❌
- Recall: 0.950 (same)
- F1: 0.821 (-2.4%)

**By Difficulty:**
- Easy: 0.746 (9a1: 0.781, -3.5%)
- Medium: 0.889 (SAME)
- Hard: 0.834 (9a1: 1.000, **-16.6%!**)

**Problem identified:**
- **50/50 = too much BM25 power**
- BM25 added noise, degraded easy and hard queries
- Indonesian common words ("cara", "barang") match too broadly

---

### Exp9b2: bge-m3 Embedding + 50/50

**Config:**
```yaml
embedding_model: BAAI/bge-m3  # Changed from MPNet
hybrid_weights: [0.5, 0.5]
```

**Results:**
- **Even worse** than Exp9b1
- Better embedding didn't help when BM25 adds noise

**Conclusion:** Problem is BM25 weights, not embedding quality

---

### Exp9b3: 70/30 Weights (Conservative)

**Config:**
```yaml
hybrid_weights: [0.7, 0.3]  # 70% semantic, 30% BM25
```

**Results:**
- Precision: **0.811** (-1.7% vs 9a1) ⚠️
- Recall: 0.950 (same)
- F1: 0.832 (-1.3%)

**By Difficulty:**
- Easy: 0.772 (9a1: 0.781, -0.9%)
- Medium: 0.889 (SAME)
- Hard: 0.834 (9a1: 1.000, **-16.6%**)

**Improvement:**
- ✅ Better than 50/50 (+1.7%)
- ✅ Weight tuning worked as expected
- ❌ But still below reranker-only

---

## Root Cause Analysis

### Why Did Hybrid Search Fail?

#### **1. BM25 Adds Noise, Not Signal**

**Text Quality Comparison (Queries 1-5):**
- **Identical results** for 5/5 queries
- BM25 (30% weight) too weak to change rankings
- When BM25 did make a difference: **100% regressions**

**Queries Where BM25 Changed Results:**
- **ecom_easy_012:** Added wrong doc (policy_returns.md), precision dropped -16.7%
- **ecom_hard_002:** Added wrong doc (troubleshooting_guide.md), precision dropped **-33.3%**

**Summary:**
- ❌ 0 queries improved by BM25
- ❌ 2 queries degraded by BM25
- ⚠️ 28 queries unchanged (BM25 too weak)

---

#### **2. Indonesian Tokenization Issues**

**Problem:** BM25 keyword matching struggles with Indonesian text

**Examples:**

```
Query: "Berapa lama batas waktu bayar setelah checkout?"

BM25 matches:
- "bayar" → policy_returns.md (refund mentions "bayar") ❌ Wrong context
- "batas waktu" → All docs (common phrase) ❌ Too broad

Result: Adds irrelevant documents
```

```
Query: "Sudah eskalasi ke CS tapi tidak ada solusi..."

BM25 matches:
- "masalah" → troubleshooting_guide.md ❌ Generic match
- "solusi" → All guides ❌ Too common

Result: Semantic understanding needed, not keywords
```

---

#### **3. Hard Queries Especially Hurt**

**Hard queries require semantic understanding:**
- Complex reasoning
- Policy interpretation
- Multi-step escalation paths

**BM25 fails here:**
- Matches keywords broadly ("masalah", "solusi")
- Misses nuance and context
- Adds wrong documents

**Evidence:**
- Exp9b3 hard query precision: 0.834 (9a1: 1.000)
- **-16.6% regression on hardest queries**

---

## Key Learnings

### ❌ **What Didn't Work:**

1. **50/50 Hybrid Weights**
   - Too much BM25 power
   - Adds significant noise
   - Regressed -3.4% precision

2. **Better Embeddings + Hybrid**
   - bge-m3 didn't help when BM25 adds noise
   - Problem is BM25, not semantic quality

3. **70/30 Conservative Weights**
   - Better than 50/50 but still below baseline
   - BM25 even at 30% adds more noise than signal

### ✅ **What We Learned:**

1. **Simpler is Better**
   - Semantic + Reranker alone = optimal
   - No need for hybrid complexity

2. **BM25 Limits for Indonesian Text**
   - Common words match too broadly
   - Keyword matching confuses Indonesian CS docs
   - Semantic understanding superior

3. **Hard Queries Need Semantics**
   - Keyword matching fails on complex queries
   - Cross-encoder reranker handles nuance better

4. **Weight Tuning Validated**
   - 70/30 > 50/50 (hypothesis confirmed)
   - But even optimal weights can't overcome fundamental mismatch

---

## Implementation Notes

### Code Created:
- ✅ `z3_core/hybrid_search.py` - Hybrid retriever implementation
- ✅ `z3_core/rag.py` - Hybrid search integration
- ✅ `z3_core/domain_config.py` - Hybrid config parameters
- ✅ `runners/test_runner.py` - Hybrid retriever building

### Key Technical Decisions:
1. **BM25 built once** (not per query) - Performance optimization
2. **EnsembleRetriever** (Langchain) - Combines FAISS + BM25
3. **Weights configurable** via YAML - Easy experimentation
4. **rank_bm25 library** - Required dependency

### Performance:
- BM25 build time: ~1 second (36 chunks)
- Query latency: +0ms (built once, reused)
- No performance penalty vs reranker-only

---

## Alternative Approaches Considered

### **1. Increase BM25 Weight (40-50%)**
**Status:** Not tested
**Reasoning:** 50/50 failed badly, 70/30 still failed. More BM25 = more noise.

### **2. Reduce BM25 k (k=5 instead of 7)**
**Status:** Not tested
**Reasoning:** 70/30 already gives BM25 low influence. Won't help fundamental issue.

### **3. Conditional Hybrid (BM25 only for keyword queries)**
**Status:** Not tested
**Reasoning:** Complex to implement, marginal benefit. Simpler to use semantic-only.

### **4. BM25-only (no semantic)**
**Status:** Not tested
**Reasoning:** Would be worse - semantic is proven to work.

---

## Comparison with Baseline

### Exp9a1 (Reranker-only) vs Exp9b3 (Best Hybrid)

| Metric | Exp9a1 | Exp9b3 | Winner |
|--------|--------|--------|--------|
| Precision | **0.828** | 0.811 | 9a1 (+1.7%) |
| Recall | 0.950 | 0.950 | TIE |
| F1 | 0.845 | 0.832 | 9a1 (+1.3%) |
| MRR | 0.950 | 0.950 | TIE |
| Easy Precision | 0.781 | 0.772 | 9a1 (+0.9%) |
| Hard Precision | 1.000 | 0.834 | 9a1 (**+16.6%**) |
| Complexity | Low | High | 9a1 (simpler) |
| Dependencies | FlagEmbedding | +rank_bm25 | 9a1 (fewer) |

**Winner:** **Exp9a1** on all metrics + simplicity

---

## Recommendations

### ✅ **For Production:**
**Use Exp9a1 (Reranker-only)**

**Config:**
```yaml
embedding_model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
chunk_size: 500
chunk_overlap: 50
retrieval_k: 7
use_reranker: true
reranker_model: BAAI/bge-reranker-base
reranker_top_k: 3
use_hybrid_search: false  # Do NOT use hybrid
```

**Rationale:**
- ✅ Target achieved (0.828 > 0.80)
- ✅ Simpler implementation
- ✅ Fewer dependencies
- ✅ Better on hard queries
- ✅ Proven reliable

---

### ❌ **Do NOT Use Hybrid Search For:**
- Indonesian language CS docs
- Queries with common words
- Complex semantic queries
- When semantic search already works well

### ⚠️ **Hybrid Search Might Work For:**
- English documents (better BM25 tokenization)
- Keyword-heavy domains (technical specs, product codes)
- When semantic search demonstrably fails on exact matches

---

## Future Work (Optional)

### **Phase 10 (If Pursuing 0.85+ Stretch Goal):**

**1. Query Preprocessing** (HIGH POTENTIAL)
- Fix typos ("gmna cra" → "bagaimana cara")
- Expand abbreviations ("CS" → "customer service")
- Expected: +2-3% precision
- Complexity: Medium (single LLM call)

**2. MMR for Multi-Doc Diversity** (MEDIUM POTENTIAL)
- Improve multi-doc query recall
- Expected: +2-4% recall
- Complexity: Low (Langchain supports MMR)

**3. Accept 0.828 and Deploy** (RECOMMENDED)
- Target exceeded
- Research complete
- Production-ready config identified

---

## Conclusion

**Phase 9B Result:** ❌ **Hybrid Search Rejected**

**Key Takeaway:**
> **For Indonesian e-commerce customer service documents, semantic search + cross-encoder reranking is optimal. BM25 keyword matching adds noise without providing value. Simpler is better.**

**Final Production Config:** **Exp9a1** (Semantic + Reranker)
- Precision: **0.828** ✅
- Simple, reliable, production-ready
- No need for hybrid complexity

---

*Phase 9B completed November 2025*
*Research finding: Hybrid search not suitable for this use case*
