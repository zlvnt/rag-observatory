# RAG Observatory 🔭

> **"When 80% precision actually means 50% quality"**
> A systematic study on why your RAG metrics might be lying to you.

[![Research Status](https://img.shields.io/badge/Research-Complete-success)](https://github.com/zlvnt/rag-observatory)
[![Experiments](https://img.shields.io/badge/Experiments-20+-blue)](https://github.com/zlvnt/rag-observatory)
[![Domain](https://img.shields.io/badge/Domain-Indonesian%20E--commerce-orange)](https://github.com/zlvnt/rag-observatory)

---

## 🎯 TL;DR

A 2-month research project optimizing RAG retrieval for Indonesian e-commerce customer service, featuring:

- **20+ experiments** across 9 systematic phases
- **30-query golden dataset** with difficulty stratification
- **Critical finding:** Automated metrics can be dangerously misleading
- **Counter-intuitive results:** SOTA models underperformed simpler alternatives

**Final Achievement:** 0.828 precision with cross-encoder reranking — but manual evaluation revealed only 56% actual quality.

---

## 📋 Table of Contents

- [The Problem](#-the-problem)
- [The Journey](#-the-journey)
- [The Plot Twist](#-the-plot-twist-metrics-vs-quality)
- [Key Findings](#-key-findings)
- [What Failed (And Why)](#-what-failed-and-why)
- [Final Configuration](#-final-configuration)
- [Repository Structure](#-repository-structure)
- [Lessons Learned](#-lessons-learned)

---

## 🔍 The Problem

Building a RAG system for Indonesian e-commerce customer service presents unique challenges:

- **Domain-specific language:** Mix of Indonesian, informal terms, and e-commerce jargon
- **Multi-document answers:** Customer queries often require synthesizing information from multiple policy documents
- **Precision requirements:** Wrong information in customer service = customer frustration

**Research Question:** What RAG configuration achieves optimal retrieval quality for this domain?

---

## 🚀 The Journey

### Phase Overview

| Phase | Focus | Experiments | Key Outcome |
|-------|-------|-------------|-------------|
| **1-4** | Framework & Dataset | - | 30-query golden dataset, evaluation pipeline |
| **5** | Baseline | 1 | Precision 0.706, Recall 0.950 |
| **6** | Parameter Optimization | 7 | Optimal k=3, MPNet, chunk=500 |
| **7** | Analysis | - | Winner: Exp6 (0.783 precision) |
| **8A** | Qualitative Analysis | - | Identified 5 failure patterns |
| **8B** | Embedding Ablation | 4 | BGE-M3 **failed** (-14% precision) |
| **8C** | Splitter Ablation | 3 | MarkdownSplitter **failed** (-17% precision) |
| **8D** | Production Config | - | Confirmed Exp6 as optimal base |
| **8E** | Parent-Child Chunking | 2 | **Failed** (-15% precision) |
| **9A** | Cross-Encoder Reranker | 2 | **Success!** +5.7% → 0.828 precision |
| **9B** | Hybrid BM25 Search | 3 | **Failed** (-3% precision) |

### Precision Progression

```
Baseline (Exp0)     ████████████████████░░░░░░░░░░  0.706
After Phase 6       ██████████████████████████░░░░  0.783  (+11%)
After Phase 9A      ████████████████████████████░░  0.828  (+17%)
                    ─────────────────────────────────────────
                    0.0                            0.80   1.0
                                                    ↑
                                              Target: 0.80
```

---

## 🎭 The Plot Twist: Metrics vs Quality

This is the most important finding of this research.

### The Numbers Look Great...

| Experiment | Precision | Recall | F1 | Status |
|------------|-----------|--------|-----|--------|
| Exp9a1 (Reranker) | **0.828** | 0.950 | 0.845 | ✅ Target exceeded! |

### But Manual Evaluation Revealed...

| Experiment | Automated Precision | Manual Quality Score |
|------------|---------------------|----------------------|
| Exp9a1 | 0.828 (83%) | **0.56 (56%)** |

**A 27-point gap between what metrics say and what users experience.**

### Why the Gap?

Through manual inspection of all 30 queries, I identified these patterns:

| Pattern | Frequency | What Happens |
|---------|-----------|--------------|
| **"Meleset Sedikit"** | 30% | Right document, wrong subsection (2-6 lines off) |
| **Context Cutting** | 40% | Answer truncated mid-procedure |
| **Ranking Issues** | 10% | Correct doc retrieved but ranked too low |
| **Multi-doc Failures** | 20% | Only 1 of 2 required docs retrieved |

**Example of "Meleset Sedikit":**
```
Query: "Berapa lama batas waktu return elektronik?"
       (What's the return deadline for electronics?)

Expected: "### Batas waktu: Elektronik 7 hari" (line 20-24)
Retrieved: "### Produk yang TIDAK BISA di-return" (line 12-18)

Metrics say: ✅ Correct document retrieved!
Reality: ❌ User gets wrong information (what CAN'T be returned, not deadline)
```

### The Lesson

> **Automated metrics measure document-level accuracy. Users need subsection-level precision.**
>
> Always validate with human evaluation. Metrics can lie.

---

## 💡 Key Findings

### 1. Simpler Often Beats SOTA

| Approach | Model Size | Precision | Verdict |
|----------|-----------|-----------|---------|
| MPNet (2019) | 420 MB | **0.783** | ✅ Winner |
| BGE-M3 (2024, SOTA) | 2.2 GB | 0.639-0.772 | ❌ Failed |

**Why?** BGE-M3's multi-functional retrieval (dense + sparse + ColBERT) added noise rather than signal for compact e-commerce documents.

### 2. Semantic Splitting ≠ Better Splitting

| Splitter | Approach | Precision | Verdict |
|----------|----------|-----------|---------|
| RecursiveCharacter | Character-based | **0.783** | ✅ Winner |
| MarkdownHeader | Semantic boundaries | 0.589-0.711 | ❌ Failed |

**Why?** Our documents have deeply nested headers. Markdown splitting created too many tiny chunks (18 vs 5 per doc), losing context.

### 3. Hybrid Search Hurts Indonesian Text

| Method | Precision | Verdict |
|--------|-----------|---------|
| Semantic + Reranker | **0.828** | ✅ Winner |
| Semantic + BM25 + Reranker | 0.794-0.811 | ❌ Failed |

**Why?** BM25 keyword matching struggles with Indonesian — common words like "cara", "barang", "untuk" match too broadly, adding irrelevant results.

### 4. Reranker is the Biggest Lever

| Configuration | Precision | Change |
|---------------|-----------|--------|
| Without Reranker | 0.783 | Baseline |
| With BGE Reranker | **0.828** | **+5.7%** |

The cross-encoder reranker provided the single largest precision improvement across all experiments.

---

## ❌ What Failed (And Why)

Documenting failures is as valuable as documenting successes.

### BGE-M3 Embedding (Phase 8B)
- **Hypothesis:** Multi-functional retrieval would outperform dense-only
- **Result:** -14.4% precision (worst variant)
- **Root Cause:** Sparse retrieval caused keyword confusion; ColBERT over-matched common Indonesian words
- **Time Invested:** 5 hours
- **Lesson:** Domain fit > Model sophistication

### MarkdownHeaderTextSplitter (Phase 8C)
- **Hypothesis:** Preserving semantic boundaries would improve quality
- **Result:** -17.2% precision (with k=5)
- **Root Cause:** Created 18 tiny chunks vs 5 medium chunks; lost parent context
- **Time Invested:** 4 hours
- **Lesson:** Context completeness > Semantic alignment

### Parent-Child Chunking (Phase 8E)
- **Hypothesis:** Hierarchical chunking would combine benefits of both approaches
- **Result:** -15.6% precision
- **Root Cause:** All sections already < threshold; became equivalent to failed Markdown approach
- **Time Invested:** 3 hours
- **Lesson:** Technique must match document characteristics

### Hybrid BM25 Search (Phase 9B)
- **Hypothesis:** Keyword matching would catch cases embeddings miss
- **Result:** -3.4% precision
- **Root Cause:** Indonesian common words matched too broadly; added noise to results
- **Time Invested:** 4 hours
- **Lesson:** BM25 may not suit all languages/domains

**Total time on failed experiments: ~16 hours**

These failures weren't wasted — they definitively proved what doesn't work for this domain.

---

## ⚙️ Final Configuration

After 20+ experiments, this configuration achieved the best results:

```yaml
# Embedding
embedding_model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2

# Chunking
text_splitter: RecursiveCharacterTextSplitter
chunk_size: 500
chunk_overlap: 50

# Retrieval
retrieval_k: 7              # Retrieve 7 candidates for reranking

# Reranking
use_reranker: true
reranker_model: BAAI/bge-reranker-base
reranker_top_k: 3           # Return top 3 after reranking

# What NOT to use
use_hybrid_search: false    # BM25 adds noise for Indonesian
```

### Performance Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Precision@3 | 0.828 | ≥ 0.80 | ✅ Exceeded |
| Recall@3 | 0.950 | ≥ 0.90 | ✅ Exceeded |
| F1 Score | 0.845 | ≥ 0.75 | ✅ Exceeded |
| MRR | 0.950 | ≥ 0.80 | ✅ Exceeded |
| Success Rate | 100% | ≥ 90% | ✅ Perfect |

### Performance by Difficulty

| Difficulty | Queries | Precision | Recall |
|------------|---------|-----------|--------|
| Easy | 19 | 0.781 | 1.000 |
| Medium | 9 | 0.889 | 0.889 |
| Hard | 2 | **1.000** | 0.750 |

---

## 📁 Repository Structure

```
rag-observatory/
├── configs/              # YAML experiment configurations
├── data/vector_stores/   # FAISS vector indexes
├── evaluators/           # Evaluation metrics & logic
├── golden_datasets/      # 30-query test set with ground truth
├── research/             # Phase summaries & detailed analysis
│   ├── PHASE_8A_SUMMARY.md
│   ├── PHASE_8B_SUMMARY.md
│   ├── PHASE_9A_SUMMARY.md
│   └── ...
├── results/              # Experiment outputs (CSV, JSON, reports)
├── runners/              # Test execution scripts
├── scripts/              # Utility & analysis scripts
├── z3_core/              # Core RAG engine
│   ├── vector.py         # FAISS & embeddings
│   ├── rag.py            # Retrieval logic
│   └── reranker.py       # Cross-encoder reranking
├── CLAUDE.md             # Project instructions (for AI assistants)
├── PROGRESS.md           # Research progress tracking
└── requirements.txt
```

---

## 🎓 Lessons Learned

### On RAG Optimization

1. **Start with simple baselines** — SOTA models aren't always better for your domain
2. **Reranker has highest ROI** — Single biggest improvement across all experiments
3. **Test one variable at a time** — Ablation studies reveal what actually matters
4. **Document negative results** — Knowing what doesn't work saves future effort

### On Evaluation

5. **Metrics can lie** — 0.828 precision ≠ 82.8% user satisfaction
6. **Manual evaluation is essential** — Subsection-level errors invisible to automated metrics
7. **Create difficulty-stratified test sets** — Easy queries can mask problems with hard ones

### On Research Process

8. **Time-box experiments** — Know when to stop and pivot (16 hours on failed approaches)
9. **Domain fit > Model size** — 420MB MPNet beat 2.2GB BGE-M3
10. **Qualitative analysis reveals root causes** — Numbers tell you what, inspection tells you why

---

## 📊 Research Stats

- **Duration:** 2 months (October - November 2025)
- **Total Experiments:** 20+
- **Phases Completed:** 9
- **Golden Dataset:** 30 queries (Easy: 19, Medium: 9, Hard: 2)
- **Documents Indexed:** 4 policy documents
- **Final Precision:** 0.828 (target: 0.80)

---

## 👤 Author

**Nando** — [@zlvnt](https://github.com/zlvnt)

---

## 📄 License

This project is for research and educational purposes.

---

<p align="center">
  <i>Research is formalized curiosity. It is poking and prying with a purpose.</i>
  <br>
  — Zora Neale Hurston
</p>