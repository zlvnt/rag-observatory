# Phase 9A: Reranker Experiments - Summary

**Status:** ✅ COMPLETED
**Goal:** Test cross-encoder reranking to bridge precision gap (0.783 → 0.80)
**Result:** **TARGET ACHIEVED** - Exp9a1 reached 0.828 precision (+5.7% improvement)

---

## Overview: Reranker Implementation

### What We Did
Implemented BGE Reranker (cross-encoder) on top of MPNet embedding (bi-encoder):
- **Baseline (Exp6):** MPNet retrieval with k=3 → Precision 0.783
- **Exp9a1 (chunk 500):** MPNet retrieves k=7 → Reranker returns top-3 → Precision 0.828 ✅
- **Exp9a2 (chunk 700):** Same reranker, larger chunks → Precision 0.778 ❌

### Quantitative Results

| Experiment | Chunk Size | Precision | Recall | F1 | MRR | Winner |
|------------|------------|-----------|--------|-----|-----|--------|
| Exp6 (Baseline) | 500 | 0.783 | 0.950 | 0.795 | 0.900 | - |
| **Exp9a1** | 500 | **0.828** | 0.950 | 0.845 | 0.950 | ✅ **WINNER** |
| **Exp9a2** | 700 | **0.778** | 0.933 | 0.800 | 0.911 | ❌ Below target |

**Key findings:**
- Exp9a1: +5.7% precision vs baseline, **TARGET ACHIEVED** (≥0.80)
- Exp9a2: -0.5% precision vs baseline, **FAILED to reach target**
- Chunk 700 actually **worse** than baseline with reranker

---

## Critical Discovery: Qualitative Analysis

### User's Manual Quality Evaluation
**Source:** Manual evaluation of 30 queries in "RAG observatory - qualitative check.csv"

| Experiment | Success Rate | Winner |
|------------|-------------|--------|
| **Exp9a1 (Chunk 500)** | **56%** (17/30 queries) | ✅ **QUALITATIVE WINNER** |
| Exp9a2 (Chunk 700) | **52%** (16/30 queries) | |

**What "quality" means:** Whether retrieved text contains the complete, correct answer with minimal noise.

### Key Finding: Exp9a1 Wins on BOTH Metrics

**Quantitative:** Exp9a1 wins (0.828 vs 0.778 precision)
**Qualitative:** Exp9a1 wins (56% vs 52% success rate)

**Conclusion:** Chunk 500 is superior to chunk 700 when using reranker.

### Why Chunk 700 Fails?

**Root cause:** Larger chunks include more noise, confusing both embeddings and reranker.

**Example from qualitative analysis:**
```
Query: "Bagaimana cara return barang yang rusak?"

Exp9a1 (Chunk 500):
✅ Retrieved exact procedure section
Note: "prosedural, kurang lengkap" (incomplete but focused)

Exp9a2 (Chunk 700):
❌ Retrieved broader policy section with extra content
Note: Same issue but with more irrelevant context mixed in
```

---

## Key Insights from Qualitative Analysis

### 1. **"Meleset sedikit"** (30% of queries)
**Pattern:** Reranker improves but doesn't eliminate subsection precision issues

**Example:**
- Query: "Return barang rusak"
- Retrieved: "Return policy general" (nearby section, not exact match)

**User observation:** Both Exp9a1 and Exp9a2 still have this problem

### 2. **"Prosedural, kurang lengkap"** (10% of queries)
**Pattern:** Chunk 500 sometimes cuts mid-procedure

**User observation:** Retrieves *start* of step-by-step instructions, missing later steps

### 3. **"Tough question, mungkin butuh semacam metode untuk bisa paraphrase"**
**Pattern:** Complex queries need query preprocessing/expansion

**User insight:** Some queries need reformulation before retrieval can work well

### 4. **"Perhaps need require real cs"**
**Pattern:** Edge cases beyond RAG capability

**Examples:**
- "Apakah bisa tukar warna setelah order?"
- "Garansi berlaku untuk kerusakan akibat salah packing?"

**User observation:** Some queries need human judgment, not just document retrieval

### 5. **Chunk Size Trade-off**
- **Chunk 500 (Exp9a1):** Better precision, less noise, but risk of incompleteness
- **Chunk 700 (Exp9a2):** Better recall, more complete, but includes irrelevant content

---

## By Difficulty Breakdown

| Difficulty | Metric | Exp9a1 | Exp9a2 | Winner |
|------------|--------|--------|--------|--------|
| **Easy (19 queries)** | Precision | 0.781 | 0.702 | Exp9a1 (+7.9%) |
| | Recall | 1.000 | 1.000 | TIE |
| | MRR | 0.947 | 0.860 | Exp9a1 (+8.7%) |
| **Medium (9 queries)** | Precision | 0.889 | 0.889 | TIE |
| | Recall | 0.889 | 0.833 | Exp9a1 (+5.6%) |
| | MRR | 0.944 | 1.000 | Exp9a2 (+5.6%) |
| **Hard (2 queries)** | Precision | 1.000 | 1.000 | TIE |
| | Recall | 0.750 | 0.750 | TIE |
| | MRR | 1.000 | 1.000 | TIE |

**Key observations:**
- Exp9a1 significantly better on **easy queries** (+7.9% precision)
- Both perform equally on **hard queries** (perfect 1.000 precision)
- Exp9a2's lower overall precision comes from easy query failures

---

## What's Still Broken?

1. **"Meleset sedikit" (30%)** - Subsection precision issues persist
2. **Incomplete procedures (10%)** - Chunking cuts mid-content
3. **Tough questions (5%)** - Need query preprocessing
4. **Edge cases** - Some queries need human CS

**Potential solutions:** Hybrid BM25, query preprocessing, better chunking strategy

---

## Conclusion

**Phase 9A: SUCCESS ✅**
- Reranker works: +5.7% precision improvement
- Target exceeded: 0.828 (goal: ≥ 0.80)
- Qualitative analysis reveals Exp9a1 as production winner
- Key learning: **Trust qualitative evaluation when metrics diverge**

**Next:** Accept Exp9a1 as production config, or continue to Phase 9B for stretch goal (0.85-0.90)
