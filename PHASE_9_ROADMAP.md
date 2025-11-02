# Phase 9 Roadmap - Advanced Retrieval Techniques

**Date:** 2025-10-26 (Updated)
**Status:** Ready to start | Phase 8 (A-E) complete (basic optimization exhausted)

---

## 🎯 Phase 9 Overview

**Goal:** Bridge the 2.2% precision gap (0.783 → 0.80+) using advanced retrieval techniques

**Target:** Precision ≥ 0.80 (minimum) | Stretch: 0.85-0.90 (with combined techniques)

**Current State:**
- Winner: Exp6 (MPNet + RecursiveCharacterTextSplitter + k=3)
- Precision: 0.783 (2.2% gap to 0.80 target)
- Recall: 0.917 (exceeds target)
- F1: 0.795 (exceeds target)

**Phase 8 Findings (What We Exhausted):**
- ✅ Embedding optimization complete (MPNet optimal, BGE-M3 failed -1.4% to -14.4%)
- ✅ Splitter optimization complete (Recursive optimal, Markdown failed -8% to -17%)
- ✅ Parent-Child approach rejected (failed -15.6%, docs too compact)
- ✅ k parameter optimized (k=3 best)
- ✅ Threshold optimized (0.3 best)

**Remaining Failure Patterns (from Phase 8A):**
1. 🔴 **Ranking issues (10%)** - Correct doc retrieved but ranked low
2. 🟠 **"Meleset sedikit" (30%)** - Right doc, wrong subsection ranked higher
3. 🟡 **Multi-doc failures (20%)** - Only 1 doc retrieved when 2+ expected
4. 🟢 **Context cutting (40%)** - Splitter cuts mid-section (cannot fix with current tools)

**Phase 9 Strategy (Revised based on Claude web insights):**
- **Priority 1:** Reranker (fixes 40% of failures: ranking + meleset sedikit)
- **Priority 2:** Hybrid Search BM25 (keyword match for exact terms)
- **Priority 3:** MMR (only if multi-doc still problematic after reranker)
- **Priority 4:** Query preprocessing (optional, +1-2% polish)
- No more embedding/splitter experiments

---

## 🗺️ Phase 9 Roadmap (Prioritized by Impact)

### **Phase 9A: Reranker (Cross-Encoder)** ⏳ HIGHEST PRIORITY

**Goal:** Fix ranking accuracy using neural cross-encoder model

**Duration:** 3-4 hours (includes model download ~600MB-1.5GB)

**Expected Impact:** +5-7% precision → **0.83-0.85** (exceeds 0.80 target!)

**Why Priority #1:**
- Fixes **40% of failures** (ranking 10% + "meleset sedikit" 30%)
- Highest ROI technique from Claude web analysis
- No config changes needed (works on top of Exp6)
- Proven technique in production RAG systems

**What is Reranker:**
- **Model:** Cross-encoder (not bi-encoder like MPNet)
- **Type:** BAAI/bge-reranker-base (600MB) or bge-reranker-v2-m3 (1.5GB)
- **Input:** (query, document) pairs → **Output:** relevance score (0-1)
- **Advantage:** Much more accurate than bi-encoder similarity
- **How different:**
  - Bi-encoder: Encode query & doc separately, compute similarity
  - Cross-encoder: Encode (query + doc) together, learns interaction

**How it works:**
```
Standard retrieval (MPNet bi-encoder, k=3):
1. policy_returns.md (similarity: 0.85) ← WRONG section
2. troubleshooting_guide.md (similarity: 0.83) ← CORRECT but ranked #2
3. product_faq.md (similarity: 0.80)

After reranking (cross-encoder, top-3):
1. troubleshooting_guide.md (score: 0.95) ← CORRECT, now ranked #1! ✅
2. policy_returns.md (score: 0.72) ← Correct section now
3. contact_escalation.md (score: 0.68)
```

**Pipeline:**
```
Query → Retrieve k=7 (MPNet) → Rerank (cross-encoder) → Return top-3
```

**Why k=7 → top-3:**
- Reranker needs more candidates to choose from
- k=3 too few for reranker to be effective
- Cross-encoder slower → filter down to top-3 for LLM

---

### **Phase 9B: Hybrid Search (BM25 + Semantic)** ⏳ OPTIONAL

**Goal:** Combine keyword matching (BM25) with semantic search (MPNet)

**Duration:** 2-3 hours

**Expected Impact:** +2-3% precision → **0.85-0.87** (if after reranker)

**Why Hybrid:**
- **BM25:** Keyword-based (good for exact terms like "1500-600", "OTP")
- **Semantic:** Meaning-based (good for paraphrases)
- **Combined:** Best of both worlds

**Target Problems:**
- Exact term queries (phone numbers, codes, product names)
- Reduce "meleset sedikit" further (+10% improvement)

**Implementation:**
```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Current vector retriever
vector_retriever = faiss_retriever  # MPNet

# Add BM25
bm25_retriever = BM25Retriever.from_documents(docs)

# Ensemble (65% semantic, 35% keyword)
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.65, 0.35]
)
```

**Only test if:**
- Reranker alone doesn't reach 0.85
- Exact-term queries still failing

---

### **Phase 9C: MMR (Maximal Marginal Relevance)** ⏳ LOWEST PRIORITY

**Goal:** Improve diversity and reduce redundancy

**Duration:** 2 hours

**Expected Impact:** +3-5% recall on multi-doc queries (precision impact minimal)

**Why Last Priority:**
- Only fixes 20% of failures (multi-doc)
- Reranker already addresses 40% of failures
- Can combine with reranker if needed

**What is MMR:**
- Algorithm: Balance relevance vs diversity
- Formula: `MMR = λ × similarity(query, doc) - (1-λ) × max_similarity(doc, selected_docs)`
- Built-in: Langchain FAISS retriever (no extra model needed)

**How it works:**
```
Standard retrieval (k=5):
→ Chunk 1, 2, 3: policy_returns.md (redundant!)
→ Chunk 4, 5: troubleshooting_guide.md

MMR retrieval (k=5, λ=0.5):
→ Chunk 1: policy_returns.md
→ Chunk 2: troubleshooting_guide.md (diverse!)
→ Chunk 3: contact_escalation.md (diverse!)
→ Chunk 4: product_faq.md (diverse!)
→ Chunk 5: policy_returns.md (different section)
```

**Only test if:**
- Multi-doc recall still < 0.70 after reranker
- Time permits

---

### **Phase 9D: Combined Approach (Optional)** ⏳ IF NEEDED

**Goal:** Test if Reranker + BM25 + MMR can reach 0.90 precision

**Approach:**
```
Query → Hybrid Retrieval (BM25 + Semantic) k=10 → MMR (diverse 7) → Rerank (top-3)
```

**Expected Impact:** +10-13% → **0.88-0.91** (90% milestone!)

**Only test if:**
- Reranker alone reaches 0.83-0.85
- Want to reach stretch goal (0.90 precision)
- Time permits

**Config:**
```yaml
retrieval_k: 10
use_hybrid: true
hybrid_weights: [0.65, 0.35]  # Semantic, BM25
use_mmr: true
mmr_lambda: 0.5
use_reranker: true
reranker_top_k: 3
```

---

## 📊 Deliverables Summary

### Phase 9A Outputs (Reranker - PRIORITY):
- ⏳ `z3_core/reranker.py` (BGEReranker class)
- ⏳ Modified `z3_core/rag.py` with reranker support
- ⏳ 1-2 reranker experiments (bge-reranker-base / bge-reranker-v2-m3)
- ⏳ Reranker comparison table
- ⏳ Qualitative analysis of ranking improvements

### Phase 9B Outputs (Hybrid BM25 - Optional):
- ⏳ Modified `z3_core/rag.py` with hybrid retriever support
- ⏳ 1-2 hybrid experiments (weight variations)
- ⏳ Hybrid comparison table

### Phase 9C Outputs (MMR - Optional):
- ⏳ Modified `z3_core/rag.py` with MMR support
- ⏳ 1-3 MMR experiments (λ variations)
- ⏳ MMR comparison table

### Phase 9D Outputs (Combined - Optional):
- ⏳ Combined experiment (Reranker + BM25 + MMR)
- ⏳ Final comparison table (all Phase 9 variants)

### Final Documentation:
- ⏳ `PHASE_9_SUMMARY.md` - Complete Phase 9 analysis
- ⏳ Updated `PROGRESS.md`
- ⏳ Production config decision (Exp6 vs Exp9 winner)

---

## 🎯 Success Criteria

**Phase 9 is successful if:**
1. ✅ At least ONE technique improves precision to ≥ 0.80 (reach target!)
2. ✅ Recall maintained ≥ 0.90
3. ✅ F1 score ≥ 0.82
4. ✅ Identified production-ready configuration

**Minimum success (Phase 9A only):**
- 🎯 Precision ≥ 0.83 (reranker alone, +5% gain)
- 🎯 MRR ≥ 0.97 (better ranking)
- 🎯 Ranking issues reduced from 10% → 3%

**Stretch goals (with combined techniques):**
- 🎯 Precision ≥ 0.85 (Reranker + BM25)
- 🎯 Precision ≥ 0.90 (Reranker + BM25 + MMR) - 90% milestone!
- 🎯 Multi-doc recall ≥ 0.70 (from 0.50)
- 🎯 All categories ≥ 0.75 precision

---

---

## 💡 Key Insights from Phase 8

**What we learned:**
1. ❌ Better embedding (BGE-M3) doesn't always help
2. ❌ Better splitter (Markdown) can make things worse
3. ✅ Simple is often better (MPNet + Recursive + k=3)
4. ✅ 40% of failures are ranking/subsection issues → Reranker promising!
5. ✅ 20% of failures are multi-doc → MMR promising!

**Phase 9 philosophy:**
- Don't change what works (MPNet, Recursive, k optimal)
- Add layers on top (MMR, reranker)
- Focus on known failure patterns
- If it works, it works. If not, accept 0.783 as production.

---

## 🚀 Next Actions (Updated Priority)

**Immediate (Start Phase 9A - Reranker):**
1. ⏳ Install bge-reranker model (`pip install FlagEmbedding`)
2. ⏳ Create `z3_core/reranker.py` (BGEReranker class)
3. ⏳ Modify `z3_core/rag.py` to support reranker
4. ⏳ Create configs for reranker experiments (1-2 variants)
5. ⏳ Run reranker experiments
6. ⏳ Analyze results → **Decision point**

**Short-term (Phase 9B/9C if needed):**
7. If precision < 0.85: Try Hybrid BM25
8. If multi-doc recall low: Try MMR
9. If targeting 0.90: Try combined approach

**Final (Documentation):**
10. Create PHASE_9_SUMMARY.md
11. Update PROGRESS.md
12. Decide production config
13. Deploy or close research

---

**Status:** Ready to start Phase 9A (Reranker) ⏳ HIGHEST PRIORITY
**Expected outcome:** Precision 0.83-0.85 (exceeds 0.80 target!)
**Time investment:** 3-4 hours minimum (reranker only), up to 12 hours if pursuing 0.90 stretch goal
**Confidence:** 80% reranker alone bridges 2.2% gap (based on Claude web analysis)
