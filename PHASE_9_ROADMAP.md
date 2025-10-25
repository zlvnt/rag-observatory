# Phase 9 Roadmap - Advanced Retrieval Techniques

**Date:** 2025-10-25
**Status:** Ready to start | Phase 8 complete (basic optimization exhausted)

---

## 🎯 Phase 9 Overview

**Goal:** Bridge the 2.2% precision gap (0.783 → 0.80+) using advanced retrieval techniques

**Current State:**
- Winner: Exp6 (MPNet + RecursiveCharacterTextSplitter + k=3)
- Precision: 0.783 (2.2% gap to 0.80 target)
- Recall: 0.917 (exceeds target)
- F1: 0.795 (exceeds target)

**Phase 8 Findings (What We Exhausted):**
- ✅ Embedding optimization complete (MPNet optimal, BGE-M3 failed)
- ✅ Splitter optimization complete (Recursive optimal, Markdown failed)
- ✅ k parameter optimized (k=3 best)
- ✅ Threshold optimized (0.3 best)

**Remaining Failure Patterns (from Phase 8A):**
1. 🔴 **Ranking issues (10%)** - Correct doc retrieved but ranked low
2. 🟠 **"Meleset sedikit" (30%)** - Right doc, wrong subsection ranked higher
3. 🟡 **Multi-doc failures (20%)** - Only 1 doc retrieved when 2+ expected
4. 🟢 **Context cutting (40%)** - Splitter cuts mid-section (cannot fix with current tools)

**Phase 9 Strategy:**
- Focus on techniques that don't require changing basic parameters
- Add post-processing layers (reranking, diversity)
- No more embedding/splitter experiments

---

## 🗺️ Phase 9 Roadmap

### **Phase 9A: MMR (Maximal Marginal Relevance)** ⏳ NEXT

**Goal:** Improve diversity and reduce redundancy in retrieved chunks

**Duration:** 2-3 hours

**What is MMR:**
- Algorithm: Balance relevance vs diversity
- Formula: `MMR = λ × similarity(query, doc) - (1-λ) × max_similarity(doc, selected_docs)`
- Built-in: Langchain FAISS retriever (no extra model needed)
- No download required

**Target Problem:**
- 🟡 **Multi-doc failures (20%)** - Retrieve chunks from different docs
- Avoid redundant chunks from same section

**How it works:**
```
Standard retrieval (k=5):
→ Chunk 1: policy_returns.md (section A)
→ Chunk 2: policy_returns.md (section A, redundant!)
→ Chunk 3: policy_returns.md (section B)
→ Chunk 4: policy_returns.md (section C)
→ Chunk 5: troubleshooting_guide.md

MMR retrieval (k=5, λ=0.5):
→ Chunk 1: policy_returns.md (section A)
→ Chunk 2: troubleshooting_guide.md (diverse!)
→ Chunk 3: contact_escalation.md (diverse!)
→ Chunk 4: policy_returns.md (section B, different from chunk 1)
→ Chunk 5: product_faq.md (diverse!)
```

**Implementation:**
- Modify `z3_core/rag.py` to use `max_marginal_relevance_search()`
- Test different λ values: 0.3, 0.5, 0.7
  - λ=1.0: Pure relevance (standard retrieval)
  - λ=0.0: Pure diversity (not useful)
  - λ=0.5: Balanced (recommended start)

**Experiments:**
1. **Exp9a_mmr_03:** λ=0.3 (high diversity)
2. **Exp9a_mmr_05:** λ=0.5 (balanced)
3. **Exp9a_mmr_07:** λ=0.7 (high relevance)

**Config (constant):**
```yaml
embedding_model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
chunk_size: 500
chunk_overlap: 50
retrieval_k: 5  # Higher k for MMR (need candidates to diversify)
relevance_threshold: 0.3
mmr_lambda: 0.5  # Vary this
```

**Expected Results:**
- Precision: 0.783 → 0.80-0.82 (+2-4%)
- Recall: Maintain 0.92+ (more docs covered)
- Multi-doc queries: Improved recall (from 0.50 → 0.70+)

**Success Criteria:**
- Multi-doc recall improvement ≥ +15%
- Overall precision ≥ 0.80 (reach target!)
- No significant precision drop on easy queries

---

#### Task 1: Implement MMR Retrieval

**Modify:** `z3_core/rag.py`

**Current code:**
```python
def get_context(retriever, query: str, k: int = 4) -> tuple:
    docs = retriever.invoke(query)
    # ...
```

**New code:**
```python
def get_context(retriever, query: str, k: int = 4, use_mmr: bool = False,
                mmr_lambda: float = 0.5) -> tuple:
    if use_mmr:
        docs = retriever.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=k*3,  # Fetch 3x candidates for diversity selection
            lambda_mult=mmr_lambda
        )
    else:
        docs = retriever.invoke(query)
    # ...
```

**Parameters:**
- `fetch_k`: Number of candidates to fetch before MMR filtering (recommend k*3)
- `lambda_mult`: Balance between relevance (1.0) and diversity (0.0)

---

#### Task 2: Create MMR Test Configs

**Create folder:** `configs/experiments_phase9a/`

**Files:**
1. `z3_agent_exp9a_mmr_03.yaml`
2. `z3_agent_exp9a_mmr_05.yaml`
3. `z3_agent_exp9a_mmr_07.yaml`

**Example config:**
```yaml
domain_name: z3_agent_exp9a_mmr_05
knowledge_base_dir: docs/
vector_store_dir: data/vector_stores/z3_agent_exp6/  # Reuse Exp6 index

embedding_model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
chunk_size: 500
chunk_overlap: 50
retrieval_k: 5
relevance_threshold: 0.3

# MMR parameters
use_mmr: true
mmr_lambda: 0.5
mmr_fetch_k: 15  # 3x retrieval_k
```

---

#### Task 3: Run MMR Experiments

**Execute:**
```bash
python runners/test_runner.py --domain z3_agent_exp9a_mmr_03 --output results/exp9a_mmr_03/
python runners/test_runner.py --domain z3_agent_exp9a_mmr_05 --output results/exp9a_mmr_05/
python runners/test_runner.py --domain z3_agent_exp9a_mmr_07 --output results/exp9a_mmr_07/
```

**Compare with baseline:**
- Exp6 (k=3, no MMR): Precision 0.783
- Exp9a variants (k=5, MMR): Target precision 0.80+

---

#### Task 4: Analyze MMR Results

**Metrics to compare:**

| Config | λ | k | Precision | Recall | F1 | Multi-doc Recall | Tokens/Query |
|--------|---|---|-----------|--------|----|--------------------|--------------|
| Exp6 (baseline) | - | 3 | 0.783 | 0.917 | 0.795 | 0.50 | 211 |
| Exp9a_mmr_03 | 0.3 | 5 | ??? | ??? | ??? | ??? | ??? |
| Exp9a_mmr_05 | 0.5 | 5 | ??? | ??? | ??? | ??? | ??? |
| Exp9a_mmr_07 | 0.7 | 5 | ??? | ??? | ??? | ??? | ??? |

**Key analyses:**
1. Does MMR improve multi-doc recall?
2. Does diversity hurt precision on easy queries?
3. What's the optimal λ value?
4. Token efficiency impact (k=5 vs k=3)

---

### **Phase 9B: bge-reranker (Cross-Encoder Reranking)** ⏳ PLANNED

**Goal:** Improve ranking accuracy using neural cross-encoder model

**Duration:** 3-4 hours (includes model download)

**What is bge-reranker:**
- Model: BAAI/bge-reranker-v2-m3 (1.5GB) or BAAI/bge-reranker-base (600MB)
- Type: Cross-encoder (not bi-encoder like MPNet)
- Input: (query, document) pairs → Output: relevance score (0-1)
- Much more accurate than bi-encoder embeddings

**Target Problem:**
- 🔴 **Ranking issues (10%)** - Fix incorrect ranking
- 🟠 **"Meleset sedikit" (30%)** - Prefer correct subsection over nearby text
- **Total 40% of failures!**

**How it works:**
```
Standard retrieval (k=7):
1. policy_returns.md (score: 0.85) ← WRONG section
2. troubleshooting_guide.md (score: 0.83) ← CORRECT but ranked low
3. product_faq.md (score: 0.80)
...

After reranking (top-3):
1. troubleshooting_guide.md (score: 0.95) ← CORRECT, now ranked #1!
2. policy_returns.md (score: 0.72) ← Dropped
3. contact_escalation.md (score: 0.68)
```

**Implementation:**
- Retrieve k=7 candidates (MPNet bi-encoder)
- Rerank with bge-reranker (cross-encoder)
- Return top-3 for LLM

**Why k=7 → top-3:**
- Reranker needs more candidates to choose from
- k=3 too few for reranker to work effectively
- Cross-encoder slower, so filter down to top-3

---

#### Task 5: Install bge-reranker

**Install FlagEmbedding:**
```bash
pip install FlagEmbedding
```

**Download model (auto on first run):**
```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
# Or smaller: 'BAAI/bge-reranker-base'
```

**Model size:**
- bge-reranker-v2-m3: ~1.5GB (best quality, multilingual)
- bge-reranker-base: ~600MB (good quality, faster)

**Recommendation:** Start with `bge-reranker-base` (smaller, faster)

---

#### Task 6: Implement Reranker Pipeline

**Create:** `z3_core/reranker.py`

```python
from FlagEmbedding import FlagReranker
from typing import List, Tuple
from langchain.schema import Document

class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", use_fp16: bool = True):
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16)

    def rerank(self, query: str, documents: List[Document], top_k: int = 3) -> List[Document]:
        """
        Rerank documents using cross-encoder model.

        Args:
            query: User query
            documents: List of retrieved documents
            top_k: Number of top documents to return

        Returns:
            Top-k reranked documents
        """
        # Prepare pairs for reranker
        pairs = [[query, doc.page_content] for doc in documents]

        # Get scores from reranker
        scores = self.reranker.compute_score(pairs)

        # Sort by score (descending)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return [doc for doc, score in doc_score_pairs[:top_k]]
```

**Modify:** `z3_core/rag.py`

```python
def get_context(retriever, query: str, k: int = 4, use_reranker: bool = False,
                reranker_top_k: int = 3) -> tuple:
    # Retrieve more candidates if using reranker
    retrieve_k = k * 2 if use_reranker else k
    docs = retriever.invoke(query, k=retrieve_k)

    # Rerank if enabled
    if use_reranker:
        from z3_core.reranker import BGEReranker
        reranker = BGEReranker()
        docs = reranker.rerank(query, docs, top_k=reranker_top_k)

    # ... rest of context processing
```

---

#### Task 7: Create Reranker Test Configs

**Create folder:** `configs/experiments_phase9b/`

**Files:**
1. `z3_agent_exp9b_reranker_base.yaml` (bge-reranker-base)
2. `z3_agent_exp9b_reranker_v2.yaml` (bge-reranker-v2-m3, if base succeeds)

**Example config:**
```yaml
domain_name: z3_agent_exp9b_reranker_base
knowledge_base_dir: docs/
vector_store_dir: data/vector_stores/z3_agent_exp6/  # Reuse Exp6 index

embedding_model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
chunk_size: 500
chunk_overlap: 50
retrieval_k: 7  # Retrieve more candidates
relevance_threshold: 0.3

# Reranker parameters
use_reranker: true
reranker_model: BAAI/bge-reranker-base
reranker_top_k: 3  # Final number to return
```

---

#### Task 8: Run Reranker Experiments

**Execute:**
```bash
python runners/test_runner.py --domain z3_agent_exp9b_reranker_base --output results/exp9b_reranker_base/
```

**Expected results:**
- Precision: 0.783 → **0.83-0.85** (+5-7%)
- Recall: Maintain 0.92+
- Ranking: Correct doc should rank #1 more often
- MRR: 0.950 → 0.97+ (better ranking)

---

#### Task 9: Compare Reranker vs Baseline

**Metrics comparison:**

| Config | Retrieval | Reranker | Precision | Recall | F1 | MRR | Tokens/Query |
|--------|-----------|----------|-----------|--------|----|----|--------------|
| Exp6 | MPNet k=3 | None | 0.783 | 0.917 | 0.795 | 0.950 | 211 |
| Exp9b_base | MPNet k=7 | bge-base | ??? | ??? | ??? | ??? | ??? |
| Exp9b_v2 | MPNet k=7 | bge-v2-m3 | ??? | ??? | ??? | ??? | ??? |

**Qualitative check:**
- Sample 5-10 failed queries from Exp6
- Check: Does reranker fix ranking issues?
- Example queries:
  - "Sudah bayar tapi status masih menunggu" (ranking issue in Exp6)
  - "OTP tidak masuk ke HP" (wrong doc ranked #1 in Exp6)

---

### **Phase 9C: Combined Approach (Optional)** ⏳ PLANNED

**Goal:** Test if MMR + Reranker can be combined

**Approach:**
```
Query → Retrieve k=10 (MPNet) → MMR (diverse 7) → Rerank (top-3)
```

**Hypothesis:**
- MMR ensures diversity (different docs)
- Reranker ensures accuracy (correct ranking)
- Combined: Best of both worlds

**Only test if:**
- Both 9A and 9B show improvement
- Time permits

**Config:**
```yaml
retrieval_k: 10
use_mmr: true
mmr_lambda: 0.5
mmr_fetch_k: 30
mmr_return_k: 7
use_reranker: true
reranker_top_k: 3
```

---

## 📊 Deliverables Summary

### Phase 9A Outputs:
- ⏳ Modified `z3_core/rag.py` with MMR support
- ⏳ 3 MMR experiments (λ=0.3, 0.5, 0.7)
- ⏳ MMR comparison table (metrics by λ)
- ⏳ Best λ value identified

### Phase 9B Outputs:
- ⏳ `z3_core/reranker.py` (BGEReranker class)
- ⏳ Modified `z3_core/rag.py` with reranker support
- ⏳ 1-2 reranker experiments (base/v2)
- ⏳ Reranker comparison table
- ⏳ Qualitative analysis of ranking improvements

### Phase 9C Outputs (Optional):
- ⏳ Combined MMR + Reranker experiment
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

**Stretch goals:**
- 🎯 Precision ≥ 0.85
- 🎯 Multi-doc recall ≥ 0.70 (from 0.50)
- 🎯 MRR ≥ 0.97
- 🎯 All categories ≥ 0.75 precision

---

## 📅 Estimated Timeline

**Total duration:** 1-2 days (5-7 hours)

**Breakdown:**
- Phase 9A (MMR): 2-3 hours
  - Implementation: 1 hour
  - Testing (3 experiments): 1.5 hours
  - Analysis: 0.5 hour

- Phase 9B (Reranker): 3-4 hours
  - Model download: 30 min
  - Implementation: 1.5 hours
  - Testing (1-2 experiments): 1.5 hours
  - Analysis: 0.5 hour

- Phase 9C (Combined): 1 hour (optional)

- Documentation: 1 hour

**Can be sequential:**
- If MMR reaches target (0.80+) → Skip Phase 9B
- If MMR fails → Proceed to Phase 9B
- If both fail individually → Try Phase 9C

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

## 🚀 Next Actions

**Immediate (Start Phase 9A):**
1. ⏳ Modify `z3_core/rag.py` to support MMR
2. ⏳ Create configs for MMR experiments (3 variants)
3. ⏳ Run MMR experiments
4. ⏳ Analyze results

**Short-term (Phase 9B if needed):**
5. Install bge-reranker model
6. Implement reranker pipeline
7. Run reranker experiments
8. Compare with baseline

**Medium-term (Finalize):**
9. Create PHASE_9_SUMMARY.md
10. Update PROGRESS.md
11. Decide production config
12. Deploy or move to other advanced techniques

---

**Status:** Ready to start Phase 9A (MMR) ⏳
**Expected outcome:** Precision 0.80+ (reach target!) or accept 0.783 as production ceiling
**Time investment:** 5-7 hours total (much less than Phase 8B+8C = 9 hours with 0% gain)
