# Next Experiments Plan - Phase 1.5 maybe

**Date:** 2025-10-17
**Status:** Planning advanced optimizations beyond baseline experiments

---

## 🎯 Current Status

**Best Configuration (Exp6):**
- Precision: 0.783 (target: 0.80)
- Recall: 0.917 (target: 0.90)
- F1: 0.795 (target: 0.75)
- Config: k=3, MPNet-v2, chunk=500, overlap=50

**Gap to close:** +2.2% precision to reach target 0.80

---

## 📊 Parameter Impact Hierarchy (Research-Based)

### TIER 1: CRITICAL (20-40% precision swing)

#### 1. Embedding Model Quality ⭐⭐⭐
**Impact:** Foundation of retrieval quality
- MiniLM → MPNet: +9% MRR, precision foundation improved
- Bad embedding = garbage retrieval regardless of other tuning

**Proven by our data:**
- Baseline (MiniLM): 0.706 precision
- Exp4/5/6 (MPNet): 0.639-0.783 precision (with different k)

#### 2. Retrieval K Parameter ⭐⭐⭐
**Impact:** Signal-to-noise ratio
- k=6 → k=3: +22% precision (0.639 → 0.783)
- Too high k = dilute relevant docs with noise

**Proven by our data:**
- k=6 (Exp2): 0.539 precision (-24% vs baseline)
- k=4 (Exp5): 0.761 precision
- k=3 (Exp6): 0.783 precision (best)

**Conclusion:** k parameter is 2nd most critical after embedding model

---

### TIER 2: HIGH IMPACT (5-15% precision swing)

#### 3. Reranker (Cross-Encoder) ⭐⭐
**Status:** Not yet tested
**Expected impact:** +5-10% precision
**Effort:** Low (no rebuild, retrieval-time only)

**How it works:**
- Retrieval: Query → Embedding → Find top-10 chunks
- Reranking: Cross-encoder re-scores top-10 → Select best 3
- Cross-encoder is more powerful (scores query+doc together, not separate)

**Why effective:**
- Fixes embedding model mistakes
- Better semantic matching
- Second layer of quality control

**Model to use:**
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (137MB, local, fast)
- `cross-encoder/ms-marco-MiniLM-L-12-v2` (larger, more accurate)

**Example improvement:**
```
Query: "Return barang rusak"

Before reranking (embedding scores):
1. policy_returns.md: 0.75 ✅
2. product_faq.md: 0.72 ❌ (false positive)
3. troubleshooting.md: 0.68 ✅

After reranking (cross-encoder scores):
1. policy_returns.md: 0.92 ✅
2. troubleshooting.md: 0.85 ✅ (promoted!)
3. product_faq.md: 0.45 ❌ (demoted!)
```

---

#### 4. Chunk Size + Overlap ⭐⭐
**Status:** Partially tested (500/50 vs 700/100)
**Impact:** +2-5% precision with optimal settings
**Effort:** High (rebuild vector store)

**Current optimal:** chunk=500, overlap=50
- 40-64% token reduction vs baseline
- Precision maintained or improved

**Further tuning options:**
- chunk=500, overlap=100 (20% overlap - more context continuity)
- chunk=500, overlap=150 (30% overlap - maximum continuity)
- Requires manual inspection of retrieved text quality first

---

#### 5. Splitter Strategy ⭐⭐
**Status:** Not tested (using RecursiveCharacterTextSplitter)
**Expected impact:** +2-5% precision for structured docs
**Effort:** High (rebuild vector store)

**Current splitter:** RecursiveCharacterTextSplitter
- Generic character-based splitting
- Fallback: \n\n → \n → space
- Works okay, but doesn't leverage doc structure

**Better options for our docs:**

**a) MarkdownHeaderTextSplitter** ⭐ RECOMMENDED
- Split by markdown headers (##, ###)
- Preserve semantic structure
- Example: "## Return Policy" becomes one semantic unit
- **Best for:** Our e-commerce markdown docs with clear headers

**b) SemanticChunker** (advanced)
- Split by semantic similarity between sentences
- More expensive (needs to encode every sentence)
- More intelligent boundaries

**Our docs structure:**
```markdown
# Policy Returns
## Eligible Items
## Return Process
## Refund Timeline
```

**With MarkdownHeaderTextSplitter:** Each section (##) becomes logical chunk
**With RecursiveCharacter:** Might split mid-section based on char count

**Recommendation:** Test MarkdownHeaderTextSplitter - expected +3-5% precision

---

### TIER 3: MEDIUM IMPACT (3-8% precision swing)

#### 6. Hybrid Search (BM25 + Semantic) ⭐
**Status:** Not tested
**Expected impact:** +10-20% for specific categories (Payment, Product)
**Effort:** High (rebuild, different indexing)

**What it is:**
- Combine lexical matching (BM25) with semantic search
- BM25: Traditional keyword-based (TF-IDF style)
- Semantic: Current vector similarity

**Why useful:**
- Payment/Product categories: 0.5-0.54 precision (low)
- Some queries need exact keyword match (e.g., "OTP", "Pre-Order")
- MPNet alone struggles with domain-specific terms

**Implementation:**
```python
# Weighted combination
final_score = 0.7 * semantic_score + 0.3 * bm25_score
```

**Expected:** Payment/Product precision → 0.65-0.70 (+15-30%)

---

#### 7. MMR (Maximal Marginal Relevance) ⭐
**Status:** Not tested
**Expected impact:** +5-10% recall for multi-doc queries
**Effort:** Low (retrieval-time only, no rebuild)

**What it is:**
- Diversity-aware retrieval algorithm
- Formula: Score = λ × (similarity to query) - (1-λ) × (similarity to already selected)
- Forces retrieval from different documents

**Problem it solves:**
```
Current behavior (without MMR):
Query: "Return barang tapi penjual tidak respon"
Expected: policy_returns.md + contact_escalation.md
Retrieved: 4 chunks from policy_returns.md only ❌

With MMR (λ=0.7):
Retrieved: 2 chunks from policy_returns.md + 2 from contact_escalation.md ✅
```

**Why useful:**
- Medium/Hard queries need multi-doc (2+ docs)
- Current system tends to retrieve all chunks from single doc
- MMR forces diversity

**Expected:** Medium/Hard recall improvement +5-10%

---

### TIER 4: LOW IMPACT (0-3% swing)

#### 8. Relevance Threshold ❌
**Status:** Tested (0.3 vs 0.5 vs 0.8)
**Impact:** NONE on precision/recall
**Finding:** All similarity scores < 0.8, threshold effectively disabled

**Proven by our data:**
- Exp6 (threshold=0.3): Precision 0.783, Recall 0.917
- Exp7 (threshold=0.5): Precision 0.783, Recall 0.917 (identical!)
- Only affects token count (0.5 retrieves more chunks from same docs)

**Recommendation:** Keep at 0.3 for efficiency, don't waste time re-testing

---

#### 9. Chunk Overlap Fine-tuning
**Status:** Tested (50 vs 100)
**Expected impact:** +1-2% precision (marginal)
**Effort:** High (rebuild)

**Recommendation:** Only test after manual text inspection shows context issues

---

## 🚀 Advanced Embedding Models (Upgrade Options)

### Models Better Than Current MPNet-v2:

| Model | Size | Dimension | Run Local? | Expected Gain vs MPNet | Speed | Best For |
|-------|------|-----------|------------|------------------------|-------|----------|
| **Current: MPNet-v2** | 420MB | 768 | ✅ Yes | Baseline | Fast | General multilingual |
| **bge-m3** ⭐ | 2.2GB | 1024 | ✅ Yes | +5-10% | Medium | SOTA multilingual, dense+sparse hybrid |
| **e5-mistral-7b** | 14GB | 4096 | ⚠️ GPU needed | +10-15% | Slow | LLM-based, very powerful |
| **jina-embeddings-v2** | 550MB | 768 | ✅ Yes | +3-5% | Fast | Long context (8k tokens) |
| **OpenAI text-embedding-3-small** | API | 1536 | ❌ API only | +7-12% | Fast | Commercial, proven quality |
| **OpenAI text-embedding-3-large** | API | 3072 | ❌ API only | +10-15% | Fast | Best quality, expensive |

### Recommendations:

**If local-only (no API):**
- **bge-m3** - Best quality upgrade, still runs on CPU (~30-40s per build)
- **jina-v2** - Faster alternative, good for long contexts

**If okay with API costs:**
- **OpenAI text-embedding-3-small** - Best bang for buck (~$0.02 per 1M tokens)
- Proven better than open-source models

**My vote:** Test **bge-m3** first (free, local, proven SOTA multilingual)

---

## 📋 Proposed Experiment Plan - Phase 3

### Priority 1: Quick Wins (No Rebuild)

**Exp8: Add Reranker to Exp6 (Winner)**
```yaml
embedding_model: MPNet-v2
chunk_size: 500
chunk_overlap: 50
retrieval_k: 3
reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
rerank_top_k: 10  # Retrieve 10, rerank, return top 3
```
**Expected:** Precision 0.83-0.85 (exceed target!)
**Effort:** 1-2 hours (add reranking layer to code)

**Exp9: Add MMR to Exp6**
```yaml
# Same as Exp6 but with MMR
retrieval_k: 5
mmr: True
mmr_lambda: 0.7  # Balance relevance vs diversity
final_k: 3
```
**Expected:** Medium/Hard recall +5-10%, multi-doc queries improved
**Effort:** 1 hour (add MMR parameter)

---

### Priority 2: Better Embedding (Rebuild Required)

**Exp10: Upgrade to bge-m3**
```yaml
embedding_model: BAAI/bge-m3
chunk_size: 500
chunk_overlap: 50
retrieval_k: 3
relevance_threshold: 0.3
```
**Expected:** Precision 0.82-0.85, better semantic understanding
**Effort:** 3-4 hours (download model + rebuild vector store)

**Exp11: bge-m3 + Reranker (Combined Best)**
```yaml
embedding_model: BAAI/bge-m3
retrieval_k: 3
reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
rerank_top_k: 10
```
**Expected:** Precision 0.85-0.88 (significant improvement!)
**Effort:** Build on Exp10

---

### Priority 3: Chunking Strategy (Rebuild Required)

**Exp12: MarkdownHeaderTextSplitter**
```yaml
splitter: MarkdownHeaderTextSplitter
headers_to_split: ["##", "###"]
chunk_size: 500  # Max size after header split
embedding_model: BAAI/bge-m3
retrieval_k: 3
```
**Expected:** Precision +3-5% for structured docs
**Effort:** 2-3 hours (implement new splitter + rebuild)

---

### Priority 4: Advanced Techniques (If Time Permits)

**Exp13: Hybrid Search (BM25 + Semantic)**
```yaml
retrieval_mode: hybrid
bm25_weight: 0.3
semantic_weight: 0.7
embedding_model: BAAI/bge-m3
```
**Expected:** Payment/Product categories +10-20% precision
**Effort:** 4-5 hours (implement BM25 indexing + fusion)

**Exp14: Query Expansion for Hard Queries**
```yaml
# Use LLM to expand/decompose complex queries
query_expansion: True
llm_model: gemini-flash (for query rewriting)
```
**Expected:** Hard query recall +10-20%
**Effort:** 3-4 hours (implement query expansion)

---

## 🎯 Recommended Execution Order

### Week 1: Quick Wins
1. ✅ **Exp8** (Reranker) - Highest ROI, no rebuild
2. ✅ **Exp9** (MMR) - Fix multi-doc queries

### Week 2: Embedding Upgrade
3. ✅ **Exp10** (bge-m3) - Better foundation
4. ✅ **Exp11** (bge-m3 + Reranker) - Combined power

### Week 3: Advanced (Optional)
5. ⚠️ **Exp12** (MarkdownSplitter) - If time permits
6. ⚠️ **Exp13** (Hybrid search) - For problem categories

---

## 🔬 Manual Inspection Plan (Before Chunk Tuning)

**Before adjusting chunk size/overlap, do this:**

1. **Sample failed queries** (low precision cases)
2. **Read retrieved chunks manually**
3. **Check for issues:**
   - Chunks cut mid-sentence?
   - Missing critical context?
   - Too much noise in chunks?
4. **Decision:**
   - If context issues → Increase overlap (50 → 100)
   - If chunks too big → Keep 500 or reduce to 400
   - If chunks good → Don't change!

**Example inspection:**
```
Query: "Sudah bayar tapi status masih menunggu pembayaran"
Expected: troubleshooting_guide.md

Retrieved chunks:
1. policy_returns.md (rank 1) - FALSE POSITIVE
   Content: "...return process within 7 days..." ❌ Not relevant
   Issue: Embedding confused "payment" with "refund"

2. troubleshooting_guide.md (rank 4) - TRUE POSITIVE
   Content: "If payment is deducted but order status..." ✅ Relevant
   Issue: Ranked too low, should be #1

Action: Add reranker to fix ranking issue
```

---

## 📊 Expected Final Results

**If we execute Priority 1-2 (Exp8-11):**

| Metric | Current (Exp6) | After Exp8 (Reranker) | After Exp11 (bge-m3+Reranker) |
|--------|----------------|----------------------|-------------------------------|
| Precision | 0.783 | 0.83-0.85 ✅ | 0.85-0.88 ✅✅ |
| Recall | 0.917 | 0.92-0.93 | 0.93-0.95 |
| F1 | 0.795 | 0.85-0.87 | 0.87-0.90 |
| MRR | 0.950 | 0.96-0.98 | 0.97-0.99 |

**Target achievement:**
- ✅ Precision ≥ 0.80 (exceeded!)
- ✅ Recall ≥ 0.90 (maintained)
- ✅ F1 ≥ 0.75 (exceeded!)

---

## 🎓 Research Rigor Notes

**All experiments must include:**
- ✅ Complete CSV exports (all 8+ experiments)
- ✅ By difficulty breakdown
- ✅ By category breakdown
- ✅ Comparison with all previous experiments
- ✅ Statistical significance testing (if sample size allows)
- ✅ Failed experiments documented (what didn't work & why)

**Documentation standards:**
- Update `EXPERIMENT_RESULTS_ANALYSIS.md` with new findings
- Export all metrics to CSV
- Keep research-grade reproducibility

---

## 🚀 Next Actions

1. **Decide:** Start with Exp8 (Reranker) or Exp10 (bge-m3)?
2. **Implement:** Add reranking layer or upgrade embedding model
3. **Run:** Execute experiment on 30-query golden dataset
4. **Analyze:** Compare with all previous experiments
5. **Document:** Update analysis files and CSV exports
6. **Iterate:** Based on results, proceed to next priority

---

**Status:** Ready to execute Phase 3 experiments
**Recommendation:** Start with Exp8 (Reranker) - quickest path to exceed target 0.80 precision
