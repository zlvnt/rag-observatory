# Phase 8 Roadmap - Qualitative Analysis & Advanced Ablation

**Date:** 2025-10-17 (Updated: 2025-10-19)
**Status:** Phase 8A complete ✅ | Phase 8B-8D planned

---

## 🎯 Phase 8 Overview

**Goal:** Move beyond metrics to understand **WHY** configurations perform differently

**Approach:**
1. **Qualitative Analysis** - Inspect actual retrieved text (not just metrics)
2. **Embedding Model Ablation** - Test if better embedding changes optimal k/threshold patterns
3. **Splitter Ablation** - Test if different chunking strategy improves quality

**Philosophy:**
- Phase 1-7 = Find optimal config (metrics-based) ✅ DONE
- Phase 8 = Understand quality deeply (text-based + ablation) 🔄 IN PROGRESS

---

## 📊 latest Phase Recap (What We Know)

**Metrics-based evaluation complete:**
- 8 experiments (Baseline + Exp1-7)
- Winner: Exp6 (k=3, MPNet, chunk=500) → Precision 0.783, Recall 0.917
- Key finding: k parameter most critical (k=6 fails, k=3/4 optimal)

**Data artifacts:**
- `all_experiments_overview.csv` - Complete metrics comparison
- `experiment_by_difficulty.csv` - Performance by difficulty
- `experiment_by_category.csv` - Performance by category
- Individual experiment reports

**What's missing:**
- ❌ No qualitative inspection of retrieved text
- ❌ No understanding of WHY Exp6 precision = 0.783, not 0.80
- ❌ No validation if better embedding maintains same k patterns

---

## 🗺️ Phase 8 Roadmap

### **Phase 8A: Qualitative Analysis Setup** ✅ COMPLETE

**Goal:** Understand retrieval quality through manual text inspection

**Duration:** 1-2 hours

#### Task 1: Create Qualitative Analysis CSV

**File:** `qualitative_analysis_exp6.csv`

**Structure:**
- 30 rows (all queries from golden dataset)
- Show actual retrieved text from Exp6 (winner)
- Enable manual inspection & pattern detection

**Columns:**
```
- query_id
- difficulty
- category
- query_text
- expected_docs
- exp6_retrieved_docs
- exp6_num_chunks
- exp6_precision
- exp6_recall
- chunk_1_text (first 150-200 chars)
- chunk_2_text (first 150-200 chars)
- chunk_3_text (first 150-200 chars)
- inspection_notes (manual observations)
```

**Example row:**
```csv
ecom_easy_008,easy,payment,"Sudah bayar tapi status masih menunggu pembayaran",troubleshooting_guide.md,"policy_returns.md; troubleshooting_guide.md",3,0.5,1.0,"Kebijakan return berlaku untuk produk yang dibeli dalam...","Jika pembayaran sudah terpotong namun status order masih menunggu...","FAQ produk pre-order dapat ditemukan di halaman...","Issue: policy_returns ranked #1 (WRONG!), troubleshooting #2 (CORRECT). Reranker needed to fix ranking."
```

**Alternative approach (more focused):**
- Option A: All 30 queries (comprehensive)
- Option B: Only failed queries (precision < 1.0 or recall < 1.0) → ~15 queries
- **Recommended:** Option A (full coverage for research completeness)

---

#### Task 2: Manual Inspection & Pattern Detection

**Process:**
1. Read through all 30 query results
2. For each query, inspect retrieved text quality
3. Document observations in `inspection_notes` column

**What to look for:**
- ✅ **Perfect matches:** Relevant text, correct ranking
- ⚠️ **Ranking issues:** Correct doc retrieved but ranked low
- ❌ **False positives:** Irrelevant docs ranked high (keyword match only)
- ❌ **Context issues:** Text cut mid-sentence, missing key info
- ❌ **Missing multi-doc:** Only 1 doc retrieved when 2+ expected

**Example findings:**
```
Pattern 1: Payment queries often retrieve policy_returns.md (keyword "refund" confused with "payment")
→ Suggests: Need reranker or hybrid search

Pattern 2: Multi-doc queries (medium/hard) only retrieve chunks from 1 document
→ Suggests: Need MMR (diversity-aware retrieval)

Pattern 3: Technical terms (OTP, pre-order) sometimes miss exact matches
→ Suggests: Hybrid search (BM25 + semantic)
```

**Output:**
- Completed `qualitative_analysis_exp6.csv` with manual notes
- Summary document: Top 5 patterns identified
- Recommendations for next experiments (reranker, MMR, hybrid search)

---

### **Phase 8B: Embedding Model Ablation Study**

**Goal:** Test if better embedding model changes optimal configuration patterns

**Duration:** 3-4 hours (rebuild vector stores for all configs)

#### Task 3: Select Embedding Model to Test

**Recommended: bge-m3** (BAAI/bge-m3)
- Size: 2.2GB
- Dimension: 1024
- Performance: SOTA multilingual, proven better than MPNet
- Run local: ✅ Yes (CPU-friendly, ~30-40s per index build)

**Alternative options:**
- jina-embeddings-v2 (550MB, faster but less powerful)
- e5-base-v2 (438MB, competitive with MPNet)
- OpenAI text-embedding-3-small (API, commercial)

**Why bge-m3:**
- Best quality among local models
- Hybrid dense+sparse retrieval capability
- Strong multilingual performance (important for Indonesian queries)

---

#### Task 4: Re-run ALL 8 Experiment Configurations with New Embedding

**Critical insight:** This is a **complete ablation study** to isolate embedding impact!

**Experiments to run:**
1. **Baseline_bge:** k=4, threshold=0.8, chunk=700, overlap=100, **bge-m3**
2. **Exp1_bge:** k=4, threshold=0.3, chunk=700, overlap=100, **bge-m3**
3. **Exp2_bge:** k=6, threshold=0.3, chunk=700, overlap=100, **bge-m3**
4. **Exp3_bge:** k=6, threshold=0.3, chunk=500, overlap=50, **bge-m3**
5. **Exp4_bge:** k=6, threshold=0.3, chunk=500, overlap=50, **bge-m3** (same as Exp3_bge - redundant)
6. **Exp5_bge:** k=4, threshold=0.3, chunk=500, overlap=50, **bge-m3**
7. **Exp6_bge:** k=3, threshold=0.3, chunk=500, overlap=50, **bge-m3**
8. **Exp7_bge:** k=3, threshold=0.5, chunk=500, overlap=50, **bge-m3**

**Note:** Exp4 original was same config as Exp3 but with MPNet. With bge-m3, Exp3_bge = Exp4_bge (skip duplicate).

**Total:** 7 experiments (skip Exp4_bge duplicate)

**Research questions:**
- Does k=6 still fail with bge-m3? (test if embedding quality can overcome k problem)
- Does k=3 remain optimal? (validate pattern consistency)
- Does threshold still have no impact? (reconfirm finding)
- What's the precision gain vs MPNet at same k? (quantify embedding impact)

---

#### Task 5: Compare Metrics - bge-m3 vs MPNet

**Create comparison table:**

| Config | k | threshold | chunk | MPNet Precision | bge-m3 Precision | Δ Precision | MPNet Recall | bge-m3 Recall | Δ Recall |
|--------|---|-----------|-------|-----------------|------------------|-------------|--------------|---------------|----------|
| Baseline | 4 | 0.8 | 700 | 0.706 | ??? | ??? | 0.950 | ??? | ??? |
| Exp1 | 4 | 0.3 | 700 | 0.706 | ??? | ??? | 0.950 | ??? | ??? |
| Exp2 | 6 | 0.3 | 700 | 0.539 | ??? | ??? | 0.967 | ??? | ??? |
| Exp5 | 4 | 0.3 | 500 | 0.761 | ??? | ??? | 0.950 | ??? | ??? |
| Exp6 | 3 | 0.3 | 500 | 0.783 | ??? | ??? | 0.917 | ??? | ??? |

**Key analyses:**
1. **Embedding impact at same k:** Does bge-m3 improve precision across ALL k values?
2. **Pattern consistency:** Is k=3/4 still optimal, or does bge-m3 change sweet spot?
3. **k=6 recovery:** Can better embedding "fix" k=6 problem? (unlikely but worth testing)
4. **Absolute gain:** How much precision gain at optimal config (Exp6)?

**Expected outcome:**
- bge-m3 improves precision +5-10% across all configs
- k=3/4 remains optimal (pattern consistent)
- k=6 still fails (embedding can't overcome noise problem)
- Best config: Exp6_bge → Precision ~0.83-0.85

---

#### Task 6: Qualitative Comparison - Text Quality

**Create:** `qualitative_analysis_exp6_bge.csv`

**Same structure as Task 1, but for bge-m3 results**

**Side-by-side comparison:**
- Compare MPNet vs bge-m3 retrieved text for same queries
- Identify: Does bge-m3 retrieve different/better chunks?

**Example comparison:**
```
Query: "OTP tidak masuk ke HP"

MPNet (Exp6):
- Retrieved: troubleshooting_guide.md, product_faq.md
- Text: "Untuk masalah teknis aplikasi...", "FAQ produk dapat dilihat..."
- Issue: product_faq irrelevant (keyword match only)

bge-m3 (Exp6_bge):
- Retrieved: troubleshooting_guide.md, contact_escalation.md
- Text: "Jika OTP tidak diterima, periksa...", "Hubungi customer service untuk..."
- Better: More relevant, better semantic understanding
```

**Output:**
- Qualitative validation of quantitative improvement
- Understanding of **what changed** in retrieval behavior
- Evidence for publication/presentation

---

### **Phase 8C: Splitter Ablation Study**

**Goal:** Test if different chunking strategy improves retrieval quality

**Duration:** 2-3 hours

#### Task 7: Select Splitter to Test

**Recommended: MarkdownHeaderTextSplitter**

**Why:**
- Our docs are Markdown (.md) with clear structure (headers, sections)
- Current RecursiveCharacterTextSplitter ignores structure
- MarkdownHeaderTextSplitter preserves semantic units

**Example difference:**
```markdown
## Return Policy
Customers can return items within 7 days...

## Refund Process
Refunds are processed within 3-5 business days...
```

**RecursiveCharacterTextSplitter (current):**
- Chunks by character count
- Might split: "...within 7 days [SPLIT] ## Refund Process Refunds..."
- Loses section context

**MarkdownHeaderTextSplitter:**
- Chunk 1: "Return Policy\nCustomers can return items within 7 days..."
- Chunk 2: "Refund Process\nRefunds are processed within 3-5 business days..."
- Preserves semantic sections

---

#### Task 8: Run Splitter Experiments

**Use best embedding from Phase 2B** (likely bge-m3)

**Experiments:**
1. **Exp_split_baseline:** RecursiveCharacterTextSplitter (for comparison)
2. **Exp_split_md:** MarkdownHeaderTextSplitter
3. **Exp_split_semantic (optional):** SemanticChunker (if time permits)

**Config (constant):**
```yaml
embedding_model: bge-m3  # Best from Phase 2B
chunk_size: 500
chunk_overlap: 50
retrieval_k: 3  # Optimal from Phase 1
relevance_threshold: 0.3
```

**Only vary:** `splitter` type

**Research question:**
- Does MarkdownHeaderTextSplitter improve precision?
- Do chunks preserve better semantic context?
- Is the rebuild effort worth the gain?

---

#### Task 9: Compare Splitter Impact

**Metrics comparison:**

| Splitter | Precision | Recall | F1 | Notes |
|----------|-----------|--------|----|----|
| RecursiveChar | 0.XXX | 0.XXX | 0.XXX | Baseline |
| MarkdownHeader | 0.XXX | 0.XXX | 0.XXX | Expected +3-5% |
| Semantic (optional) | 0.XXX | 0.XXX | 0.XXX | Expected +2-4% |

**Qualitative comparison:**
- Sample 5-10 queries
- Compare chunk boundaries
- Check: Does MarkdownHeader preserve better context?

**Example:**
```
Query: "Berapa lama refund cair?"

RecursiveChar chunk:
"...return dalam 7 hari. ## Refund Process Ref..."
(Split mid-section, loses context)

MarkdownHeader chunk:
"Refund Process\nRefunds are processed within 3-5 business days after..."
(Complete section, better context)
```

---

### **Phase 8D: Final Optimal Configuration**

**Goal:** Combine all best findings into production-ready config

**Duration:** 1 hour

#### Task 10: Identify Winning Combination

**Based on Phase 8A-8C results:**
- **Best embedding:** bge-m3 (from 8B)
- **Best splitter:** MarkdownHeaderTextSplitter (from 8C, if proven better)
- **Optimal k:** 3 (from Phase 5-6, reconfirmed in 8B)
- **Optimal chunk:** 500, overlap 50 (from Phase 5-6)

**Final config:**
```yaml
domain_name: z3_agent_production_v2
embedding_model: BAAI/bge-m3
splitter: MarkdownHeaderTextSplitter
chunk_size: 500
chunk_overlap: 50
retrieval_k: 3
relevance_threshold: 0.3
```

**Expected performance:**
- Precision: 0.85-0.88 (exceed target 0.80!) ✅
- Recall: 0.92-0.95 (maintain high coverage) ✅
- F1: 0.87-0.90 (balanced excellence) ✅

---

#### Task 11: Create Production Config & Documentation

**Files to create:**
1. `configs/z3_agent_production_v2.yaml` - Final optimal config
2. `PRODUCTION_DEPLOYMENT_GUIDE.md` - How to deploy
3. Update `PROGRESS.md` with Phase 8 completion

**Documentation includes:**
- Final config with rationale for each parameter
- Performance benchmarks (precision, recall, F1, latency, tokens)
- Comparison with Phase 1 winner (Exp6)
- Migration guide from Exp6 to production_v2

---

## 📊 Deliverables Summary

### Phase 8A Outputs: ✅ COMPLETE
- ✅ `qualitative_analysis_exp6.csv` (30 queries with retrieved text + inspection notes)
- ✅ `scripts/create_qualitative_csv.py` (reusable script)
- ✅ `PHASE_8A_SUMMARY.md` - Complete analysis with top 5 failure patterns
- ✅ Root cause diagnosis: Splitter (70%), Embedding (60%), Chunk size (40%)
- ✅ Recommendations prioritized for Phase 8B-8C

### Phase 8B Outputs: ⏳ PLANNED
- ⏳ 7 new experiment results (Baseline_bge through Exp7_bge)
- ⏳ `embedding_ablation_comparison.csv` (MPNet vs bge-m3 metrics)
- ⏳ `qualitative_analysis_exp6_bge.csv` (text quality comparison)
- ⏳ Updated `all_experiments_overview.csv` (now 15 rows total)

### Phase 8C Outputs: ⏳ PLANNED
- ⏳ 2-3 splitter experiments
- ⏳ `splitter_comparison.csv` (metrics & qualitative notes)
- ⏳ Best splitter identified

### Phase 8D Outputs: ⏳ PLANNED
- ⏳ `configs/z3_agent_production_v2.yaml`
- ⏳ `PRODUCTION_DEPLOYMENT_GUIDE.md`
- ⏳ Updated `PROGRESS.md`
- ⏳ Final performance report

---

## 💡 Additional Suggestions

### 1. Pilot Test Before Full Ablation

**Smart approach:** Test new embedding on 2-3 queries first

**Process:**
1. Pick 3 representative queries (1 easy, 1 medium, 1 hard)
2. Run bge-m3 on just these 3
3. Check: Does it retrieve different/better text?
4. If yes → proceed with full 30 queries
5. If no → reconsider, might not be worth rebuild time

**Benefit:** Save time if bge-m3 doesn't show improvement

---

### 2. Prioritize Local Models (Avoid API Costs)

**For Phase 8B embedding selection:**

**Local-only (recommended):**
- ✅ bge-m3 (best quality, SOTA)
- ✅ jina-v2 (faster, good quality)
- ✅ e5-base-v2 (competitive with MPNet)

**API (only if local insufficient):**
- ⚠️ OpenAI text-embedding-3-small (costs ~$0.02 per 1M tokens)

**My vote:** Start with bge-m3 (local, free, proven best)

---

### 3. Focus on Failed Queries in Qualitative Analysis

**Alternative to inspecting all 30 queries:**

**Focused approach:**
- Only inspect queries where precision < 1.0 or recall < 1.0
- From 30 queries, ~15 have issues
- Deeper analysis on problem cases

**Benefits:**
- More efficient use of time
- Focus on actionable insights
- Easier to identify patterns

**Trade-off:**
- Miss insights from "perfect" queries (why they work)
- Less comprehensive

**Recommendation:**
- Start with failed queries (quick wins)
- If time permits, inspect all 30 (research completeness)

---

### 4. Document Failure Patterns for Future Work

**In qualitative CSV, highlight specific issues:**

**Pattern categories:**
- 🔴 **Ranking issue:** Correct doc retrieved but ranked low
- 🟠 **Keyword match:** Irrelevant doc matches keywords only (no semantic understanding)
- 🟡 **Context loss:** Text cut mid-sentence, missing key info
- 🟢 **Multi-doc failure:** Only 1 doc retrieved when 2+ expected
- 🔵 **Perfect:** Correct retrieval and ranking

**Use for:**
- Prioritize next experiments (reranker for ranking, MMR for multi-doc, etc.)
- Publication/presentation (show concrete examples)
- Training data for future fine-tuning

---

### 5. Consider Separate CSVs for Clarity

**Instead of one giant CSV with many columns:**

**Option A:** Single CSV with Exp6 + bge-m3 side-by-side
- Pros: Easy comparison
- Cons: Wide CSV, hard to read

**Option B (chosen):** ✅ Separate CSVs
- `qualitative_analysis_exp6.csv` (MPNet only) ✅ CREATED
- `qualitative_analysis_exp6_bge.csv` (bge-m3 only) - will create in Phase 8B
- Pros: Cleaner, easier to read
- Cons: Need to open 2 files for comparison

**Decision:** Using Option B (separate files for clarity)

---

### 6. Track Time Investment vs Gain

**For each phase, document:**
- Time spent (hours)
- Precision gain (%)
- Worth it? (yes/no)

**Example:**
```
Phase 8B (bge-m3 ablation):
- Time: 4 hours
- Gain: +5% precision (0.783 → 0.83)
- Worth it: YES (1.25% gain per hour)

Phase 8C (MarkdownSplitter):
- Time: 3 hours
- Gain: +2% precision (0.83 → 0.85)
- Worth it: MAYBE (0.67% gain per hour, diminishing returns)
```

**Use for:**
- Prioritize future optimizations
- Publication (show effort vs impact)
- Decision-making (stop when gains too small)

---

## 🎯 Success Criteria

**Phase 8 is successful if:**
1. ✅ Qualitative analysis reveals actionable patterns (top 5 issues identified)
2. ✅ bge-m3 improves precision by +5% or more vs MPNet
3. ✅ Final config exceeds target precision 0.80 (reach 0.85+)
4. ✅ Pattern consistency confirmed (k=3/4 remains optimal with bge-m3)
5. ✅ Production-ready config documented and deployable

**Stretch goals:**
- 🎯 Precision ≥ 0.85
- 🎯 Recall ≥ 0.92
- 🎯 F1 ≥ 0.87
- 🎯 All categories ≥ 0.75 precision (fix Payment/Product)

---

## 📅 Estimated Timeline

**Total duration:** 1-2 days (8-12 hours)

**Breakdown:**
- Phase 8A (Qualitative): ✅ 1-2 hours COMPLETE
- Phase 8B (Embedding ablation): ⏳ 3-4 hours
- Phase 8C (Splitter ablation): ⏳ 2-3 hours
- Phase 8D (Final config): ⏳ 1 hour
- Buffer for analysis/documentation: 1-2 hours

**Can be parallelized:**
- Manual inspection (8A) can happen while experiments run (8B)
- CSV creation can happen while vector stores rebuild

---

## 🚀 Next Actions

**Completed (2025-10-19):**
1. ✅ Created `qualitative_analysis_exp6.csv` with retrieved text
2. ✅ Generated reusable script `scripts/create_qualitative_csv.py`
3. ✅ Manual inspection complete (sampled queries with detailed notes)
4. ✅ Created `PHASE_8A_SUMMARY.md` with complete analysis
5. ✅ Identified failure patterns: Context Cutting (40%), Meleset Sedikit (30%), Multi-doc (20%)
6. ✅ Updated PROGRESS.md and PHASE_8_ROADMAP.md

**Immediate (Next):**
7. ⏳ Download bge-m3 model (2.2GB, prepare for Phase 8B)
8. ⏳ Setup configs for bge-m3 ablation study

**Short-term (This Week):**
6. Run bge-m3 ablation study (7 experiments across all configs)
7. Compare MPNet vs bge-m3 metrics & text quality
8. Run splitter experiments (if bge-m3 proves worthy)

**Medium-term (Next Week):**
9. Create final production config v2
10. Document all findings
11. Update EXPERIMENT_RESULTS_ANALYSIS.md with Phase 8 results
12. Prepare for advanced techniques (reranker, MMR, hybrid search - if needed)

---

**Status:** Phase 8A complete ✅ | Ready for Phase 8B
**Next task:** Download bge-m3 model and begin embedding ablation study
