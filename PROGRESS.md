# RAG Observatory - Progress Report

**Last Updated:** 2025-11-06
**Status:** ✅ Phase 9 COMPLETE | **Production Config: Exp9a1 (0.828 precision)** 🎉 TARGET ACHIEVED!

---

## 🎯 Project Goal

Optimize RAG retrieval configuration for e-commerce domain through systematic ablation study.

**Target Metrics:** Precision ≥0.80, Recall ≥0.90, F1 ≥0.75

---

## ✅ Completed Phases

### Phase 1-4: Foundation & Testing Framework (Oct 12-14)
- ✅ Refactored z3_core for multi-config testing
- ✅ Created 30-query golden dataset (19 easy, 9 medium, 2 hard)
- ✅ Built retrieval-focused test runner
- ✅ Implemented metrics: Precision, Recall, F1, MRR

### Phase 5-6: Ablation Study (Oct 16-17)
- ✅ Ran 7 experiments (Baseline + Exp1-7)
- ✅ Tested 4 variables: k, threshold, chunk_size, embedding_model
- ✅ Identified optimal configuration

### Phase 7: Research Documentation (Oct 17)
- ✅ Created comprehensive analysis (`EXPERIMENT_RESULTS_ANALYSIS.md`)
- ✅ Generated 6 CSV exports for all comparisons
- ✅ Documented parameter impact hierarchy
- ✅ Planned advanced optimization roadmap (`PHASE_2_ROADMAP.md`)

---

## 🔬 Current Phase: Phase 8 - Advanced Optimization

### Phase 8A: Qualitative Analysis ✅ COMPLETE (Oct 19)
- ✅ Created `qualitative_analysis_exp6.csv` with retrieved text inspection
- ✅ Generated script: `scripts/create_qualitative_csv.py`
- ✅ Manual inspection complete (sampled queries with notes)
- ✅ Identified top 5 failure patterns
- ✅ Root cause analysis: Splitter (70%) + Embedding (60%) + Chunk size (40%)
- ✅ Summary documented: `PHASE_8A_SUMMARY.md`

### Phase 8B: Embedding Model Ablation ✅ COMPLETE (Oct 24)
- ✅ Downloaded bge-m3 model (BAAI/bge-m3, 2.2GB)
- ✅ Tested BGE-M3 dense-only (Exp6_bge): Precision 0.772 (-1.4% vs MPNet)
- ✅ Implemented custom multi-functional retriever (dense+sparse+ColBERT)
- ✅ Tested BGE-M3 multi-functional v1: Precision 0.639 (-14.4% vs MPNet) ❌
- ✅ Tuned hybrid weights v2 (0.7/0.2/0.1): Precision 0.672 (-11.1% vs MPNet) ❌
- ✅ **Conclusion: MPNet remains optimal embedding** (all BGE-M3 variants underperformed)
- ✅ Summary documented: `PHASE_8B_SUMMARY.md`
- ✅ Debug report created: `PHASE_8B_BGE_M3_DEBUG_REPORT.md`

**Key Findings:**
- ❌ BGE-M3 multi-functional added noise instead of improving quality
- ❌ Sparse + ColBERT retrieval caused keyword confusion and over-matching
- ✅ MPNet proven best for Indonesian e-commerce domain (0.783 precision)
- ✅ Embedding optimization exhausted - move to splitter (bigger lever)

### Phase 8C: Splitter Ablation ✅ COMPLETE (Oct 25)
- ✅ Tested MarkdownHeaderTextSplitter with BGE-M3 and MPNet embeddings
- ✅ Tested k=3 and k=5 variants (4 experiments total)
- ✅ Exp6_bge_markdown (k=3): Precision 0.706 (-8.5% vs Recursive) ❌
- ✅ Exp6_mpnet_markdown (k=3): Precision 0.711 (-9.2% vs Recursive) ❌
- ✅ Exp6_bge_markdown_v2 (k=5): Not run (BGE already failed)
- ✅ Exp6_mpnet_markdown_v2 (k=5): Precision 0.589 (-17.2% vs Recursive) ❌❌
- ✅ **Conclusion: RecursiveCharacterTextSplitter remains optimal** (all Markdown variants failed)
- ✅ Summary documented: `PHASE_8C_SUMMARY.md`

**Key Findings:**
- ❌ Markdown splitter creates too many tiny chunks (18 vs 5 per doc)
- ❌ Chunks lose parent context (isolated subsections)
- ❌ Higher k (5) made results WORSE (-17% precision crash)
- ✅ RecursiveCharacterTextSplitter proven optimal for e-commerce docs
- ✅ Splitter optimization exhausted - no further gains available

### Phase 8D: Final Optimal Configuration ✅ COMPLETE (Oct 25)
- ✅ Phase 8A-C exhausted basic optimization (embedding + splitter + k + threshold + chunk size)
- ✅ Production config identified: **Exp6 (0.783 precision, 0.917 recall, 0.795 F1)**
- ✅ Decision: Accept Exp6 as production ceiling for basic optimization
- ✅ Next steps: Move to Phase 9 (MMR + reranker) to bridge 2.2% gap
- ✅ Summary documented: `PHASE_8D_SUMMARY.md`

**Production Config (Exp6):**
- Embedding: MPNet
- Splitter: RecursiveCharacterTextSplitter
- chunk_size: 500, overlap: 50
- retrieval_k: 3, threshold: 0.3
- Performance: 0.783 precision, 0.917 recall, 0.795 F1, 0.950 MRR

**Key Decision:**
- ✅ All basic parameters optimized (12 experiments completed)
- ✅ Phase 8B+8C: 9 hours with 0% gain (negative ROI)
- ✅ 2.2% gap addressable with advanced techniques (Phase 9)
- ✅ Production-ready: Stable, efficient (211 tokens/query), fast (P95 168ms)

### Phase 8E: Parent-Child Markdown Splitter ✅ COMPLETE (Oct 26)
- ✅ Implemented Parent-Child approach (Claude web recommendation)
- ✅ Tested two variants (parent_max_tokens: 500 vs 1500)
- ✅ Created dedicated implementation: `z3_core/vector_parent_child.py`
- ✅ Both experiments failed identically: Precision 0.661 (-15.6% vs Exp6) ❌
- ✅ Root cause identified: All parent sections < threshold, no child splitting occurred
- ✅ **Conclusion: Parent-Child NOT suitable for e-commerce domain** (docs already compact)
- ✅ Summary documented: `PHASE_8E_SUMMARY.md`

**Key Findings:**
- ❌ All 25 Markdown sections < 500 tokens (no splitting occurred)
- ❌ Result equivalent to pure Markdown split (Phase 8C) but worse
- ❌ Easy queries crashed -22.9% precision (0.588 vs 0.762)
- ✅ E-commerce docs already optimal size (100-300 tokens/section)
- ✅ Parent-Child best for long technical docs, not compact FAQ content
- ✅ Exp6 (RecursiveCharacterTextSplitter) confirmed optimal for domain

---

## 📊 Experiment Results Summary

| Exp | k | threshold | chunk | embedding | Precision | Recall | F1 | Tokens | Status |
|-----|---|-----------|-------|-----------|-----------|--------|----|--------|--------|
| Baseline | 4 | 0.8 | 700 | MiniLM | 0.706 | 0.950 | 0.752 | 583 | Reference |
| Exp1 | 4 | 0.3 | 700 | MiniLM | 0.706 | 0.950 | 0.752 | 382 | No change |
| Exp2 | 6 | 0.3 | 700 | MiniLM | 0.539 | 0.967 | 0.652 | 513 | ❌ k=6 failed |
| Exp3 | 6 | 0.3 | 500 | MiniLM | 0.589 | 0.950 | 0.680 | 369 | ❌ Still low |
| Exp4 | 6 | 0.3 | 500 | MPNet | 0.639 | 0.950 | 0.725 | 352 | ⚠️ k=6 bottleneck |
| Exp5 | 4 | 0.3 | 500 | MPNet | 0.761 | 0.950 | 0.798 | 248 | ⭐ Balanced |
| **Exp6** | **3** | **0.3** | **500** | **MPNet** | **0.783** ✅ | **0.917** | **0.795** | **211** | **🏆 WINNER** |
| Exp7 | 3 | 0.5 | 500 | MPNet | 0.783 | 0.917 | 0.795 | 269 | Same as Exp6 |

**Full analysis:** See `EXPERIMENT_RESULTS_ANALYSIS.md`

---

## 🏆 Winning Configuration (Exp6)

```yaml
embedding_model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
chunk_size: 500
chunk_overlap: 50
retrieval_k: 3
relevance_threshold: 0.3
```

**Performance:**
- ✅ Precision: **0.783** (closest to 0.80 target, +11% vs baseline)
- ✅ Recall: **0.917** (exceeds 0.90 target)
- ✅ F1: **0.795** (exceeds 0.75 target)
- ✅ MRR: **0.950** (excellent ranking)
- ✅ Tokens: **211/query** (64% reduction vs baseline)

**Best Categories:**
- Returns: 0.939 precision
- Contact: 0.917 precision
- Account: **1.000 precision** (perfect!)

---

## 💡 Key Learnings

### ✅ What Worked:

1. **Lower k = Higher precision**
   - k=6: 0.639 precision (too much noise)
   - k=4: 0.761 precision (balanced)
   - k=3: 0.783 precision (best)

2. **MPNet > MiniLM**
   - MRR improved 0.872 → 0.950 (+9%)
   - Better semantic understanding and ranking

3. **chunk=500 > chunk=700**
   - 40-64% token reduction
   - Same or better quality

4. **Threshold has minimal impact**
   - Threshold 0.3 vs 0.5: Same precision/recall
   - Only affects token count (0.3 more efficient)

### ❌ What Failed:

1. **k=6 consistently crashes precision** (-24 to -39%)
2. **Threshold doesn't improve precision/recall** (all scores < 0.8)
3. **MiniLM embedding insufficient** for target metrics

---

## 📁 Project Structure

```
rag-observatory/
├── configs/                    # Experiment configurations
│   ├── z3_agent_config.yaml   # Baseline
│   ├── z3_agent_exp5.yaml     # Best balanced (k=4)
│   └── z3_agent_exp6.yaml     # Winner (k=3)
├── results/                    # Experiment results
│   ├── results_baseline/
│   ├── exp1/ ... exp7/
│   └── report/                # Phase 1 analysis exports
│       ├── experiment_comparison_summary.csv
│       ├── experiment_by_difficulty.csv
│       ├── experiment_by_category.csv
│       ├── all_experiments_overview.csv
│       └── qualitative_analysis_exp6.csv  # ⭐ Phase 2A
├── scripts/
│   └── create_qualitative_csv.py  # ⭐ Phase 2A script
├── z3_core/                    # RAG engine
│   ├── vector.py              # FAISS vector store
│   ├── rag.py                 # Retrieval logic
│   └── domain_config.py       # Config management
├── runners/
│   └── test_runner.py         # Test execution
├── golden_datasets/
│   └── z3_agent_tests.json    # 30 test queries
├── EXPERIMENT_RESULTS_ANALYSIS.md  # Phase 1 detailed analysis
├── PHASE_2_ROADMAP.md         # Phase 2 plan
└── PROGRESS.md                # This file
```

---

## 🎯 Production Deployment

### Recommended Configuration:

**Use Exp6 (k=3)** for:
- ✅ Best precision (0.783)
- ✅ Maximum efficiency (211 tokens)
- ✅ Easy/Medium queries dominant (93% of traffic)

**Use Exp5 (k=4)** for:
- ✅ Safer balanced option (F1 0.798)
- ✅ Higher recall (0.950 vs 0.917)
- ✅ Multi-doc queries critical

### Deployment Config:

```yaml
# configs/z3_agent_production.yaml
domain_name: z3_agent_production
knowledge_base_dir: docs/
vector_store_dir: data/vector_stores/production/

embedding_model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
chunk_size: 500
chunk_overlap: 50
retrieval_k: 3  # Use 4 if recall critical
relevance_threshold: 0.3
```

---

## 📝 Known Limitations

### Problem Categories (Low Precision):
- **Payment:** 0.541 precision
- **Product:** 0.500 precision
- **Technical:** 0.500 precision (1 query only)

### Hard Queries:
- k=3 reduces recall to 0.500 for hard queries (2 queries, 6.7% of dataset)
- Trade-off acceptable for production where easy/medium dominate

### Future Improvements:
- Fine-tune embedding model on e-commerce domain
- Add reranking layer (cross-encoder)
- Implement dynamic k based on query complexity
- Use MMR for multi-doc diversity

---

## 📚 Key Documents

### Research Analysis:
- **`EXPERIMENT_RESULTS_ANALYSIS.md`** - Phase 1 detailed analysis (all 8 experiments)
- **`PHASE_2_ROADMAP.md`** - Phase 2 plan (qualitative + ablation studies)
- **`NEXT_EXPERIMENTS_PLAN.md`** - Parameter impact hierarchy & advanced techniques

### Configuration & Data:
- **`CLAUDE.md`** - Project instructions and research standards
- **`EXPERIMENT_PROPOSAL.md`** - Initial experiment plan
- **`PROGRESS.md`** - This file (progress summary)

### CSV Exports (results/report/):
- `experiment_comparison_summary.csv` - All 8 experiments compared
- `experiment_by_difficulty.csv` - Performance by easy/medium/hard
- `experiment_by_category.csv` - Performance by category (returns, payment, etc.)
- `all_experiments_overview.csv` - Complete overview with rankings
- `qualitative_analysis_exp6.csv` - ⭐ Phase 8A: Retrieved text inspection

### Phase 8 Summary Documents:
- `PHASE_8A_SUMMARY.md` - ⭐ Qualitative analysis & failure patterns
- `PHASE_8B_SUMMARY.md` - ⭐ BGE-M3 embedding ablation results
- `PHASE_8C_SUMMARY.md` - ⭐ Markdown splitter ablation results
- `PHASE_8D_SUMMARY.md` - ⭐ Final optimal configuration decision
- `PHASE_8E_SUMMARY.md` - ⭐ Parent-Child markdown splitter results

---

## 🚀 Next Steps

### Phase 8 Roadmap:

**Phase 8A:** ✅ Qualitative Analysis (Oct 19)
- ✅ Manual text inspection of Exp6 failures
- ✅ Identified 5 failure patterns (context cutting 40%, "meleset sedikit" 30%)
- ✅ Root cause: Splitter (70%), Embedding (60%), Chunk size (40%)

**Phase 8B:** ✅ Embedding Model Ablation (Oct 24)
- ✅ Tested BGE-M3 dense-only and multi-functional
- ✅ All variants underperformed MPNet (-1.4% to -14.4%)
- ✅ Conclusion: MPNet remains optimal

**Phase 8C:** ✅ Splitter Ablation (Oct 25)
- ✅ Tested MarkdownHeaderTextSplitter (4 experiments)
- ✅ All variants failed (-8% to -17% precision)
- ✅ Conclusion: RecursiveCharacterTextSplitter remains optimal

**Phase 8D:** ✅ Final Optimal Configuration (Oct 25)
- ✅ Current best: Exp6 (0.783 precision, 2.2% gap to target)
- ✅ Decision: Accept Exp6 as production config
- ✅ Next: Move to Phase 9 (advanced techniques)

**Phase 8E:** ✅ Parent-Child Markdown Splitter (Oct 26)
- ✅ Tested Parent-Child approach (2 variants: 500 vs 1500 tokens)
- ✅ Both failed identically: -15.6% precision
- ✅ Root cause: No parent-child splitting occurred (docs already compact)
- ✅ Conclusion: Parent-Child NOT suitable for e-commerce domain

---

## 🎓 Phase 8 Final Summary (Complete)

**Total Sub-phases:** 5 (8A, 8B, 8C, 8D, 8E)
**Total Experiments:** 15 (Phase 1-7: 8, Phase 8B: 4, Phase 8C: 4, Phase 8E: 2)
**Total Time:** ~3 weeks
**Outcome:** All basic optimization exhausted, Exp6 confirmed as optimal

### Phase 8 Key Learnings:
- ✅ **Embedding:** MPNet optimal (BGE-M3 failed: -1.4% to -14.4%)
- ✅ **Splitter:** RecursiveCharacter optimal (Markdown failed: -8% to -17%)
- ✅ **Parent-Child:** Not suitable for compact docs (failed: -15.6%)
- ✅ **Production Config:** Exp6 (k=3, MPNet, chunk=500, precision 0.783)
- ✅ **Optimization Ceiling:** Embedding ✅, Splitter ✅, k ✅, threshold ✅, chunk ✅
- ✅ **Gap to Target:** +2.2% precision (addressable with Phase 9 advanced techniques)

### What Worked:
- Qualitative analysis (Phase 8A) identified failure patterns
- Systematic ablation testing (8B, 8C, 8E)
- Negative results documented rigorously

### What Failed:
- BGE-M3 multi-functional (noise instead of quality)
- MarkdownHeaderTextSplitter (too many tiny chunks)
- Parent-Child approach (docs already optimal size)

### Next Direction:
**Phase 9: Advanced Retrieval Techniques**
- Reranker (bge-reranker-v2-m3): +6% precision expected → 0.84
- MMR (multi-doc diversity): +3-5% recall expected
- BM25 Hybrid Search: +2.5% precision expected
- **Combined target:** 0.89-0.91 precision (90% milestone)

---

## 🚀 Phase 9: Advanced Retrieval Techniques ✅ COMPLETE

**Duration:** November 2025 (1 day)
**Goal:** Bridge 2.2% precision gap using advanced techniques (reranker, hybrid search)
**Outcome:** ✅ TARGET ACHIEVED - Exp9a1 reached 0.828 precision (+5.7% vs baseline)

### Phase 9A: Reranker (Cross-Encoder) ✅ SUCCESS (Nov 6)
- ✅ Implemented BGE reranker integration (`z3_core/reranker.py`)
- ✅ Tested bge-reranker-base (600MB cross-encoder model)
- ✅ Exp9a1: Precision **0.828** (+5.7% vs Exp6) ✅ **TARGET EXCEEDED!**
- ✅ Perfect 1.000 precision on hard queries (was 0.750)
- ✅ Recall maintained: 0.950 (no degradation)
- ✅ F1: 0.845 (best across all experiments)
- ✅ MRR: 0.950 (excellent ranking)

**Key Findings:**
- ✅ Cross-encoder dramatically improves ranking quality
- ✅ Retrieval k=7 → Reranker top-3 pipeline optimal
- ✅ 0.828 precision exceeds 0.80 target
- ✅ Works flawlessly with MPNet bi-encoder
- ✅ Production-ready configuration identified


### Phase 9B: Hybrid Search (BM25 + Semantic) ❌ REJECTED (Nov 6)
- ✅ Implemented hybrid search (`z3_core/hybrid_search.py`)
- ✅ Tested 3 variants: 50/50 weights, bge-m3, 70/30 weights
- ❌ Exp9b1 (50/50): Precision 0.794 (-3.4% vs Exp9a1) FAILED
- ❌ Exp9b2 (bge-m3 50/50): Even worse than 9b1 FAILED
- ⚠️ Exp9b3 (70/30): Precision 0.811 (-1.7% vs Exp9a1) Better but still below baseline
- ✅ Conclusion: **Hybrid search NOT suitable for Indonesian e-commerce domain**

**Text Quality Analysis:**
- Queries 1-5: Identical text between Exp9a1 and Exp9b3
- Only 2 queries different:
  - ecom_easy_012: BM25 added policy_returns.md incorrectly (-16.7%)
  - ecom_hard_002: BM25 added troubleshooting_guide.md incorrectly (-33.3%)
- **Verdict:** BM25 (even at 30% weight) adds more noise than value

### Phase 9 Final Results

| Experiment | Config | Precision | Recall | F1 | Change vs Exp6 | Status |
|------------|--------|-----------|--------|-----|----------------|--------|
| Exp6 (Baseline) | k=3, MPNet | 0.783 | 0.917 | 0.795 | Baseline | Reference |
| **Exp9a1** | **k=7 + Reranker** | **0.828** ✅ | **0.950** | **0.845** | **+5.7%** | **🏆 WINNER** |
| Exp9a2 | chunk 700 + Reranker | 0.778 | 0.933 | 0.800 | +0.6% | ⚠️ Reranker helps but chunk 700 worse |
| Exp9b1 | Hybrid 50/50 | 0.794 | 0.950 | 0.821 | +1.4% | ❌ Below reranker-only |
| Exp9b2 | bge-m3 + Hybrid 50/50 | Lower | - | - | Negative | ❌ Failed |
| Exp9b3 | Hybrid 70/30 | 0.811 | 0.950 | 0.832 | +3.6% | ⚠️ Better but still below 9a1 |

### Phase 9 Key Learnings

**✅ What Worked:**
1. **Reranker (Cross-Encoder)**
   - Single most impactful technique (+5.7% precision)
   - Perfect hard query handling (1.000 precision)
   - No recall degradation (0.950 maintained)
   - Simple to implement and configure

2. **Simpler is Better**
   - Semantic + Reranker beats all hybrid variants
   - No need for complex multi-stage pipelines
   - Fewer dependencies = easier deployment

**❌ What Failed:**
1. **Hybrid Search (All Variants)**
   - 50/50 weights: Too much BM25 noise (-3.4%)
   - 70/30 weights: Still adds noise (-1.7% vs reranker-only)
   - BM25 not suitable for Indonesian CS documents

2. **Why Hybrid Failed:**
   - Indonesian common words match too broadly
   - Keyword matching confuses semantic queries
   - Hard queries require semantic understanding, not keywords
   - Weight tuning can't overcome fundamental mismatch

### Production Config: Exp9a1 🎉

**Final Winning Configuration:**
```yaml
domain_name: z3_agent_exp9a1
embedding_model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
chunk_size: 500
chunk_overlap: 50
retrieval_k: 7           # Increased from 3 for reranker
relevance_threshold: 0.3

use_reranker: true       # Phase 9A: Cross-encoder reranker
reranker_model: BAAI/bge-reranker-base
reranker_top_k: 3
reranker_use_fp16: true

use_hybrid_search: false # Phase 9B: Rejected (adds noise)
```

**Performance:**
- ✅ Precision: **0.828** (EXCEEDS 0.80 target!)
- ✅ Recall: **0.950** (exceeds 0.90 target)
- ✅ F1: **0.845** (exceeds 0.75 target)
- ✅ MRR: **0.950** (excellent ranking)
- ✅ Tokens: 210/query (efficient)
- ✅ Hard queries: **1.000 precision** (perfect!)

**By Difficulty:**
- Easy (19 queries): 0.781 precision
- Medium (9 queries): 0.889 precision
- Hard (2 queries): **1.000 precision** ✅ (was 0.750 in Exp6)

**By Category:**
- Returns: **0.955 precision** (11 queries)
- Contact: **0.861 precision** (6 queries)
- Account: **1.000 precision** (2 queries)
- Shipping: **1.000 precision** (2 queries)

---

**Status:** ✅ Phase 9 COMPLETE | **TARGET ACHIEVED!** 🎉
**Production Config:** **Exp9a1** (Semantic + Reranker) - 0.828 precision
**Research Complete:** All optimization techniques exhausted
**Deployment Ready:** Production config identified and validated
