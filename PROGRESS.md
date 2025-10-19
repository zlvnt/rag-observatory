# RAG Observatory - Progress Report

**Last Updated:** 2025-10-19
**Status:** 🔬 Phase 8 in progress | Qualitative Analysis + Advanced Optimization

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

### Phase 8B: Embedding Model Ablation (Planned)
- ⏳ Download bge-m3 model (BAAI/bge-m3, 2.2GB)
- ⏳ Re-run ALL 8 experiment configs with bge-m3
- ⏳ Compare MPNet vs bge-m3 side-by-side
- ⏳ Create `qualitative_analysis_exp6_bge.csv`

### Phase 8C: Splitter Ablation (Planned)
- ⏳ Test MarkdownHeaderTextSplitter
- ⏳ Compare with RecursiveCharacterTextSplitter
- ⏳ Re-run optimal config with new splitter

### Phase 8D: Final Optimal Configuration (Planned)
- ⏳ Combine best embedding + best splitter + optimal k
- ⏳ Document in `configs/z3_agent_production_v2.yaml`

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
- `PHASE_8A_SUMMARY.md` - ⭐ Phase 8A: Complete analysis & findings

---

## 🚀 Next Steps

### Phase 8 Roadmap:

**Phase 8A (Current):** ✅ Qualitative Analysis
- ✅ Created CSV with retrieved text previews
- ⏳ Manual inspection to identify failure patterns

**Phase 8B (Next):** Embedding Model Ablation
- Test bge-m3 across all 8 experiment configs
- Compare MPNet vs bge-m3 performance
- Expected improvement: +5-10% precision

**Phase 8C:** Splitter Ablation
- Test MarkdownHeaderTextSplitter for structured docs
- Compare with current RecursiveCharacterTextSplitter
- Expected improvement: +3-5% precision

**Phase 8D:** Final Optimal Configuration
- Combine best findings from 8A-8C
- Document production-ready config v2

---

**Status:** Phase 8A complete ✅ | Ready for Phase 8B (bge-m3 ablation)
**Current Best:** Exp6 (k=3, precision 0.783, recall 0.917, F1 0.795)
**Target Gap:** +2.2% precision to reach 0.80 target
**Root Causes Identified:** Splitter (70%), Embedding (60%), Chunk size (40%)
