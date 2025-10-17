# RAG Observatory - Progress Report

**Last Updated:** 2025-10-16
**Status:** Phase 1 complete (5 experiments), Phase 2 ready with refined hypothesis

---

## 🎯 Current Objective

**Phase 2:** Test optimal hypothesis (MPNet + k=4) to achieve target metrics.

**Hypothesis:** MPNet embedding + k=4 + chunk=500 will achieve:
- Precision ≥ 0.75-0.80 (combining MPNet quality with k=4 selectivity)
- Recall ≥ 0.90-0.95 (maintain high coverage)
- F1 ≥ 0.80 (balanced improvement)

**Next Experiment:** Exp5 (CRITICAL - validates hypothesis)

**Success Criteria:** Precision ≥ 0.80, Recall ≥ 0.90, F1 ≥ 0.75

---

## ✅ Completed Work

### Phase 1: Foundation (Oct 12-13)
- ✅ Refactored `z3_core/` modules to be configurable
- ✅ Created `domain_config.py` for YAML-based configuration
- ✅ Removed hardcoded paths and production dependencies
- ✅ Added debug info capture (`return_debug_info=True` pattern)

### Phase 2: Golden Dataset (Oct 13)
- ✅ Created 30-query test dataset for e-commerce domain
- ✅ Distribution: 19 easy, 9 medium, 2 hard
- ✅ Categories: returns, contact, payment, shipping, product, account, technical
- ✅ Defined `expected_docs` for each query (ground truth)

### Phase 3: Test Runner (Oct 14)
- ✅ Built `runners/test_runner.py` - retrieval-focused evaluation
- ✅ **Removed routing evaluation** (out of scope)
- ✅ **Removed LLM generation** (out of scope - focus on retrieval only)
- ✅ Simplified to pure retrieval metrics

### Phase 4: Metrics & Output (Oct 14)
- ✅ Implemented Precision@K, Recall@K, F1, MRR
- ✅ Added true/false positives tracking
- ✅ Created simplified CSV output (retrieval-focused)
- ✅ Generated human-readable reports with RAG config tracking
- ✅ Fixed metrics calculation (count unique docs, not chunks)

### Phase 5: Baseline Test (Oct 14)
- ✅ Ran 30 queries with baseline config
- ✅ Results analyzed and documented

### Phase 6: Ablation Study - Phase 1 (Oct 16)
- ✅ Ran 4 experiments (Exp1-4) with different configurations
- ✅ Identified k=6 as bottleneck (precision drop -23.6%)
- ✅ Validated MPNet superiority (MRR +9%, 0.950)
- ✅ Found threshold has no impact on precision/recall
- ✅ Developed optimal hypothesis: MPNet + k=4 + chunk=500

---

## 📊 Phase 1 Results Summary

### Comparative Metrics:

| Experiment | Config Changes | Precision | Recall | F1 | MRR | Tokens/Q | Verdict |
|------------|----------------|-----------|--------|----|----|----------|---------|
| **Baseline** | k=4, t=0.8, chunk=700, MiniLM | **0.706** | 0.950 | **0.752** | 0.872 | 583 | ⚠️ Reference |
| **Exp1** | t=0.3 | **0.706** | 0.950 | **0.752** | 0.872 | **382** ✅ | ⚠️ No metric change |
| **Exp2** | k=6 | **0.539** ❌ | 0.967 | **0.652** | 0.872 | 513 | ❌ Precision crashed |
| **Exp3** | k=6, chunk=500 | **0.589** | 0.950 | **0.680** | 0.861 | **369** ✅ | ⚠️ Below baseline |
| **Exp4** | k=6, chunk=500, MPNet | **0.639** | 0.950 | **0.725** | **0.950** ⭐ | **352** ✅ | ⭐ Best MRR |

**Target Metrics:** Precision ≥0.80, Recall ≥0.90, F1 ≥0.75

### Key Insights from Phase 1:

✅ **What Worked:**
1. **MPNet embedding** - MRR improved to 0.950 (+9%), best ranking quality
2. **Token efficiency** - Exp1 reduced tokens 34% without hurting metrics
3. **Smaller chunks** - Exp3/4 achieved ~350 tokens/query (40% reduction)

❌ **What Failed:**
1. **k=6** - Precision dropped 23.6% (0.706 → 0.539), too much noise
2. **Lowering threshold** - No impact on precision/recall (threshold irrelevant)
3. **Stacking changes** - Exp2-4 couldn't recover from k=6 damage

💡 **Critical Discovery:**
**k=6 is the bottleneck** - All experiments with k=6 performed worse than baseline. MPNet showed promise (MRR 0.950) but was tested with suboptimal k=6. **Hypothesis:** MPNet + k=4 could achieve precision 0.75-0.80.

---

## 🔍 Key Findings

### ✅ What's Working

1. **Excellent Recall (95%)**
   - System rarely misses relevant documents
   - Shows good coverage of knowledge base

2. **Fast Performance (19ms avg)**
   - Production-ready latency
   - FAISS vector search very efficient

3. **100% Success Rate**
   - Every query retrieves at least 1 relevant doc
   - No complete failures

4. **Good Ranking (MRR 0.872)**
   - Relevant docs usually ranked high
   - First result often correct

### ❌ Problems Identified

1. **Precision Too Low (70.6%)**
   - 30% of retrieved docs are irrelevant (false positives)
   - `policy_returns.md` appears in almost all queries (even out-of-scope)
   - **Root cause:** Threshold 0.8 is too high, all docs have score < 0.8

2. **Threshold Not Working**
   - All relevance scores < 0.8
   - Threshold effectively disabled
   - System retrieves k=4 docs regardless of quality

3. **Multi-Doc Queries Struggle**
   - Medium queries (2 docs needed): Recall 88.9% (missing 11%)
   - Hard queries (2 docs needed): Recall 75% (missing 25%)
   - System tends to retrieve chunks from single doc only

4. **Category-Specific Issues**
   - Payment queries: Only 50% precision (many false positives)
   - Product queries: Only 55.6% precision
   - Suggests embedding model struggles with certain query types

5. **Out-of-Scope Detection Missing**
   - Query "Resep nasi goreng" still retrieves docs
   - No mechanism to detect irrelevant queries

### 📈 Specific Problem Cases

**Example 1: Low Precision (ecom_easy_008)**
```
Query: "Sudah bayar tapi status masih menunggu pembayaran"
Expected: troubleshooting_guide.md
Retrieved:
  - policy_returns.md (rank #1, score 0.25) ❌ False positive
  - policy_returns.md (rank #2, score 0.0)  ❌ False positive
  - product_faq.md (rank #3, score 0.0)     ❌ False positive
  - troubleshooting_guide.md (rank #4, score 0.5) ✓ True positive

Precision: 0.333 (only 1/3 unique docs relevant)
Problem: Correct doc ranked last!
```

**Example 2: Missing Multi-Doc (ecom_medium_001)**
```
Query: "Return barang tapi penjual tidak respon 3 hari"
Expected: policy_returns.md + contact_escalation.md
Retrieved: policy_returns.md only (4 chunks from same doc)

Precision: 1.0 (doc retrieved is correct)
Recall: 0.5 (missing contact_escalation.md)
Problem: No diversity in retrieval
```

---

## 🔧 Phase 2 Planned Experiments

### Priority Experiments (Next Steps):

| Exp | chunk_size | overlap | embedding | k | threshold | Expected Result | Rationale |
|-----|------------|---------|-----------|---|-----------|-----------------|-----------|
| **Exp5** ⭐ | 500 | 50 | **MPNet-v2** | **4** | 0.3 | Prec 0.75-0.80 | **Optimal hypothesis** |
| **Exp6** | 500 | 50 | MPNet-v2 | **3** | 0.3 | Prec 0.80+, Rec 0.90 | Test lower k |
| **Exp7** | 500 | 50 | MPNet-v2 | **5** | 0.3 | Balance test | K sweet spot |
| **Exp8** | 700 | 100 | MPNet-v2 | 4 | 0.3 | Embedding ablation | Isolate MPNet impact |

**Why Exp5 is Critical:**
- Combines MPNet's ranking quality (MRR 0.950 from Exp4)
- With k=4's precision (baseline's best parameter)
- Plus chunk=500's token efficiency (352 tokens/query)
- **This was never tested in Phase 1** - Exp4 used k=6 which hurt precision

**Estimated Time:**
- Exp5: ~5 min (rebuild vector store - new config)
- Exp6-7: <1 min each (k change only, no rebuild)
- Exp8: ~5 min (rebuild - different chunk size)
- **Total:** ~15 minutes for priority experiments

---

## 📁 Results Location

```
results/
├── results_baseline/  (Oct 16, 10:52) - Original baseline
├── exp1/              (Oct 16, 10:55) - Threshold 0.3
├── exp2/              (Oct 16, 11:04) - k=6
├── exp3/              (Oct 16, 11:05) - k=6, chunk=500
├── exp4/              (Oct 16, 16:50) - k=6, chunk=500, MPNet
└── exp5/              (Planned) - k=4, chunk=500, MPNet ⭐
```

**Each experiment folder contains:**
- `detailed/` - 30 JSON files with per-query details
- `summary_TIMESTAMP.csv` - Quick metrics comparison
- `report_TIMESTAMP.txt` - Human-readable analysis

---

## 🎯 Next Steps

### Immediate (Now)
1. ✅ Create Exp5 config (z3_agent_exp5.yaml) - **DONE**
2. ✅ Update EXPERIMENT_PROPOSAL.md with Phase 1 results - **DONE**
3. ✅ Update PROGRESS.md - **DONE**
4. **RUN Exp5** - Critical test of optimal hypothesis
5. Analyze Exp5 results vs baseline and Exp4

### If Exp5 Succeeds (Precision ≥0.75):
6. Run Exp6 (k=3) to test if lower k improves further
7. Run Exp7 (k=5) to find sweet spot
8. Create comparison report across all experiments
9. Document winning configuration

### If Exp5 Fails (Precision <0.75):
6. Run Exp8 (embedding ablation) to isolate MPNet impact
7. Consider alternative approaches:
   - Reranking layer (cross-encoder)
   - Hybrid search (BM25 + semantic)
   - Query expansion or reformulation
   - MMR (Maximal Marginal Relevance) for diversity

### Future Enhancements (After Optimal Config Found):
- Build comparison dashboard/visualization
- Test on additional domains beyond e-commerce
- Implement production deployment script
- Add monitoring and A/B testing framework

---

## 📝 Notes & Observations

### Design Decisions

**Why remove routing & generation from evaluation?**
- **Focus:** We want to optimize retrieval first
- **Speed:** No LLM calls = 100x faster iteration
- **Cost:** Free to run unlimited experiments
- **Clarity:** Isolate retrieval quality from prompt engineering

**Why count unique docs instead of chunks?**
- **User perspective:** Users care if we find the right document, not how many chunks
- **Metrics accuracy:** Precision/Recall should reflect document-level correctness
- **Avoid inflation:** 4 chunks from 1 doc ≠ 4 relevant results

**Why track true/false positives?**
- **Actionable insights:** Know exactly which docs are noise
- **Pattern detection:** See if certain docs always appear (like policy_returns.md)
- **Debugging:** Understand why precision is low

### Technical Insights

**Relevance score distribution:**
- Max observed: 0.6 (ecom_easy_005)
- Typical range: 0.0 - 0.5
- Threshold 0.8 is unreachable with current embedding model

**Chunk retrieval patterns:**
- System often retrieves 4 chunks from same document
- Lacks diversity in multi-doc scenarios
- May need MMR (Maximal Marginal Relevance) for diversity

**Latency breakdown:**
- Vector search: ~10-15ms (fast)
- First query (cache miss): ~240ms
- Subsequent queries: ~10ms
- Embedding inference is bottleneck on first query

---

## 🤝 Collaboration Notes

**For Future Sessions:**
- All test configs in `configs/` directory
- Results organized by experiment name
- Each result includes `rag_config` for reproducibility
- CSV provides quick comparison, JSON has full details

**Key Files:**
- `CLAUDE.md` - Project instructions (static)
- `PROGRESS.md` - Current state (this file, update regularly)
- `configs/z3_agent_config.yaml` - Baseline configuration
- `results/baseline/report_*.txt` - Detailed findings

---

**Status:** Phase 1 complete ✅ | Phase 2 ready - Exp5 config created 🚀

---

## 📊 Quick Reference - Phase 1 Learnings

**What We Learned:**
1. **Threshold doesn't matter** - All scores < 0.8, so threshold 0.8 vs 0.3 = same result
2. **k=6 is too high** - Precision drops 24% due to noise
3. **MPNet > MiniLM** - Better ranking (MRR +9%) and semantic understanding
4. **chunk=500 > chunk=700** - Better token efficiency (40% reduction) with same quality

**What to Test Next:**
- **Exp5 (CRITICAL):** MPNet + k=4 + chunk=500 - Combines best parameters
- **Exp6-7:** Optimize k value (test k=3 and k=5)
- **Exp8:** Isolate MPNet impact (keep chunk=700 to isolate embedding change)

**Predicted Optimal Config:**
```yaml
embedding_model: paraphrase-multilingual-mpnet-base-v2
chunk_size: 500
chunk_overlap: 50
retrieval_k: 3 or 4  # To be determined by Exp5-7
relevance_threshold: 0.3  # Doesn't matter, but keep for consistency
```
