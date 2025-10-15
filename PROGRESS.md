# RAG Observatory - Progress Report

**Last Updated:** 2025-10-14
**Status:** Baseline complete, ready for ablation experiments

---

## 🎯 Current Objective

**Optimize RAG retrieval configuration** through systematic experimentation.

**Goal:** Find the best combination of:
- Chunk size & overlap
- Retrieval k parameter
- Relevance threshold
- Embedding model

**Success Criteria:** Precision ≥ 0.80, Recall ≥ 0.70, F1 ≥ 0.75

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

---

## 📊 Baseline Results

**Configuration:**
```yaml
chunk_size: 700
chunk_overlap: 100
embedding_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
vector_store: FAISS
retrieval_k: 4
relevance_threshold: 0.8
```

**Metrics:**
| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Precision@3 | **0.706** | 0.80 | ⚠️ Below target |
| Recall@3 | **0.950** | 0.70 | ✅ Excellent |
| F1 Score | **0.752** | 0.75 | ✅ At target |
| MRR | **0.872** | 0.80 | ✅ Good |
| Success Rate | **100%** | 90% | ✅ Perfect |

**Performance:**
- Avg Latency: 19ms (excellent)
- P95 Latency: 15ms
- Avg Tokens/Query: 583

**By Category Performance:**
- 🏆 Best: Technical (1.000), Account (0.834)
- ⚠️ Worst: Payment (0.500), Product (0.556)

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

## 🔧 Planned Experiments

### Experiment Matrix

| Exp | chunk_size | overlap | embedding | k | threshold | Expected Impact |
|-----|------------|---------|-----------|---|-----------|-----------------|
| **Baseline** | 700 | 100 | MiniLM-L12 | 4 | 0.8 | (current) |
| **Exp1** | 700 | 100 | MiniLM-L12 | 4 | **0.3** | Precision +10-15% |
| **Exp2** | 700 | 100 | MiniLM-L12 | **6** | 0.3 | Recall +5% (multi-doc) |
| **Exp3** | **500** | **50** | MiniLM-L12 | 6 | 0.3 | Precision +5% |
| **Exp4** | 500 | 50 | **mpnet-v2** | 6 | 0.3 | Precision +15-20% |

**Rationale:**
1. **Exp1:** Fix threshold to actually filter low-quality docs
2. **Exp2:** Increase k to improve multi-doc retrieval
3. **Exp3:** Smaller chunks for more granular retrieval
4. **Exp4:** Better embedding model for improved semantic understanding

**Estimated Time:**
- Exp1: 1 min (config change only)
- Exp2: 1 min (config change only)
- Exp3: 5 min (rebuild vector store)
- Exp4: 15 min (download model + rebuild)
- **Total:** ~25 minutes

---

## 📁 Results Location

```
results/
└── baseline/  (Oct 14, 16:16)
    ├── detailed/
    │   ├── ecom_easy_001.json
    │   ├── ... (30 files)
    ├── summary_20251014_161627.csv
    └── report_20251014_161627.txt
```

**Baseline can be renamed to:**
```bash
mv results results_baseline
# Or keep as-is and create new folders for experiments
```

---

## 🎯 Next Steps

### Immediate (Today)
1. Create experiment configs (exp1-exp4.yaml)
2. Run Exp1 (threshold 0.3)
3. Compare results with baseline
4. Decide: continue with Exp2-4 or iterate on threshold?

### Short-term (This Week)
5. Complete experiment matrix
6. Create comparison script/tool
7. Identify winning configuration
8. Document findings

### Future (Optional)
- Hybrid search (BM25 + semantic)
- Reranking layer (cross-encoder)
- Query classification for routing
- Expand to additional domains

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

**Status:** Ready to execute experiment matrix 🚀
