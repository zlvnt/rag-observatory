# Phase 8E Summary: Parent-Child Markdown Splitter

**Status:** FAILED - Approach not suitable for e-commerce domain
**Date:** 2025-10-26
**Precision Impact:** -15.6% (0.783 → 0.661)

---

## Objective

Test Parent-Child Markdown splitting approach recommended by Claude web exploration based on Perplexity architecture insights:
- **Parent:** Split on Markdown headers (##)
- **Child:** Apply RecursiveCharacterTextSplitter if parent > token threshold
- **Store:** Children for retrieval, parent metadata for context
- **Expected:** +2-4% precision improvement to reach 0.80+ target

---

## Experiment Setup

### Two Variants Tested

**Exp8E_v1 (Claude Recommendation):**
- `parent_max_tokens`: 500 (conservative threshold)
- `child_chunk_size`: 500
- `child_chunk_overlap`: 50
- Config: `configs/experiments_phase8e/z3_agent_exp8e_v1.yaml`

**Exp8E_v2 (Larger Context):**
- `parent_max_tokens`: 1500 (preserve more context)
- `child_chunk_size`: 450
- `child_chunk_overlap`: 50
- Config: `configs/experiments_phase8e/z3_agent_exp8e_v2.yaml`

### Common Parameters
- Embedding: `paraphrase-multilingual-mpnet-base-v2`
- Retrieval k: 3
- Relevance threshold: 0.3
- Headers to split: `["##"]` (level 2 headers only)

### Implementation
- Created: `z3_core/vector_parent_child.py` (new splitter)
- Created: `runners/test_runner_parent_child.py` (dedicated runner)
- Updated: `z3_core/domain_config.py` (added parent-child parameters)

---

## Results

### Both Variants Failed Identically

| Metric | Exp8E_v1 | Exp8E_v2 | Exp6 (Baseline) | Change |
|--------|----------|----------|-----------------|--------|
| **Precision@3** | 0.661 | 0.661 | 0.783 | -15.6% |
| **Recall@3** | 0.933 | 0.933 | 0.917 | +1.7% |
| **F1 Score** | 0.720 | 0.720 | 0.795 | -9.4% |
| **MRR** | 0.928 | 0.928 | 0.983 | -5.6% |
| Success Rate | 100.0% | 100.0% | 100.0% | 0.0% |
| Avg Chunks/Query | 1.9 | 1.9 | 2.0 | -5.0% |
| Avg Tokens/Query | 249 | 249 | 261 | -4.6% |

### Performance by Difficulty

| Difficulty | Exp8E Precision | Exp6 Precision | Change |
|------------|-----------------|----------------|--------|
| **Easy (19)** | 0.588 | 0.762 | **-22.9%** |
| **Medium (9)** | 0.796 | 0.815 | -2.3% |
| **Hard (2)** | 0.750 | 0.833 | -10.0% |

**Critical finding:** Easy queries suffered massive precision drop (-22.9%)

---

## Root Cause Analysis

### Verification Test Results

```
Raw docs: 4
Children chunks: 25
Complete sections (not split): 25  ← ALL kept intact
Split sections: 0                  ← NO parent-child splitting occurred!

Chunk size distribution:
- Average: 572 chars (~143 tokens)
- Minimum: 39 chars (~9 tokens)
- Maximum: ~2000 chars (~500 tokens)
```

### What Went Wrong

1. **All parent sections < threshold**
   - All 25 Markdown sections were already < 500 tokens
   - No parent exceeded `parent_max_tokens` threshold
   - No child splitting occurred

2. **Result = Pure Markdown split (Phase 8C)**
   - Equivalent to failed Phase 8C approach
   - But worse: Some chunks only 39 characters (too small)
   - Lost context from cutting at header boundaries

3. **Why both variants identical?**
   - v1: 500 token threshold → no sections exceeded
   - v2: 1500 token threshold → definitely no sections exceeded
   - Both resulted in 25 complete sections, no splitting

4. **Implementation was correct**
   - Algorithm worked as designed
   - Setup matched Claude recommendation
   - Just doesn't fit our use case

---

## Why Parent-Child Failed for E-commerce

### Domain Characteristics
- Docs already compact and well-structured
- Average section: ~143 tokens (well below thresholds)
- Semantic units (FAQ items, policy sections) already small
- No "large parent sections" to split

### Parent-Child Best For:
- Long technical documentation
- Research papers with large sections
- Books with chapter/section structure
- Content with 1000+ token sections

### Our Domain (E-commerce FAQ):
- Policy sections: 100-300 tokens each
- FAQ items: 50-150 tokens each
- Already optimal chunk sizes
- No benefit from parent-child hierarchy

---

## Conclusions

### Verdict
**Parent-Child Markdown splitter REJECTED for e-commerce domain**

### Key Learnings
1. **Domain fit matters** - Advanced techniques don't always help
2. **Compact docs don't need parent-child** - Our docs already optimal size
3. **Exp6 remains optimal** - RecursiveCharacter with chunk_size=500 still best
4. **Phase 8C insight validated** - Pure Markdown splitting hurts precision
5. **No magic bullet** - Need different approach to bridge 2.2% gap

### Impact on Research
- **Phase 8 ablation study complete:** Tested chunking, embeddings, headers, parent-child
- **Negative results valuable:** Confirms Exp6 strategy is solid
- **Next direction clear:** Move to retrieval enhancements (reranker, MMR)

---

## Next Steps: Phase 9

Based on Claude web recommendations, prioritize:

### Option A: Reranker (Highest Impact)
- **Expected:** +6% precision (0.783 → 0.84)
- **Approach:** bge-reranker-v2-m3 on retrieved results
- **Effort:** 2-3 days
- **Confidence:** High (proven technique)

### Option B: MMR (Multi-doc Diversity)
- **Expected:** +3-5% recall on multi-doc queries
- **Approach:** Maximal Marginal Relevance for diversity
- **Effort:** 1-2 days
- **Confidence:** Medium

### Option C: Hybrid BM25 + Semantic
- **Expected:** +2.5% precision
- **Approach:** Combine keyword (BM25) + semantic (MPNet)
- **Effort:** 3-4 days
- **Confidence:** Medium-High

### Recommended: Start with Reranker (highest ROI)

---

## Files & Artifacts

### Configurations
- `configs/experiments_phase8e/z3_agent_exp8e_v1.yaml`
- `configs/experiments_phase8e/z3_agent_exp8e_v2.yaml`

### Implementation
- `z3_core/vector_parent_child.py`
- `runners/test_runner_parent_child.py`

### Results
- `results/phase8e/exp8e_v1/report_20251026_095852.txt`
- `results/phase8e/exp8e_v2/report_20251026_095933.txt`
- `results/phase8e/exp8e_v1/detailed_results.csv`
- `results/phase8e/exp8e_v2/detailed_results.csv`

### Commands to Reproduce
```bash
# Run v1
python runners/test_runner_parent_child.py --domain z3_agent_exp8e_v1 --output results/phase8e/exp8e_v1/

# Run v2
python runners/test_runner_parent_child.py --domain z3_agent_exp8e_v2 --output results/phase8e/exp8e_v2/
```

---

## Final Status

**Phase 8E:** COMPLETE (FAILED)
**Exp6 remains winner:** Precision 0.783, Recall 0.917, F1 0.795
**Gap to target:** Still need +2.2% precision to reach 0.80
**Phase 8 complete:** Ready to close and move to Phase 9
