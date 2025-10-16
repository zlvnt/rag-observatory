# RAG Experiment Matrix - Research-Grade Ablation Study

## Phase 1: Initial Experiments (Run First) ⏱️ ~25 min

| Exp | threshold | k | chunk | overlap | embedding | Purpose | Rebuild? |
|-----|-----------|---|-------|---------|-----------|---------|----------|
| **Baseline** | 0.8 | 4 | 700 | 100 | MiniLM-L12 | Current state | ✅ Done |
| **Exp1** | 0.3 | 4 | 700 | 100 | MiniLM-L12 | Fix precision | ❌ |
| **Exp2** | 0.3 | 6 | 700 | 100 | MiniLM-L12 | Improve recall | ❌ |
| **Exp3** | 0.3 | 6 | 500 | 50 | MiniLM-L12 | Granular chunks | ✅ |
| **Exp4** | 0.3 | 6 | 500 | 50 | MPNet-v2 | Best combined | ✅ |

**Run commands:**
```bash
python runners/test_runner.py --domain z3_agent_exp1 --output results_exp1/
python runners/test_runner.py --domain z3_agent_exp2 --output results_exp2/
python runners/test_runner.py --domain z3_agent_exp3 --output results_exp3/
python runners/test_runner.py --domain z3_agent_exp4 --output results_exp4/
```

---

## Phase 2: Extended Ablation (After Phase 1 Analysis) ⏱️ ~30 min

### Wave 2A: Ablation & Safety Nets (3 experiments)

| Exp | threshold | k | chunk | overlap | embedding | Purpose | Rebuild? |
|-----|-----------|---|-------|---------|-----------|---------|----------|
| **Exp5** | 0.5 | 4 | 700 | 100 | MiniLM-L12 | Mid-range threshold | ❌ |
| **Exp6** | 0.3 | 8 | 700 | 100 | MiniLM-L12 | Higher K test | ❌ |
| **Exp7** | 0.3 | 6 | 700 | 100 | MPNet-v2 | Embeddings isolation | ✅ |

**Rationale:**
- **Exp5**: Bridge gap between 0.8 (too strict) and 0.3 (too loose)
- **Exp6**: Test if k=8 improves recall for multi-doc queries
- **Exp7**: Isolate embedding impact (vs Exp4 which also changes chunk_size)

### Wave 2B: Extreme Ranges (3 experiments)

| Exp | threshold | k | chunk | overlap | embedding | Purpose | Rebuild? |
|-----|-----------|---|-------|---------|-----------|---------|----------|
| **Exp8** | 0.3 | 6 | 1000 | 150 | MiniLM-L12 | Larger chunks | ✅ |
| **Exp9** | 0.3 | 6 | 300 | 30 | MiniLM-L12 | Very small chunks | ✅ |
| **Exp10** | 0.3 | 10 | 500 | 50 | MPNet-v2 | Max retrieval test | ❌ |

**Rationale:**
- **Exp8**: Test if more context helps (trade-off: noise?)
- **Exp9**: Maximum granularity for precision
- **Exp10**: Push best config (Exp4) to retrieval limits

---

## Complete Matrix Overview

**Total: 11 experiments** (1 baseline + 10 variations)

### Parameter Coverage:
- **Threshold**: 0.3, 0.5, 0.8
- **K**: 4, 6, 8, 10
- **Chunk Size**: 300, 500, 700, 1000
- **Embeddings**: MiniLM-L12, MPNet-v2

### Execution Time:
- **Phase 1**: ~25 min (baseline + 4 exp)
- **Phase 2**: ~30 min (6 exp)
- **Total**: ~55 min

---

## Run Commands (Phase 2)

### Fast Experiments (No Rebuild)
```bash
python runners/test_runner.py --domain z3_agent_exp5 --output results_exp5/
python runners/test_runner.py --domain z3_agent_exp6 --output results_exp6/
python runners/test_runner.py --domain z3_agent_exp10 --output results_exp10/
```

### Slow Experiments (Rebuild Required)
```bash
python runners/test_runner.py --domain z3_agent_exp7 --output results_exp7/
python runners/test_runner.py --domain z3_agent_exp8 --output results_exp8/
python runners/test_runner.py --domain z3_agent_exp9 --output results_exp9/
```

---

## Decision Point

**After Phase 1 results:**
- If Precision already >0.85 → Skip Exp5
- If Recall already >0.95 → Skip Exp6, Exp10
- If Exp4 shows huge gain → Prioritize Exp7 (embedding isolation)
- If chunking unclear → Run Exp8, Exp9

**Goal:** Don't blindly run all - use Phase 1 insights to select most valuable Phase 2 experiments.

---

## Analysis Plan

Compare metrics across experiments:
1. **Single-variable impact**: Exp1 vs Baseline (threshold only)
2. **Cumulative gains**: Baseline → Exp1 → Exp2 → Exp3 → Exp4
3. **Ablation**: Exp4 vs Exp7 (chunk impact), Exp2 vs Exp6 (k impact)
4. **Extremes**: Exp3 vs Exp8 vs Exp9 (chunk size sweet spot)

**Output:** Identify optimal RAG configuration for e-commerce domain.
