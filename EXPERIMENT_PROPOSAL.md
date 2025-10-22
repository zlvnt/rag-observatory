# RAG Experiment Matrix - Research-Grade Ablation Study

## Phase 1: Initial Experiments ✅ COMPLETED (Oct 16, 2025)

| Exp | threshold | k | chunk | overlap | embedding | Precision | Recall | F1 | MRR | Status |
|-----|-----------|---|-------|---------|-----------|-----------|--------|----|----|--------|
| **Baseline** | 0.8 | 4 | 700 | 100 | MiniLM-L12 | **0.706** | 0.950 | **0.752** | 0.872 | ⚠️ Reference |
| **Exp1** | 0.3 | 4 | 700 | 100 | MiniLM-L12 | **0.706** | 0.950 | **0.752** | 0.872 | ⚠️ No change |
| **Exp2** | 0.3 | 6 | 700 | 100 | MiniLM-L12 | **0.539** | 0.967 | **0.652** | 0.872 | ❌ Precision drop |
| **Exp3** | 0.3 | 6 | 500 | 50 | MiniLM-L12 | **0.589** | 0.950 | **0.680** | 0.861 | ⚠️ Still low |
| **Exp4** | 0.3 | 6 | 500 | 50 | MPNet-v2 | **0.639** | 0.950 | **0.725** | **0.950** | ⭐ Best MRR |

### 🔍 Phase 1 Key Findings:

**❌ Failed Approaches:**
1. **Lowering threshold (Exp1)**: No impact on precision/recall - threshold doesn't matter when all scores < 0.8
2. **Increasing k to 6 (Exp2-4)**: **MAJOR precision drop** (-23.6%) - introduces too much noise
3. **Smaller chunks (Exp3)**: Marginal improvement, but offset by k=6 damage

**✅ Successful Approaches:**
1. **MPNet embedding (Exp4)**: **MRR improved to 0.950** (+9%) - better ranking quality
2. **Token efficiency**: Exp1 achieved 382 tokens/query (-34% vs baseline) without hurting metrics

**💡 Critical Insight:**
- **k=6 is the bottleneck** - All experiments with k=6 had worse precision than baseline
- **MPNet is better than MiniLM** - But never tested with optimal k=4
- **Hypothesis:** MPNet + k=4 + chunk=500 could achieve **precision 0.75-0.80**

---

## Phase 2: Refined Experiments (Based on Phase 1 Insights) ⏱️ ~20 min

### 🎯 Priority Experiments (Run These First)

| Exp | threshold | k | chunk | overlap | embedding | Purpose | Expected Result | Rebuild? |
|-----|-----------|---|-------|---------|-----------|---------|-----------------|----------|
| **Exp5** ⭐ | 0.3 | **4** | 500 | 50 | **MPNet-v2** | **Optimal hypothesis** | Prec 0.75-0.80 | ✅ |
| **Exp6** | 0.3 | **3** | 500 | 50 | **MPNet-v2** | Test lower k | Prec 0.80+, Rec ~0.90 | ❌ |
| **Exp7** | 0.3 | **5** | 500 | 50 | **MPNet-v2** | K sweet spot search | Balance test | ❌ |

**Rationale:**
- **Exp5** ⭐: **MOST IMPORTANT** - Test MPNet with k=4 (baseline k). Phase 1 showed k=6 hurts precision, but MPNet improves ranking. This combines best of both.
- **Exp6**: If Exp5 succeeds, test if k=3 can push precision even higher (trade-off: might hurt multi-doc recall)
- **Exp7**: Find optimal k value between 3-5

### 🔬 Optional Ablation Tests (Lower Priority)

| Exp | threshold | k | chunk | overlap | embedding | Purpose | Rebuild? |
|-----|-----------|---|-------|---------|-----------|---------|----------|
| **Exp8** | 0.3 | 4 | 700 | 100 | **MPNet-v2** | Isolate embedding only | ✅ |
| **Exp9** | 0.3 | 4 | 1000 | 150 | **MPNet-v2** | Larger chunks test | ✅ |
| **Exp10** | 0.3 | 4 | 300 | 30 | **MPNet-v2** | Very small chunks | ✅ |

**Rationale:**
- **Exp8**: Pure embedding ablation - isolate MPNet impact without changing chunk_size
- **Exp9-10**: Test chunk size extremes with optimal k=4 + MPNet

---

## 📊 Experiment Summary & Comparison

### Phase 1 Results (Completed):

```
Metric Comparison:
                  Precision  Recall   F1     MRR    Tokens/Q  Latency
Baseline (k=4)    0.706     0.950   0.752   0.872    583      21ms
Exp1 (t=0.3)      0.706     0.950   0.752   0.872    382 ✅   19ms
Exp2 (k=6)        0.539 ❌  0.967   0.652   0.872    513      20ms
Exp3 (chunk=500)  0.589     0.950   0.680   0.861    369 ✅   11ms ⚡
Exp4 (MPNet)      0.639     0.950   0.725   0.950 ⭐ 352 ✅   29ms
```

**Winner by Category:**
- **Best Precision:** Baseline (0.706)
- **Best Recall:** Exp2 (0.967, but precision terrible)
- **Best F1:** Baseline (0.752)
- **Best MRR:** Exp4 (0.950) ⭐
- **Best Efficiency:** Exp4 (352 tokens, 11ms latency)
- **Most Balanced:** Baseline or Exp1

### Phase 2 Execution Plan:

**Run Order (Priority):**
1. **Exp5** ⭐ (Critical test - optimal hypothesis)
2. **Exp6** (If Exp5 succeeds, test k=3)
3. **Exp7** (K optimization)
4. **Exp8-10** (Optional ablation, if time permits)

**Total Experiments:** 6 new experiments + 5 from Phase 1 + Baseline = **12 total**

### Parameter Coverage:
- **Threshold**: 0.3, 0.8 (0.3 is standard now)
- **K**: 3, 4, 5, 6 (focused on 3-5 range)
- **Chunk Size**: 300, 500, 700, 1000
- **Embeddings**: MiniLM-L12, MPNet-v2 (MPNet is superior)

### Execution Time:
- **Phase 1**: ✅ ~25 min (completed)
- **Phase 2**: ~20 min (6 experiments, 3 fast + 3 rebuild)
- **Total**: ~45 min

---

## Run Commands (Phase 2)

### 🎯 Priority Experiments (Run First):
```bash
# Exp5 ⭐ CRITICAL - Test optimal hypothesis (MPNet + k=4 + chunk=500)
python runners/test_runner.py --domain z3_agent_exp5 --output results/exp5/

# Exp6 - Test if k=3 improves precision further
python runners/test_runner.py --domain z3_agent_exp6 --output results/exp6/

# Exp7 - Find sweet spot between k=3 and k=5
python runners/test_runner.py --domain z3_agent_exp7 --output results/exp7/
```

### 🔬 Optional Ablation (If Time Permits):
```bash
# Exp8 - Pure embedding ablation (MPNet with baseline chunk settings)
python runners/test_runner.py --domain z3_agent_exp8 --output results/exp8/

# Exp9 - Larger chunks test
python runners/test_runner.py --domain z3_agent_exp9 --output results/exp9/

# Exp10 - Very small chunks test
python runners/test_runner.py --domain z3_agent_exp10 --output results/exp10/
```

---

## 🎯 Decision Point (After Phase 1)

**Phase 1 Verdict:**
- ❌ Precision NOT >0.85 → **RUN Exp5** (critical)
- ✅ Recall already >0.95 → Skip experiments focused on recall
- ✅ Exp4 shows MRR gain → **RUN Exp5** to isolate MPNet + k=4
- ⚠️ k=6 consistently hurts precision → **Focus on k=3-5 range**

**Phase 2 Strategy:**
1. **Must run:** Exp5 (optimal hypothesis test)
2. **Should run:** Exp6-7 (k optimization)
3. **Nice to have:** Exp8-10 (ablation studies)

**Goal:** Find config that achieves **Precision ≥0.80, Recall ≥0.90, F1 ≥0.75**

---

## 📈 Analysis Plan

### Phase 1 Analysis (Completed):
1. ✅ **Threshold impact**: Exp1 vs Baseline → **No effect** (threshold irrelevant)
2. ✅ **K impact**: Exp2 vs Baseline → **Negative** (k=6 hurts precision -23.6%)
3. ✅ **Chunk impact**: Exp3 vs Exp2 → **Marginal positive** (+9.3% precision, but still below baseline)
4. ✅ **Embedding impact**: Exp4 vs Exp3 → **Positive** (+8.5% precision, +10% MRR)

### Phase 2 Analysis (Planned):
1. **Optimal config test**: Exp5 vs Baseline → Validate MPNet + k=4 hypothesis
2. **K optimization**: Exp5 vs Exp6 vs Exp7 → Find optimal k (expect k=3 or k=4)
3. **Embedding ablation**: Exp8 vs Baseline → Pure MPNet impact (no chunk change)
4. **Chunk size ablation**: Exp5 vs Exp9 vs Exp10 → Find optimal chunk size

**Expected Output:**
- **Winning configuration** for e-commerce RAG
- **Clear understanding** of each parameter's impact
- **Quantified trade-offs** (precision vs recall, latency vs quality)
