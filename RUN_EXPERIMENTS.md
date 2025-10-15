# How to Run Experiments

## Quick Reference

### Experiment 1: Lower Threshold (Fast - 1 min)
```bash
python runners/test_runner.py --domain z3_agent_exp1 --output results_exp1/
```
**Change:** threshold 0.8 → 0.3
**Expected:** Precision +10-15%
**Rebuild needed:** ❌ No (reuses baseline vector store)

---

### Experiment 2: Increase K (Fast - 1 min)
```bash
python runners/test_runner.py --domain z3_agent_exp2 --output results_exp2/
```
**Change:** k=4 → k=6, threshold=0.3
**Expected:** Recall +5% (multi-doc queries)
**Rebuild needed:** ❌ No (reuses baseline vector store)

---

### Experiment 3: Smaller Chunks (Slow - 5 min)
```bash
# Vector store will auto-rebuild with new chunk size
python runners/test_runner.py --domain z3_agent_exp3 --output results_exp3/
```
**Change:** chunk_size 700→500, overlap 100→50, k=6, threshold=0.3
**Expected:** Precision +5%
**Rebuild needed:** ✅ Yes (different chunk_size)

---

### Experiment 4: Better Embeddings (Slowest - 15 min)
```bash
# Will download mpnet model (~500MB) + rebuild vector store
python runners/test_runner.py --domain z3_agent_exp4 --output results_exp4/
```
**Change:** MiniLM-L12 → MPNet-base-v2, chunk=500, k=6, threshold=0.3
**Expected:** Precision +15-20%
**Rebuild needed:** ✅ Yes (different embedding model)

---

## Configuration Summary

| Experiment | threshold | k | chunk_size | overlap | embedding | Rebuild? |
|------------|-----------|---|------------|---------|-----------|----------|
| Baseline | 0.8 | 4 | 700 | 100 | MiniLM-L12 | Done ✅ |
| **Exp1** | **0.3** | 4 | 700 | 100 | MiniLM-L12 | ❌ |
| **Exp2** | 0.3 | **6** | 700 | 100 | MiniLM-L12 | ❌ |
| **Exp3** | 0.3 | 6 | **500** | **50** | MiniLM-L12 | ✅ |
| **Exp4** | 0.3 | 6 | 500 | 50 | **MPNet-v2** | ✅ |

---

## Run All Experiments (Sequential)

```bash
# Fast experiments first (no rebuild)
python runners/test_runner.py --domain z3_agent_exp1 --output results_exp1/
python runners/test_runner.py --domain z3_agent_exp2 --output results_exp2/

# Slow experiments (require rebuild)
python runners/test_runner.py --domain z3_agent_exp3 --output results_exp3/
python runners/test_runner.py --domain z3_agent_exp4 --output results_exp4/
```

**Total time:** ~25 minutes

---

## Compare Results

After running experiments, compare CSV summaries:

```bash
# Quick comparison - open all CSVs
ls results_*/summary_*.csv

# Or view specific metric columns
for dir in results_*/; do
  echo "=== $(basename $dir) ==="
  tail -n +2 $(ls $dir/summary_*.csv) | cut -d',' -f8,9,10 | head -5
done
```

---

## Expected Results Progression

**Baseline → Exp1 (Lower Threshold):**
- Precision: 0.706 → **~0.80** (+10-15%)
- Recall: 0.950 → ~0.950 (same)
- Why: Filter out low-scoring irrelevant docs

**Exp1 → Exp2 (Increase K):**
- Precision: ~0.80 → ~0.78 (-2%, slight dilution)
- Recall: 0.950 → **~1.00** (+5%)
- Why: Retrieve more docs = better multi-doc coverage

**Exp2 → Exp3 (Smaller Chunks):**
- Precision: ~0.78 → **~0.83** (+5%)
- Recall: ~1.00 → ~1.00 (same)
- Why: More granular chunks = less noise per chunk

**Exp3 → Exp4 (Better Embeddings):**
- Precision: ~0.83 → **~0.95** (+15%)
- Recall: ~1.00 → ~1.00 (same)
- Why: Better semantic understanding = fewer false positives

**Final Expected (Exp4):**
- Precision: **~0.95** (target: 0.80) ✅
- Recall: **~1.00** (target: 0.70) ✅
- F1: **~0.97** (target: 0.75) ✅

---

## Troubleshooting

### Experiment fails with "vector store not found"
```bash
# Vector store auto-builds on first run
# If stuck, manually delete and rerun:
rm -rf data/vector_stores/z3_agent_exp*/
python runners/test_runner.py --domain z3_agent_exp3 --output results_exp3/
```

### "Model not found" error (Exp4)
```bash
# MPNet model will auto-download (~500MB)
# Requires internet connection
# If download fails, try again or check HuggingFace status
```

### Compare results
```bash
# Open reports in text editor
cat results_baseline/report_*.txt
cat results_exp1/report_*.txt
cat results_exp2/report_*.txt
# ... etc
```

---

## Notes

- Each experiment outputs to separate folder for easy comparison
- Configs are in `configs/z3_agent_exp*.yaml`
- Baseline results in `results_baseline/`
- All experiments use same golden dataset (30 queries)
- No LLM calls = fast and free to run repeatedly

---

**Ready to start!** 🚀
