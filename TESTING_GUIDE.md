# RAG Testing Guide

## Quick Start

### 1. Prepare Your Environment

Make sure you have:
- ✅ Golden dataset: `golden_datasets/z3_agent_tests.json`
- ✅ Domain config: `configs/z3_agent_config.yaml`
- ✅ Knowledge base: `docs/` (from z3-agent)
- ✅ Environment variable: `GEMINI_API_KEY` set

### 2. Run Tests

```bash
# Basic run
python runners/test_runner.py --domain z3_agent --output results/

# With verbose output (see prompts and detailed logs)
python runners/test_runner.py --domain z3_agent --output results/ --verbose

# Limit tests (for quick debugging)
python runners/test_runner.py --domain z3_agent --output results/ --limit 5
```

### 3. View Results

After running, you'll get:

```
results/
├── detailed/
│   ├── z3_easy_001.json      # Full pipeline trace per query
│   ├── z3_easy_002.json
│   └── ...
├── summary_20250111_153045.csv     # CSV for analysis
└── report_20250111_153045.txt      # Human-readable report
```

---

## Understanding the Output

### Progress Bar (Real-time)

```
Testing: [████████████████████--------] 15/25 (60%) | z3_easy_015 ✓
```

- Shows current progress
- ✓ = routing correct
- ✗ = routing failed

### Summary CSV

Open in Excel/Google Sheets for analysis:

| test_id | category | difficulty | routing_correct | precision | recall | keyword_coverage | total_latency_ms |
|---------|----------|------------|----------------|-----------|--------|------------------|------------------|
| z3_easy_001 | policy | easy | TRUE | 0.667 | 1.000 | 0.750 | 2202 |
| z3_easy_002 | contact | easy | TRUE | 1.000 | 1.000 | 1.000 | 1856 |

### Report File

```
=============================================================
RAG EVALUATION REPORT
=============================================================

Domain: z3_agent
Date: 2025-01-11 15:30:45
Total Queries: 25

------------------------------------------------------------
OVERALL METRICS
------------------------------------------------------------
Routing Accuracy:  92.0% (23/25)
Avg Precision@3:   0.780
Avg Recall@3:      0.820
Avg F1 Score:      0.799
Mean Reciprocal Rank: 0.850
Keyword Coverage:  71.0%
Success Rate:      88.0%

Latency (avg):     2100ms
Latency (P50):     1950ms
Latency (P95):     2800ms
Latency (P99):     3200ms

Status: Routing ✓ | Precision ✓ | Recall ✓ | Keywords ⚠

------------------------------------------------------------
BY DIFFICULTY
------------------------------------------------------------
Easy (10 queries):
  Precision@3: 0.850 | Recall@3: 0.900 | Keywords: 0.800
Medium (10 queries):
  Precision@3: 0.760 | Recall@3: 0.800 | Keywords: 0.700
Hard (5 queries):
  Precision@3: 0.650 | Recall@3: 0.700 | Keywords: 0.600

------------------------------------------------------------
BY CATEGORY
------------------------------------------------------------
Policy (8 queries):
  Precision@3: 0.820 | Recall@3: 0.850 | Keywords: 0.750
Contact (7 queries):
  Precision@3: 0.880 | Recall@3: 0.900 | Keywords: 0.800
Troubleshooting (10 queries):
  Precision@3: 0.680 | Recall@3: 0.750 | Keywords: 0.650

------------------------------------------------------------
FAILED/PROBLEMATIC QUERIES (2)
------------------------------------------------------------
1. z3_hard_003 (hard): Apakah bisa custom packaging untuk gift?...
   Issue: Routing incorrect
2. z3_medium_007 (medium): Bagaimana cara track pengiriman inter...
   Issue: Low recall (0.33)

------------------------------------------------------------
RECOMMENDATIONS
------------------------------------------------------------
⚠ Keyword coverage low - review prompt engineering
✓ Overall system performance meets targets
```

### Detailed JSON (Per Query)

```json
{
  "test_id": "z3_easy_001",
  "query": "Bagaimana cara return barang?",
  "category": "policy",
  "difficulty": "easy",
  "timestamp": "2025-01-11T15:30:22",

  "pipeline_trace": {
    "routing": {
      "decision": "docs",
      "expected": "docs",
      "correct": true,
      "latency_ms": 234
    },
    "retrieval": {
      "docs_retrieved": [
        {"source": "policy_returns.md", "relevance_score": 0.89, "rank": 0, "passed_threshold": true},
        {"source": "faq_shipping.md", "relevance_score": 0.65, "rank": 1, "passed_threshold": false}
      ],
      "num_docs_initial": 3,
      "num_docs_final": 2,
      "retrieved_context": "[Docs] Kebijakan Return: Anda dapat...",
      "latency_ms": 145
    },
    "prompt_construction": {
      "final_prompt": "Identity: z3 from Instagram CS...\nContext: [Docs]...",
      "prompt_tokens_approx": 1847,
      "template_used": "default"
    },
    "generation": {
      "answer": "Untuk return barang, Anda bisa menghubungi...",
      "latency_ms": 1823
    },
    "total_latency_ms": 2202
  },

  "evaluation": {
    "routing_correct": true,
    "precision": 0.667,
    "recall": 1.000,
    "reciprocal_rank": 1.000,
    "keyword_coverage": 0.750,
    "keywords_found": ["return", "7 hari", "CS"],
    "keywords_missing": ["refund"]
  }
}
```

---

## Metrics Explained

### Precision@3
**Question:** "Of the 3 docs retrieved, how many are actually relevant?"

```
Precision@3 = (Relevant docs in top-3) / 3

Good: > 0.80 (80% of retrieved docs are relevant)
OK:   > 0.70
Bad:  < 0.70
```

### Recall@3
**Question:** "Of all relevant docs, how many did we retrieve?"

```
Recall@3 = (Relevant docs retrieved) / (Total relevant docs)

Good: > 0.80 (found 80% of relevant docs)
OK:   > 0.70
Bad:  < 0.70
```

### MRR (Mean Reciprocal Rank)
**Question:** "How quickly do we find the first relevant doc?"

```
RR = 1 / (rank of first relevant doc)
MRR = average across all queries

Good: > 0.80 (relevant docs appear early)
OK:   > 0.60
Bad:  < 0.60
```

### Keyword Coverage
**Question:** "Does the answer contain expected information?"

```
Coverage = (Keywords found) / (Total expected keywords)

Good: > 0.80 (answer is complete)
OK:   > 0.70
Bad:  < 0.70
```

### Latency P95
**Question:** "What's the worst-case latency for 95% of queries?"

```
P95 = 95th percentile of all latencies

Good: < 2s
OK:   < 3s
Bad:  > 3s
```

---

## Common Issues & Solutions

### Issue: Vector store not found

```
⚠ Vector store not found, building...
```

**Solution:** The runner will automatically build it. Make sure:
- Knowledge base exists at `knowledge_bases/z3_agent/` or path in config
- You have documents (.md, .txt files) in the knowledge base

### Issue: Supervisor prompt not found

```
INFO: No supervisor prompt, defaulting to 'docs' mode
```

**Solution:**
- Create supervisor prompt file (optional)
- Or accept default "docs" routing for all queries

### Issue: Low precision/recall

**Possible causes:**
1. Knowledge base incomplete - missing relevant docs
2. Threshold too strict - lower `relevance_threshold` in config
3. Embedding model mismatch - rebuild vector store with correct model

**Debug:**
```bash
# Run with --verbose to see retrieval details
python runners/test_runner.py --domain z3_agent --output results/ --verbose
```

### Issue: Low keyword coverage

**Possible causes:**
1. Prompt engineering - improve personality config
2. Context quality - review retrieved docs
3. Expected keywords too specific - review golden dataset

**Debug:**
- Check `detailed/*.json` files
- Look at `prompt_construction.final_prompt`
- Compare with `generation.answer`

---

## Advanced Usage

### Test Specific Queries

Edit golden dataset to include only queries you want to test, or:

```bash
# Test first 5 queries only
python runners/test_runner.py --domain z3_agent --output results/ --limit 5
```

### Compare Different Configurations

```bash
# Test with threshold 0.8
python runners/test_runner.py --domain z3_agent --output results/threshold_08/

# Edit config: change relevance_threshold to 0.6

# Test with threshold 0.6
python runners/test_runner.py --domain z3_agent --output results/threshold_06/

# Compare summary CSVs
```

### Analyze Results in Python

```python
import pandas as pd

# Load summary
df = pd.read_csv("results/summary_20250111_153045.csv")

# Filter by difficulty
hard_queries = df[df["difficulty"] == "hard"]
print(f"Hard queries avg precision: {hard_queries['precision'].mean():.3f}")

# Find low performers
low_recall = df[df["recall"] < 0.5]
print(f"Queries with low recall:\n{low_recall[['test_id', 'query', 'recall']]}")
```

---

## Next Steps

After running tests:

1. **Review report** - understand overall performance
2. **Analyze failures** - check detailed JSONs for failed queries
3. **Iterate**:
   - Update knowledge base if docs missing
   - Adjust config (threshold, k, prompts)
   - Improve golden dataset (add edge cases)
4. **Re-run tests** to measure improvement

Happy testing! 🚀
