# Phase 10: Threshold Calibration Study

**Created:** November 23, 2025
**Status:** Planned (belum dimulai)

---

## Overview

Experiment untuk mencari optimal BGE reranker score threshold. Melanjutkan dari threshold mixed experiment yang dilakukan di agentic-rag (threshold 1.0 → 0.786 quality).

## Background

### BGE Reranker Score
- Cross-encoder neural network (BAAI/bge-reranker-base)
- Score range: negatif sampai positif (bukan 0-1)
- Score interpretation:
  - Negatif (< 0): Tidak relevan
  - Sekitar 0: Borderline
  - Positif (> 0): Relevan
  - Tinggi (> 1.0): Sangat relevan/confident

### Previous Finding (agentic-rag)
- Threshold 1.0 → 0.786 validated quality
- Dengan adaptive fallback mechanism

---

## Experiment Design

### Threshold Values to Test

| Experiment | Threshold | Expected Behavior |
|------------|-----------|-------------------|
| exp10a | 0.5 | Lenient - lebih banyak chunk lolos |
| exp10b | 1.0 | Balanced (baseline) |
| exp10c | 2.0 | Strict - hanya chunk confident |
| exp10d | 3.0 | Very strict - hanya chunk sangat confident |

### Trade-offs

| Threshold | Precision | Recall | Noise |
|-----------|-----------|--------|-------|
| Rendah (0.5) | Lower | Higher | More noise |
| Tinggi (3.0) | Higher | Lower | Less noise |

### Base Configuration

```yaml
embedding_model: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
chunk_size: 500
chunk_overlap: 50
retrieval_k: 7

use_reranker: true
reranker_model: BAAI/bge-reranker-base
reranker_top_k: 3
reranker_use_fp16: true
reranker_threshold: [VARIABLE]  # 0.5, 1.0, 2.0, 3.0

use_hybrid_search: false
```

---

## Metrics to Collect

### Automated Metrics
- Precision
- Recall
- F1 Score
- MRR
- Avg chunks per query
- Queries with fallback (no chunks pass threshold)

### Manual Quality Assessment
- Quality score per query
- Failure patterns
- Context completeness

**IMPORTANT:** Manual quality assessment adalah yang paling penting! Automated metrics bisa misleading (lesson dari exp9a1).

---

## Questions to Answer

1. **Optimal threshold:** Threshold berapa yang menghasilkan quality terbaik?
2. **Precision vs Recall:** Bagaimana trade-off di berbagai threshold?
3. **Fallback frequency:** Seberapa sering adaptive fallback digunakan?
4. **Sweet spot:** Di mana balance antara strictness dan coverage?

---

## Expected Outcomes

### Hypothesis
- Threshold terlalu rendah (0.5) → noise meningkat, quality turun
- Threshold terlalu tinggi (3.0) → miss relevant chunks, quality turun
- Sweet spot kemungkinan di range 1.0 - 2.0

### Success Criteria
- Identify threshold dengan quality ≥ 0.786 (baseline)
- Atau temukan threshold yang lebih baik dari 1.0

---

## Implementation Plan

1. [ ] Create config files untuk setiap threshold (exp10a, exp10b, exp10c, exp10d)
2. [ ] Pastikan adaptive fallback logic sudah di rag.py
3. [ ] Run experiments dengan test_runner.py
4. [ ] Collect automated metrics
5. [ ] Manual quality assessment untuk setiap experiment
6. [ ] Compare results dan identify optimal threshold
7. [ ] Document findings

---

## Notes

- Experiment ini dilakukan di **rag-observatory** (bukan agentic-rag)
- Menggunakan golden dataset yang sama (30 queries)
- Fokus pada retrieval quality, bukan end-to-end generation

---

*Experiment ini melanjutkan research setelah Phase 9B (Hybrid Search rejection).*
