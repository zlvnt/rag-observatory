# When Precision Lies: The Gap Between RAG Metrics and Real Quality

> **TL;DR:** Our RAG system achieved 0.828 precision — a number that would make any ML engineer proud. But when we manually evaluated every single query, actual quality was only 58%. This document explores why automated metrics can be dangerously misleading and what to do about it.

---

## Table of Contents

- [The Discovery](#the-discovery)
- [The Numbers](#the-numbers)
- [Case Studies: Where Metrics Lied](#case-studies-where-metrics-lied)
- [Why This Happens](#why-this-happens)
- [Implications for RAG Systems](#implications-for-rag-systems)
- [Recommendations](#recommendations)

---

## The Discovery

After 20+ experiments optimizing a RAG system for Indonesian e-commerce customer service, we achieved what looked like success:

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Precision@3 | **0.828** | ≥ 0.80 | ✅ Exceeded |
| Recall@3 | 0.950 | ≥ 0.90 | ✅ Exceeded |
| F1 Score | 0.845 | ≥ 0.75 | ✅ Exceeded |
| MRR | 0.950 | ≥ 0.80 | ✅ Exceeded |

All green. Target exceeded. Ship it, right?

**Not so fast.**

Something felt off during development. Retrieved chunks often looked "correct" by document name, but the actual text inside wasn't quite answering the question. We decided to do something that most RAG evaluations skip: **manually evaluate every single query**.

What we found was sobering.

---

## The Numbers

### Automated vs Manual Evaluation

| Evaluation Method | Score | What It Measures |
|-------------------|-------|------------------|
| **Automated Precision** | 82.8% | "Did we retrieve the right document?" |
| **Manual Quality** | 58.67% | "Did the user get a complete, correct answer?" |

**A 24-point gap.**

Nearly a quarter of our "successes" were actually failures when viewed from the user's perspective.

### Quality Score Distribution

We evaluated all 30 queries on a 0-1 scale:
- **1.0** = All required information present
- **0.5** = Partial information (some missing)
- **0** = Critical information missing or wrong

| Score Range | Count | Percentage | Meaning |
|-------------|-------|------------|---------|
| 1.0 (Perfect) | 13 | 43% | Complete answer |
| 0.5 - 0.7 (Partial) | 4 | 13% | Missing pieces |
| 0 - 0.1 (Failed) | 10 | 33% | Wrong or missing |
| N/A (Edge cases) | 3 | 10% | Out of scope |

**One-third of queries completely failed** — despite metrics saying we're at 83% precision.

---

## Case Studies: Where Metrics Lied

### Case 1: The Phone Number That Wasn't There

**Query:** "Nomor customer service TokoPedia berapa?"  
*(What's TokoPedia's customer service number?)*

**What We Retrieved:**
```markdown
### Community Forum
- **URL**: community.tokopedia.com
- **Best for**: Tips, diskusi, sharing pengalaman
- **Moderator response**: 1-2 hari kerja

## Level 2: Customer Service

### Live Chat
- **Jam operasional**: 08:00 - 22:00 WIB
- **Queue time**: 5-15 menit (peak: 30 menit)
```

**What User Actually Needed:** `1500-600` (the phone number)

**Automated Metric Said:** ✅ Correct! (Retrieved from `contact_escalation.md`)

**Reality:** ❌ Complete failure. User asked for phone number, got forum URL and live chat hours.

**Quality Score:** 0

---

### Case 2: The "Almost Right" Trap

**Query:** "Response time CS berapa lama untuk kasus high priority?"  
*(How long is CS response time for high priority cases?)*

**What We Retrieved:**
```markdown
### Kategori LOW (Response <48 jam):
- Saran & feedback
- Pertanyaan umum
- Request fitur
- Report bug minor
```

**What User Actually Needed:** HIGH priority response time (which was just 2 lines above in the source document)

**Automated Metric Said:** ✅ Correct! (Retrieved from correct document, correct section about response times)

**Reality:** ❌ User asked about HIGH, got LOW. Literally the opposite information.

**Our Note:** *"Meleset dikit, sekitar dua poin diatas nya"* (Slightly off, about two bullet points above)

**Quality Score:** 0

---

### Case 3: Procedural Truncation

**Query:** "Bagaimana cara return barang yang rusak?"  
*(How do I return a damaged item?)*

**What We Retrieved:**
```markdown
### Produk yang BISA di-return:
- Produk rusak/cacat dari pabrik
- Produk tidak sesuai deskripsi/foto
- Produk salah kirim (warna/ukuran/model)
...

### Step 3: Kirim Balik Produk
**Packaging:**
- Gunakan packaging original jika ada
- Sertakan semua aksesoris/bonus
```

**What User Actually Needed:** Complete procedure (Step 1 → Step 2 → Step 3 → Step 4)

**Automated Metric Said:** ✅ Correct! (Retrieved from `policy_returns.md`)

**Reality:** ❌ User got "what CAN be returned" (not how) and jumped straight to Step 3 (missing Steps 1-2)

**Our Note:** *"prosedural, kurang lengkap"* (procedural, incomplete)

**Quality Score:** 0

---

### Case 4: When Even Easy Questions Fail

**Query:** "Kenapa OTP tidak masuk ke HP saya?"  
*(Why isn't my OTP arriving on my phone?)*

**What We Retrieved:**
```markdown
## Masalah Login dan Akun

### Tidak bisa login ke akun
**Gejala:** Error "Username/password salah" atau "Akun tidak ditemukan"

**Solusi:**
1. Pastikan email/nomor HP sudah benar
2. Cek huruf besar/kecil pada password
3. Gunakan fitur "Lupa Password"

### Tidak menerima OTP
**Gejala:** Kode OTP tidak masuk ke SMS/email
```

**What User Actually Needed:** The solution steps under "Tidak menerima OTP" section

**Automated Metric Said:** ✅ Correct! (Retrieved from `troubleshooting_guide.md`, OTP section visible)

**Reality:** ⚠️ Partial. The header "Tidak menerima OTP" is there, but the text was truncated before the actual solution steps.

**Quality Score:** 0.1

---

### Case 5: A Success Story (For Comparison)

**Query:** "Berapa lama batas waktu return untuk produk elektronik?"  
*(What's the return deadline for electronics?)*

**What We Retrieved:**
```markdown
### Batas waktu pengajuan return:
- **Elektronik**: 7 hari sejak diterima
- **Fashion**: 3 hari sejak diterima
- **Lainnya**: 2 hari sejak diterima
```

**What User Actually Needed:** Exactly this information.

**Automated Metric Said:** ✅ Correct!

**Reality:** ✅ Actually correct! Complete, accurate answer.

**Quality Score:** 1.0

---

## Why This Happens

### The Document vs Subsection Problem

Traditional RAG metrics evaluate at the **document level**:
- Did we retrieve `policy_returns.md`? → Yes → ✅ Correct

But users need answers at the **subsection level**:
- Did we retrieve the specific paragraph about return deadlines? → Sometimes no → ❌ Failure

This creates a systematic blind spot.

### The Three Failure Patterns

After analyzing all 30 queries, we identified three recurring patterns:

#### Pattern 1: "Meleset Sedikit" (Slightly Off) — 30% of queries

**What Happens:** Retrieved chunk is from the correct document, even the correct section, but the wrong *subsection*. Often just 2-6 lines away from the right answer.

**Why Metrics Miss It:** Document-level matching says "correct document retrieved."

**Example:** User asks about HIGH priority response time, gets LOW priority response time. Same section, opposite information.

---

#### Pattern 2: Context Truncation — 40% of queries

**What Happens:** Answer requires multiple paragraphs (e.g., Step 1-4 of a procedure), but chunking cuts mid-content. User gets Step 3, missing Steps 1-2.

**Why Metrics Miss It:** The retrieved chunk IS from the correct document and contains SOME relevant keywords.

**Example:** "How to return?" retrieves "Step 3: Ship the product back" but misses "Step 1: Submit return request."

---

#### Pattern 3: Keyword Match, Context Mismatch — 20% of queries

**What Happens:** Retrieved text contains the keywords from the query, but in a different context than what the user needs.

**Why Metrics Miss It:** Embedding similarity is high because keywords match.

**Example:** User asks for "phone number," retrieves text that mentions "phone" in context of "phone verification" not "phone to call."

---

### Visualizing the Gap

```
User Query: "What's the CS phone number?"
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│              Document Level (Metrics)               │
│  ┌─────────────────────────────────────────────┐   │
│  │         contact_escalation.md                │   │
│  │  ┌──────────────────────────────────────┐   │   │
│  │  │ Section: Community Forum             │   │   │
│  │  │ Section: Live Chat        ◄── RETRIEVED   │  │
│  │  │ Section: Phone Support    ◄── NEEDED  │   │  │
│  │  │ Section: Email            │           │   │  │
│  │  └──────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Metric says: ✅ Correct document!                  │
│  Reality: ❌ Wrong section, user gets no phone #    │
└─────────────────────────────────────────────────────┘
```

---

## Implications for RAG Systems

### 1. Precision ≠ User Satisfaction

A precision score of 0.828 sounds great. But if one-third of your users get wrong or incomplete answers, your support tickets won't go down.

**The gap exists because:**
- Metrics measure retrieval correctness
- Users experience answer quality
- These are not the same thing

### 2. The "Easy Query" Trap

In our evaluation:
- **Easy queries** (straightforward, single-doc answers): 33% failure rate
- **Hard queries** (complex reasoning): Lower failure rate (counterintuitively)

Why? Easy queries have specific answers. "What's the phone number?" has exactly one correct answer. Any deviation = complete failure.

Hard queries often have multiple valid perspectives in retrieved content.

### 3. Automated Benchmarks Are Necessary But Not Sufficient

Standard RAG benchmarks (Precision, Recall, F1, MRR) are essential for:
- Comparing experiments
- Detecting regressions
- Quick iteration

But they cannot replace:
- Human evaluation of answer quality
- Testing on real user queries
- Qualitative analysis of failure modes

---

## Recommendations

### For Evaluation

#### 1. Add Manual Quality Checks

Don't just compute metrics — regularly sample and manually evaluate retrievals.

**Our Approach:**
- Evaluate all 30 test queries manually
- Score 0-1 based on answer completeness
- Document failure patterns in notes
- Time investment: ~2 hours per experiment

**When to do it:**
- Before any production deployment
- After significant architecture changes
- Periodically for production monitoring

---

#### 2. Evaluate at Subsection Level, Not Document Level

Instead of:
```python
# Document-level (standard)
correct = retrieved_doc in expected_docs
```

Consider:
```python
# Subsection-level (more accurate)
correct = expected_content in retrieved_text
```

This requires more effort (defining expected content, not just expected documents) but reveals true quality.

---

#### 3. Create Difficulty-Stratified Test Sets

Our dataset breakdown:
- Easy: 19 queries (simple, single-fact answers)
- Medium: 9 queries (multi-step or multi-doc)
- Hard: 2 queries (complex reasoning)

Easy queries are the canary in the coal mine. If precision is high but easy queries fail, your metrics are lying.

---

### For Architecture

#### 4. Consider Rerankers

Our single biggest improvement (+5.7% precision) came from adding a cross-encoder reranker:

```yaml
retrieval_k: 7           # Retrieve more candidates
use_reranker: true
reranker_model: BAAI/bge-reranker-base
reranker_top_k: 3        # Return fewer, better results
```

Rerankers operate at a more granular level than embedding similarity, helping catch "meleset sedikit" cases.

---

#### 5. Optimize Chunk Boundaries

Many of our failures came from chunks that cut mid-procedure or mid-section.

**What didn't work:**
- Markdown-header-based splitting (too granular)
- Parent-child chunking (documents too small)

**What worked:**
- RecursiveCharacterTextSplitter with 500 chars, 50 overlap
- Keeping procedural content intact

---

#### 6. Test Failure Patterns, Not Just Metrics

After identifying our three failure patterns, we could specifically test for them:

- **"Meleset sedikit"**: Check if retrieved text contains exact expected phrase, not just expected keywords
- **Context truncation**: For procedural queries, check if all steps are present
- **Keyword mismatch**: Check if query intent matches retrieved content intent

---

## Conclusion

**Automated metrics are a compass, not a GPS.**

They point you in the right direction for experimentation, but they can't tell you if you've actually arrived at a good user experience.

Our 0.828 precision looked like success. But manual evaluation revealed that one-third of users would get wrong or incomplete answers. That's not a system ready for production — it's a system that needs more work.

**The lesson:** Before shipping any RAG system, invest the time to manually evaluate a meaningful sample of queries. The gap between your metrics and reality might be bigger than you think.

---

## Methodology Notes

### Evaluation Criteria

**Quality Score (0-1):**
| Score | Meaning |
|-------|---------|
| 1.0 | All information needed to answer the query is present |
| 0.7 | Most information present, minor gaps |
| 0.5 | Partial information, significant gaps |
| 0.1 | Minimal relevant information |
| 0 | No relevant information or wrong information |

### Test Set

- **Total queries:** 30
- **Difficulty distribution:** Easy (19), Medium (9), Hard (2)
- **Domain:** Indonesian e-commerce customer service
- **Document types:** FAQ, Policy documents, Troubleshooting guides

### Time Investment

- Manual evaluation: ~2 hours per experiment
- Analysis and pattern identification: ~1 hour
- Total for thorough quality assessment: ~3 hours

This investment revealed a 24-point gap that automated metrics completely missed.

---

<p align="center">
  <i>"Not everything that counts can be counted, and not everything that can be counted counts."</i>
  <br>
  — William Bruce Cameron
</p>