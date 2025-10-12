Instruksi untuk Claude Code:

Context dari Browser Session:
Kita sedang develop RAG evaluation framework dengan multi-domain testing. Task 1 (simplify z3_core) sudah selesai. Sekarang persiapan Task 2 (golden dataset creation).
Critical insight: Kita perlu capture full pipeline trace termasuk final_prompt_to_llm untuk transparency dan debugging power.

Task: Update Result Structure
Goal: Capture complete pipeline: Query → Retrieval → Prompt Construction → Generation
Changes needed:
1. Update result logging structure
Tambahkan fields baru di result JSON:
python{
  "test_id": "ecom_easy_001",
  "query": "user question",
  "timestamp": "2025-01-11T...",
  
  "pipeline_trace": {
    "routing": {
      "decision": "docs",
      "latency_ms": 234
    },
    "retrieval": {
      "docs_retrieved": [
        {"source": "policy_returns.md", "score": 0.89, "chunk_id": 3}
      ],
      "num_docs": 3,
      "retrieved_context": "Full context text...",  # NEW
      "latency_ms": 145
    },
    "prompt_construction": {  # NEW SECTION
      "final_prompt_to_llm": "System: ...\n\nContext: ...\n\nUser: ...",
      "prompt_tokens": 1847,
      "template_used": "personality_ecommerce.json"
    },
    "generation": {
      "answer": "Generated response...",
      "answer_tokens": 89,
      "latency_ms": 1823
    }
  },
  
  "evaluation": {
    "routing_correct": true,
    "precision": 0.67,
    "recall": 1.0
  }
}
2. Capture mechanism
Di test runner, sebelum call LLM:

Capture constructed prompt (after context injection)
Log exact string yang dikirim ke LLM
Count tokens (approximate: len(prompt)/4)

3. Files affected
Likely need to modify:

Test runner script (future - belum ada)
z3_core/reply.py - add verbose mode untuk return prompt

Suggestion: Add optional verbose=True parameter ke generate_reply() yang return tuple: (answer, prompt_used)

Why This Matters:

Debugging: Lihat exact apa yang LLM lihat
Metrics: Evaluate context quality, grounding
Reproducibility: Replay exact scenarios
Analysis: Compare prompts across domains


Priority:
Medium - not blocking Task 2 (golden dataset), tapi perlu untuk Task 3 (test runner).
Bisa prep sekarang atau implement saat bikin runner nanti.

Questions for Claude Code:

Prefer implement sekarang atau wait sampai test runner?
Need clarification on any part?


