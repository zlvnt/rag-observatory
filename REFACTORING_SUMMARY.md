# z3_core Refactoring Summary

**Task:** Task 1 - Simplify z3_core for Multi-Domain Flexibility
**Date:** October 11, 2025
**Status:** ✅ Completed

---

## Changes Made

### 1. **vector.py**
- ✅ Removed globals `_DOCS_DIR` dan `_VEC_DIR`
- ✅ Removed dependency ke `settings`
- ✅ Removed `@lru_cache`
- ✅ Added parameters: `docs_dir`, `vector_dir`, `embedding_model`, `chunk_size`, `chunk_overlap`, `k`
- ✅ Semua fungsi sekarang menerima config via parameters

### 2. **rag.py**
- ✅ Added `retriever` parameter (optional)
- ✅ Added `vector_dir` dan `embedding_model` parameters
- ✅ `relevance_threshold` sudah configurable
- ✅ Web search dijadikan optional (graceful fallback)
- ✅ Removed hardcoded imports

### 3. **router.py**
- ✅ Removed global `_SUPERVISOR_PROMPT`
- ✅ Removed `@lru_cache`
- ✅ Added `supervisor_prompt_path` parameter
- ✅ Added `model_name` dan `temperature` parameters
- ✅ Removed platform-specific functions (`handle`, `_record_routing_decision`)

### 4. **reply.py**
- ✅ Consolidated ke single `generate_reply()` function
- ✅ Removed `generate_telegram_reply()`
- ✅ Removed platform-specific parameters (`post_id`, `comment_id`, `username`)
- ✅ Removed `save_conv()` calls
- ✅ Added `personality_config_path` parameter (optional)
- ✅ Added `model_name` dan `temperature` parameters

### 5. **domain_config.py** (NEW)
- ✅ Created `DomainConfig` dataclass
- ✅ Implemented `from_yaml()` class method
- ✅ Implemented `load_domain_config()` helper function
- ✅ Added path validation
- ✅ Added `list_available_domains()` utility

### 6. **Example Config**
- ✅ Created `configs/example_config.yaml` template

---

## Usage Example

```python
from z3_core.domain_config import load_domain_config
from z3_core.vector import build_index, get_retriever
from z3_core.rag import retrieve_context
from z3_core.reply import generate_reply

# Load config untuk domain tertentu
config = load_domain_config('example')

# Build vector index
build_index(
    docs_dir=config.knowledge_base_dir,
    vector_dir=config.vector_store_dir,
    embedding_model=config.embedding_model
)

# Get retriever
retriever = get_retriever(
    vector_dir=config.vector_store_dir,
    embedding_model=config.embedding_model
)

# Retrieve context
context = retrieve_context(
    query="What is Product A?",
    retriever=retriever,
    relevance_threshold=config.relevance_threshold
)

# Generate reply
reply = generate_reply(
    query="What is Product A?",
    context=context,
    conversation_history="",  # Optional: previous conversation
    personality_config_path=config.personality_config_path,
    model_name=config.llm_model,
    temperature=config.llm_temperature
)
```

---

## Config File Format

`configs/{domain_name}_config.yaml`:

```yaml
domain_name: example
knowledge_base_dir: knowledge_bases/example/
vector_store_dir: data/vector_stores/example/
embedding_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
llm_model: gemini-pro
llm_temperature: 0.7
relevance_threshold: 0.8
chunk_size: 700
chunk_overlap: 100
retrieval_k: 4
```

---

## Environment Variables

Hanya perlu satu environment variable:

```bash
GEMINI_API_KEY=your_api_key_here
```

---

## Debug Info Capture (for Test Runner)

Both `retrieve_context()` and `generate_reply()` support optional debug info return:

```python
# Retrieve with debug info
context, retrieval_debug = retrieve_context(
    query="What is Product A?",
    retriever=retriever,
    return_debug_info=True  # Returns tuple
)

# retrieval_debug contains:
# - docs_retrieved: [{"source": "...", "relevance_score": 0.85, ...}]
# - num_docs_initial: 4
# - num_docs_final: 3
# - retrieval_mode: "docs"

# Generate with debug info
reply, generation_debug = generate_reply(
    query="What is Product A?",
    context=context,
    return_debug_info=True  # Returns tuple
)

# generation_debug contains:
# - final_prompt: "Exact prompt sent to LLM..."
# - prompt_tokens_approx: 1847
# - template_used: "path/to/config.json"
# - context_length: 456
```

This enables test runner to capture full pipeline trace for analysis.

---

## Next Steps

1. Buat knowledge base di `knowledge_bases/{domain}/`
2. Buat config YAML di `configs/`
3. Build vector index dengan `build_index()`
4. Create golden datasets
5. Build test runner dengan debug info capture
6. Mulai testing!
