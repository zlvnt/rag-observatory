# Session Handoff - RAG Observatory Project

**From:** Claude Code Session (Oct 13, 2025)
**To:** Next Claude Code Session
**Project:** RAG Observatory - Multi-domain RAG Evaluation Framework

---

## Current Status Summary

### ✅ Completed Tasks

#### **Phase 1: Foundation (Task 1) - COMPLETED**
- Refactored `z3_core/` modules (vector.py, rag.py, router.py, reply.py)
- Removed all hardcoded paths and globals
- Removed platform-specific code (Instagram/Telegram)
- Added `domain_config.py` for multi-domain configuration
- Added debug info capture (`return_debug_info=True`) for test runner
- All functions now accept parameters instead of reading from settings

**Key Achievement:** z3_core is now fully decoupled and flexible for multi-domain testing

#### **Phase 2: Test Data (Task 2) - COMPLETED**
- Golden dataset created: `golden_datasets/z3_agent_tests.json`
- Contains test cases with expected_route, expected_docs, expected_keywords

#### **Phase 3: Test Execution (Task 3) - COMPLETED**
- Created `evaluators/metrics.py` - metric calculations (Precision@K, Recall@K, MRR, etc.)
- Created `runners/test_runner.py` - full test execution pipeline
- Created `configs/z3_agent_config.yaml` - domain configuration for z3-agent
- Created `TESTING_GUIDE.md` - complete testing documentation

**Key Achievement:** Test runner ready to execute full RAG pipeline evaluation

---

## Project Structure

```
rag-observatory/
├── z3_core/                    # Refactored RAG engine
│   ├── vector.py              # Vector store (configurable)
│   ├── rag.py                 # Context retrieval (configurable)
│   ├── router.py              # Query routing (configurable)
│   ├── reply.py               # Response generation (configurable)
│   └── domain_config.py       # Domain config management
├── configs/
│   ├── example_config.yaml
│   └── z3_agent_config.yaml   # ✅ Ready for testing
├── golden_datasets/
│   └── z3_agent_tests.json    # ✅ Test cases ready
├── runners/
│   └── test_runner.py         # ✅ Main test execution script
├── evaluators/
│   └── metrics.py             # ✅ Metrics calculator
├── docs/                       # Knowledge base from z3-agent
├── content/                    # Prompts & personalities
├── results/                    # Output directory (will be created)
├── CLAUDE.md                   # Project instructions (updated with phases)
├── REFACTORING_SUMMARY.md      # Task 1 summary
└── TESTING_GUIDE.md            # How to run tests
```

---

## What's Ready to Use

### Test Runner Usage:
```bash
# Run all tests
python runners/test_runner.py --domain z3_agent --output results/

# Run with verbose (see prompts)
python runners/test_runner.py --domain z3_agent --output results/ --verbose

# Run limited tests (debugging)
python runners/test_runner.py --domain z3_agent --output results/ --limit 5
```

### Expected Output:
1. **Console:** Progress bar dengan real-time status
2. **Detailed JSON:** `results/detailed/*.json` (per-query pipeline trace)
3. **Summary CSV:** `results/summary_TIMESTAMP.csv` (for analysis)
4. **Report:** `results/report_TIMESTAMP.txt` (human-readable)

### Key Metrics Tracked:
- Routing Accuracy
- Precision@3, Recall@3, F1
- Mean Reciprocal Rank (MRR)
- Keyword Coverage
- Latency (avg, P50, P95, P99)
- Success Rate

---

## Next Steps (Not Started Yet)

### Phase 4: Analysis & Iteration
- Run first full test
- Analyze results
- Identify issues (low precision/recall/coverage)
- Iterate on:
  - Knowledge base (add missing docs)
  - Configuration (threshold, k, prompts)
  - Golden dataset (add edge cases)

### Phase 5: Multi-Domain Expansion (Future)
- Add 2-3 more domains
- Compare cross-domain performance
- Final report & recommendations

---

## Important Context for Next Session

### User's Testing Environment:
- **Original z3-agent codebase:** Located in `app/` (production code, don't modify)
- **Evaluation codebase:** `z3_core/` (refactored, decoupled from production)
- **Golden dataset:** Already created by user with Claude browser session
- **Knowledge base:** `docs/` - 4 markdown files from z3-agent

### Key Design Decisions Made:

1. **Debug Info Pattern:**
   - Both `retrieve_context()` and `generate_reply()` support `return_debug_info=True`
   - Returns tuple `(result, debug_info)` for test runner
   - Keeps all print statements from original (for human readability)
   - Pattern: "Print for human, return for program"

2. **Not Backward Compatible:**
   - z3_core is evaluation fork, NOT drop-in replacement for app/
   - This is intentional - evaluation code separate from production

3. **Metrics Priority:**
   - P0: Routing, Precision@3, Recall@3, Keyword Coverage
   - P1: MRR, Latency, Success Rate
   - P2: F1, NDCG (nice to have)

### User Preferences:
- Bahasa Indonesia OK for communication
- Likes clear explanations with concrete examples
- Wants to understand "how" before implementing
- Prefers not too verbose summaries

---

## Likely First Question from User

**"Bagaimana cara run test nya?"**

Answer:
```bash
# Quick test (5 queries)
python runners/test_runner.py --domain z3_agent --output results/ --limit 5

# Full test
python runners/test_runner.py --domain z3_agent --output results/
```

Then guide them to check:
1. Console output (progress bar)
2. `results/report_TIMESTAMP.txt` (human-readable summary)
3. `results/summary_TIMESTAMP.csv` (for detailed analysis)

---

## Files to Reference

- **CLAUDE.md** - Overall project plan (updated with phases)
- **TESTING_GUIDE.md** - Complete testing documentation
- **REFACTORING_SUMMARY.md** - Task 1 summary
- **prompt/step1.md** - Original refactoring instructions (reference only)
- **prompt/prclaude.md** - RAG testing strategy document

---

## Environment Setup

User needs:
- ✅ Python environment with dependencies installed
- ✅ `GEMINI_API_KEY` environment variable set
- ✅ All files in place (configs, golden dataset, knowledge base)

Vector store will be auto-built on first run if not exists.

---

## Common Issues to Watch For

1. **Vector store building:** First run will take time to build FAISS index
2. **API rate limits:** Gemini API calls for each query (watch quotas)
3. **Path issues:** All paths in config are relative to project root

---

## Communication Style with This User

- Use Bahasa Indonesia when appropriate (user is comfortable with it)
- Provide concrete examples before explaining concepts
- Ask clarifying questions when requirements unclear
- Don't over-explain if user just wants to proceed
- Use emojis sparingly (only when user uses them first)

---

## Last Context

User was about to run the test runner for the first time. Session ended due to context window limit before first test execution.

**Immediate action for next session:** Help user run first test and interpret results.

---

**Good luck with the testing phase! 🚀**

*End of handoff. Delete this file after reading.*
