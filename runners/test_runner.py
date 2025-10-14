"""RAG Test Runner - Execute golden dataset tests and generate reports.

Usage:
    python runners/test_runner.py --domain z3_agent --output results/
"""

import json
import csv
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from z3_core.domain_config import load_domain_config
from z3_core.vector import build_index, get_retriever
from z3_core.rag import retrieve_context
from z3_core.router import supervisor_route
from z3_core.reply import generate_reply
from evaluators.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_keyword_coverage,
    aggregate_metrics,
    group_by_category,
)


class ProgressBar:
    """Simple progress bar for terminal output."""

    def __init__(self, total: int, prefix: str = "Progress"):
        self.total = total
        self.current = 0
        self.prefix = prefix

    def update(self, current: int, status: str = ""):
        """Update progress bar."""
        self.current = current
        percent = (current / self.total) * 100
        filled = int(50 * current / self.total)
        bar = "█" * filled + "━" * (50 - filled)

        print(f"\r{self.prefix}: [{bar}] {current}/{self.total} ({percent:.1f}%) {status}", end="", flush=True)

        if current == self.total:
            print()  # New line when complete


def load_golden_dataset(dataset_path: Path) -> Dict[str, Any]:
    """Load golden dataset from JSON file.

    Args:
        dataset_path: Path to golden dataset JSON file

    Returns:
        Dictionary with metadata and test_cases
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"✓ Loaded golden dataset: {dataset_path}")
    print(f"  Domain: {data['metadata']['domain']}")
    print(f"  Version: {data['metadata']['version']}")
    print(f"  Test cases: {len(data['test_cases'])}")

    return data


def ensure_vector_store(config) -> bool:
    """Ensure vector store exists, build if necessary.

    Args:
        config: Domain configuration

    Returns:
        True if vector store ready
    """
    index_path = config.vector_store_dir / "index.faiss"

    if index_path.exists():
        print(f"✓ Vector store found: {config.vector_store_dir}")
        return True

    print(f"⚠ Vector store not found, building...")
    print(f"  Docs dir: {config.knowledge_base_dir}")
    print(f"  Vector dir: {config.vector_store_dir}")

    try:
        build_index(
            docs_dir=config.knowledge_base_dir,
            vector_dir=config.vector_store_dir,
            embedding_model=config.embedding_model,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        print(f"✓ Vector store built successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to build vector store: {e}")
        return False


def run_single_test(
    test_case: Dict[str, Any],
    config,
    retriever,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run a single test case through the full RAG pipeline.

    Args:
        test_case: Test case from golden dataset
        config: Domain configuration
        retriever: Pre-built retriever
        verbose: Print detailed information

    Returns:
        Result dictionary with pipeline trace and evaluation
    """
    query = test_case["query"]
    test_id = test_case["id"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Test: {test_id}")
        print(f"Query: {query}")
        print(f"{'='*60}")

    result = {
        "test_id": test_id,
        "query": query,
        "category": test_case.get("category", "unknown"),
        "difficulty": test_case.get("difficulty", "unknown"),
        "timestamp": datetime.now().isoformat(),
        "pipeline_trace": {},
        "evaluation": {}
    }

    # Step 1: Routing
    start_time = time.time()
    try:
        if config.supervisor_prompt_path and config.supervisor_prompt_path.exists():
            routing_decision = supervisor_route(
                user_input=query,
                supervisor_prompt_path=config.supervisor_prompt_path,
                model_name=config.llm_model,
                temperature=0  # Use 0 for deterministic routing
            )
        else:
            # Default to docs if no supervisor prompt
            routing_decision = "docs"
            print(f"INFO: No supervisor prompt, defaulting to 'docs' mode")

        routing_latency = (time.time() - start_time) * 1000

        result["pipeline_trace"]["routing"] = {
            "decision": routing_decision,
            "expected": test_case.get("expected_route", "docs"),
            "correct": routing_decision == test_case.get("expected_route", "docs"),
            "latency_ms": round(routing_latency, 2)
        }

    except Exception as e:
        print(f"✗ Routing failed: {e}")
        result["pipeline_trace"]["routing"] = {
            "error": str(e),
            "decision": "docs",  # Fallback
            "latency_ms": 0
        }
        routing_decision = "docs"

    # Step 2: Retrieval
    start_time = time.time()
    try:
        context, retrieval_debug = retrieve_context(
            query=query,
            retriever=retriever,
            mode=routing_decision,
            k_docs=config.retrieval_k,
            relevance_threshold=config.relevance_threshold,
            return_debug_info=True
        )

        retrieval_latency = (time.time() - start_time) * 1000

        # Extract doc sources
        retrieved_doc_names = [doc["source"] for doc in retrieval_debug["docs_retrieved"]]

        result["pipeline_trace"]["retrieval"] = {
            "docs_retrieved": retrieval_debug["docs_retrieved"],
            "num_docs_initial": retrieval_debug["num_docs_initial"],
            "num_docs_final": retrieval_debug["num_docs_final"],
            "retrieved_context": context[:500] + "..." if len(context) > 500 else context,  # Truncate for storage
            "latency_ms": round(retrieval_latency, 2)
        }

    except Exception as e:
        print(f"✗ Retrieval failed: {e}")
        context = ""
        retrieved_doc_names = []
        retrieval_latency = 0
        result["pipeline_trace"]["retrieval"] = {
            "error": str(e),
            "docs_retrieved": [],
            "latency_ms": 0
        }

    # Step 3: Generation
    start_time = time.time()
    try:
        answer, generation_debug = generate_reply(
            query=query,
            context=context,
            conversation_history="",  # No history for golden dataset tests
            personality_config_path=config.personality_config_path,
            model_name=config.llm_model,
            temperature=config.llm_temperature,
            verbose=verbose,
            return_debug_info=True
        )

        generation_latency = (time.time() - start_time) * 1000

        result["pipeline_trace"]["prompt_construction"] = {
            "final_prompt": generation_debug["final_prompt"][:1000] + "..." if len(generation_debug["final_prompt"]) > 1000 else generation_debug["final_prompt"],
            "prompt_tokens_approx": generation_debug["prompt_tokens_approx"],
            "template_used": generation_debug["template_used"]
        }

        result["pipeline_trace"]["generation"] = {
            "answer": answer,
            "latency_ms": round(generation_latency, 2)
        }

    except Exception as e:
        print(f"✗ Generation failed: {e}")
        answer = ""
        generation_latency = 0
        result["pipeline_trace"]["generation"] = {
            "error": str(e),
            "answer": "",
            "latency_ms": 0
        }

    # Calculate total latency
    result["pipeline_trace"]["total_latency_ms"] = round(
        result["pipeline_trace"]["routing"]["latency_ms"] +
        result["pipeline_trace"]["retrieval"]["latency_ms"] +
        result["pipeline_trace"]["generation"]["latency_ms"],
        2
    )

    # Step 4: Evaluation
    expected_docs = test_case.get("expected_docs", [])

    # Calculate retrieval metrics
    precision = calculate_precision_at_k(retrieved_doc_names, expected_docs, k=3)
    recall = calculate_recall_at_k(retrieved_doc_names, expected_docs, k=3)
    rr = calculate_reciprocal_rank(retrieved_doc_names, expected_docs)

    # Calculate keyword coverage
    expected_keywords = test_case.get("expected_keywords", [])
    keyword_result = calculate_keyword_coverage(answer, expected_keywords)

    result["evaluation"] = {
        "routing_correct": result["pipeline_trace"]["routing"].get("correct", False),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "reciprocal_rank": round(rr, 3),
        "keyword_coverage": round(keyword_result["coverage"], 3),
        "keywords_found": keyword_result["found"],
        "keywords_missing": keyword_result["missing"]
    }

    return result


def save_detailed_result(result: Dict[str, Any], output_dir: Path):
    """Save detailed result for a single test case.

    Args:
        result: Test result dictionary
        output_dir: Output directory
    """
    detailed_dir = output_dir / "detailed"
    detailed_dir.mkdir(parents=True, exist_ok=True)

    filename = detailed_dir / f"{result['test_id']}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def save_summary_csv(results: List[Dict[str, Any]], output_dir: Path, timestamp: str):
    """Save summary results to CSV.

    Args:
        results: List of test results
        output_dir: Output directory
        timestamp: Timestamp string for filename
    """
    csv_file = output_dir / f"summary_{timestamp}.csv"

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "test_id", "category", "difficulty", "query",
            "routing_correct", "routing_decision",
            "docs_retrieved", "precision", "recall", "reciprocal_rank",
            "answer", "final_prompt_preview",
            "keyword_coverage", "keywords_missing", "total_latency_ms"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            # Get docs retrieved
            docs_retrieved = [doc["source"] for doc in result["pipeline_trace"]["retrieval"].get("docs_retrieved", [])]
            docs_str = "; ".join(docs_retrieved) if docs_retrieved else "None"

            # Get answer
            answer = result["pipeline_trace"]["generation"].get("answer", "")

            # Get final prompt (truncated)
            final_prompt = result["pipeline_trace"].get("prompt_construction", {}).get("final_prompt", "")
            prompt_preview = final_prompt[:200] + "..." if len(final_prompt) > 200 else final_prompt

            # Get keywords missing
            keywords_missing = result["evaluation"].get("keywords_missing", [])
            keywords_str = "; ".join(keywords_missing) if keywords_missing else "None"

            writer.writerow({
                "test_id": result["test_id"],
                "category": result["category"],
                "difficulty": result["difficulty"],
                "query": result["query"],
                "routing_correct": result["evaluation"]["routing_correct"],
                "routing_decision": result["pipeline_trace"]["routing"].get("decision", "unknown"),
                "docs_retrieved": docs_str,
                "precision": result["evaluation"]["precision"],
                "recall": result["evaluation"]["recall"],
                "reciprocal_rank": result["evaluation"]["reciprocal_rank"],
                "answer": answer,
                "final_prompt_preview": prompt_preview,
                "keyword_coverage": result["evaluation"]["keyword_coverage"],
                "keywords_missing": keywords_str,
                "total_latency_ms": result["pipeline_trace"]["total_latency_ms"]
            })

    print(f"\n✓ Summary CSV saved: {csv_file}")


def generate_report(results: List[Dict[str, Any]], output_dir: Path, timestamp: str, config):
    """Generate human-readable report.

    Args:
        results: List of test results
        output_dir: Output directory
        timestamp: Timestamp string for filename
        config: Domain configuration
    """
    report_file = output_dir / f"report_{timestamp}.txt"

    # Calculate aggregated metrics
    overall_metrics = aggregate_metrics(results)

    # Group by difficulty
    by_difficulty = group_by_category(results, "difficulty")
    difficulty_metrics = {
        diff: aggregate_metrics(tests)
        for diff, tests in by_difficulty.items()
    }

    # Group by category
    by_category = group_by_category(results, "category")
    category_metrics = {
        cat: aggregate_metrics(tests)
        for cat, tests in by_category.items()
    }

    # Find failed queries
    failed = [r for r in results if not r["evaluation"]["routing_correct"] or r["evaluation"]["recall"] < 0.5]

    # Generate report
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("RAG EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Domain: {config.domain_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Queries: {overall_metrics['total_queries']}\n\n")

        f.write("-" * 60 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Routing Accuracy:  {overall_metrics['routing_accuracy']*100:.1f}% ({overall_metrics['routing_correct_count']}/{overall_metrics['total_queries']})\n")
        f.write(f"Avg Precision@3:   {overall_metrics['avg_precision']:.3f}\n")
        f.write(f"Avg Recall@3:      {overall_metrics['avg_recall']:.3f}\n")
        f.write(f"Avg F1 Score:      {overall_metrics['avg_f1']:.3f}\n")
        f.write(f"Mean Reciprocal Rank: {overall_metrics['mrr']:.3f}\n")
        f.write(f"Keyword Coverage:  {overall_metrics['avg_keyword_coverage']*100:.1f}%\n")
        f.write(f"Success Rate:      {overall_metrics['success_rate']*100:.1f}%\n\n")

        f.write(f"Latency (avg):     {overall_metrics['latency_avg']:.0f}ms\n")
        f.write(f"Latency (P50):     {overall_metrics['latency_p50']:.0f}ms\n")
        f.write(f"Latency (P95):     {overall_metrics['latency_p95']:.0f}ms\n")
        f.write(f"Latency (P99):     {overall_metrics['latency_p99']:.0f}ms\n\n")

        # Status indicators
        def status(value, threshold_good, threshold_warn):
            if value >= threshold_good:
                return "✓"
            elif value >= threshold_warn:
                return "⚠"
            else:
                return "✗"

        f.write(f"Status: Routing {status(overall_metrics['routing_accuracy'], 0.9, 0.8)} | ")
        f.write(f"Precision {status(overall_metrics['avg_precision'], 0.8, 0.7)} | ")
        f.write(f"Recall {status(overall_metrics['avg_recall'], 0.8, 0.7)} | ")
        f.write(f"Keywords {status(overall_metrics['avg_keyword_coverage'], 0.8, 0.7)}\n\n")

        # By difficulty
        f.write("-" * 60 + "\n")
        f.write("BY DIFFICULTY\n")
        f.write("-" * 60 + "\n")
        for diff in ["easy", "medium", "hard"]:
            if diff in difficulty_metrics:
                m = difficulty_metrics[diff]
                f.write(f"{diff.capitalize()} ({len(by_difficulty[diff])} queries):\n")
                f.write(f"  Precision@3: {m['avg_precision']:.3f} | Recall@3: {m['avg_recall']:.3f} | ")
                f.write(f"Keywords: {m['avg_keyword_coverage']:.3f}\n")
        f.write("\n")

        # By category
        f.write("-" * 60 + "\n")
        f.write("BY CATEGORY\n")
        f.write("-" * 60 + "\n")
        for cat, m in category_metrics.items():
            f.write(f"{cat.capitalize()} ({len(by_category[cat])} queries):\n")
            f.write(f"  Precision@3: {m['avg_precision']:.3f} | Recall@3: {m['avg_recall']:.3f} | ")
            f.write(f"Keywords: {m['avg_keyword_coverage']:.3f}\n")
        f.write("\n")

        # Failed queries
        if failed:
            f.write("-" * 60 + "\n")
            f.write(f"FAILED/PROBLEMATIC QUERIES ({len(failed)})\n")
            f.write("-" * 60 + "\n")
            for i, r in enumerate(failed, 1):
                f.write(f"{i}. {r['test_id']} ({r['difficulty']}): {r['query'][:60]}...\n")
                if not r["evaluation"]["routing_correct"]:
                    f.write(f"   Issue: Routing incorrect\n")
                if r["evaluation"]["recall"] < 0.5:
                    f.write(f"   Issue: Low recall ({r['evaluation']['recall']:.2f})\n")
            f.write("\n")

        # Recommendations
        f.write("-" * 60 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 60 + "\n")

        if overall_metrics['routing_accuracy'] < 0.85:
            f.write("⚠ Routing accuracy below target - review supervisor prompt\n")
        if overall_metrics['avg_precision'] < 0.8:
            f.write("⚠ Precision below target - review retrieval strategy\n")
        if overall_metrics['avg_recall'] < 0.7:
            f.write("⚠ Recall below target - consider increasing k or adjusting threshold\n")
        if overall_metrics['avg_keyword_coverage'] < 0.8:
            f.write("⚠ Keyword coverage low - review prompt engineering\n")
        if overall_metrics['latency_p95'] > 3000:
            f.write("⚠ P95 latency > 3s - optimize pipeline performance\n")

        if (overall_metrics['routing_accuracy'] >= 0.85 and
            overall_metrics['avg_precision'] >= 0.8 and
            overall_metrics['avg_recall'] >= 0.7):
            f.write("✓ Overall system performance meets targets\n")

        f.write("\n")
        f.write("=" * 60 + "\n")

    print(f"✓ Report saved: {report_file}")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="RAG Test Runner")
    parser.add_argument("--domain", required=True, help="Domain name (e.g., z3_agent)")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--limit", type=int, help="Limit number of tests (for debugging)")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("RAG EVALUATION TEST RUNNER")
    print("="*60 + "\n")

    # Load domain config
    try:
        config = load_domain_config(args.domain)
        print(f"✓ Loaded domain config: {args.domain}")
    except Exception as e:
        print(f"✗ Failed to load domain config: {e}")
        return 1

    # Load golden dataset
    dataset_path = Path(f"golden_datasets/{args.domain}_tests.json")
    try:
        dataset = load_golden_dataset(dataset_path)
    except Exception as e:
        print(f"✗ Failed to load golden dataset: {e}")
        return 1

    # Ensure vector store exists
    if not ensure_vector_store(config):
        return 1

    # Get retriever
    try:
        retriever = get_retriever(
            vector_dir=config.vector_store_dir,
            embedding_model=config.embedding_model,
            k=config.retrieval_k
        )
        print(f"✓ Retriever initialized")
    except Exception as e:
        print(f"✗ Failed to initialize retriever: {e}")
        return 1

    # Prepare output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get test cases
    test_cases = dataset["test_cases"]
    if args.limit:
        test_cases = test_cases[:args.limit]

    print(f"\n{'='*60}")
    print(f"Running {len(test_cases)} tests...")
    print(f"{'='*60}\n")

    # Run tests with progress bar
    results = []
    progress = ProgressBar(len(test_cases), prefix="Testing")

    for i, test_case in enumerate(test_cases):
        result = run_single_test(test_case, config, retriever, verbose=args.verbose)
        results.append(result)

        # Save detailed result
        save_detailed_result(result, output_dir)

        # Update progress
        status = f"| {test_case['id']} "
        if result["evaluation"]["routing_correct"]:
            status += "✓"
        else:
            status += "✗"
        progress.update(i + 1, status)

    print()

    # Save summary
    save_summary_csv(results, output_dir, timestamp)

    # Generate report
    generate_report(results, output_dir, timestamp, config)

    print(f"\n{'='*60}")
    print("TESTING COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}/")
    print(f"  - Detailed: {output_dir}/detailed/")
    print(f"  - Summary:  {output_dir}/summary_{timestamp}.csv")
    print(f"  - Report:   {output_dir}/report_{timestamp}.txt")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
