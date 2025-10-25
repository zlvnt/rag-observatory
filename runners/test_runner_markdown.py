"""
Markdown Splitter Test Runner - Phase 8C

Usage:
    python runners/test_runner_markdown.py --domain z3_agent_exp6_markdown --output results/exp6_markdown/
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

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from z3_core.domain_config import load_domain_config
from z3_core.vector import build_index, get_retriever
from z3_core.rag import retrieve_context
from evaluators.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_f1_score,
    aggregate_metrics,
    group_by_category,
)


class ProgressBar:
    #bar aja

    def __init__(self, total: int, prefix: str = "Progress"):
        self.total = total
        self.current = 0
        self.prefix = prefix

    def update(self, current: int, status: str = ""):
        self.current = current
        percent = (current / self.total) * 100
        filled = int(50 * current / self.total)
        bar = "█" * filled + "━" * (50 - filled)

        print(f"\r{self.prefix}: [{bar}] {current}/{self.total} ({percent:.1f}%) {status}", end="", flush=True)

        if current == self.total:
            print() 


def load_golden_dataset(dataset_path: Path) -> Dict[str, Any]:
    #Load golden dataset from JSON file.
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
    # Ensure vector store exists, build if necessary.

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
            chunk_overlap=config.chunk_overlap,
            use_markdown_splitter=True  # FORCE markdown splitter
        )
        print(f"✓ Vector store built successfully (MarkdownHeaderTextSplitter)")
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
   # Run a single test case - RETRIEVAL FOCUSED

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
        "expected_docs": test_case.get("expected_docs", []),
        "timestamp": datetime.now().isoformat(),
        "rag_config": {
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "embedding_model": config.embedding_model,
            "vector_store": "FAISS",
            "retrieval_k": config.retrieval_k,
            "relevance_threshold": config.relevance_threshold
        },
        "retrieval_trace": {},
        "evaluation": {}
    }

    # no routing, always use docs mode
    start_time = time.time()
    try:
        context, retrieval_debug = retrieve_context(
            query=query,
            retriever=retriever,
            mode="docs",  # Always docs mode
            k_docs=config.retrieval_k,
            relevance_threshold=config.relevance_threshold,
            return_debug_info=True
        )

        retrieval_latency = (time.time() - start_time) * 1000

        # Extract unique doc sources (filenames only)
        all_doc_sources = [doc["source"] for doc in retrieval_debug["docs_retrieved"]]
        unique_doc_names = []
        seen = set()
        for doc in all_doc_sources:
            # Extract just filename from path
            filename = doc.split("/")[-1] if "/" in doc else doc
            if filename not in seen:
                unique_doc_names.append(filename)
                seen.add(filename)

        # Calculate token count (rough estimate: 1 token ≈ 4 chars)
        context_tokens = len(context) // 4

        result["retrieval_trace"] = {
            "docs_retrieved": retrieval_debug["docs_retrieved"],  # Full chunk details
            "unique_docs": unique_doc_names,  # Unique filenames for metrics
            "num_chunks_retrieved": retrieval_debug["num_docs_final"],
            "num_unique_docs": len(unique_doc_names),
            "context_length_chars": len(context),
            "context_tokens_approx": context_tokens,
            "retrieved_context_preview": context[:500] + "..." if len(context) > 500 else context,
            "latency_ms": round(retrieval_latency, 2)
        }

    except Exception as e:
        print(f"✗ Retrieval failed: {e}")
        unique_doc_names = []
        result["retrieval_trace"] = {
            "error": str(e),
            "docs_retrieved": [],
            "unique_docs": [],
            "num_chunks_retrieved": 0,
            "num_unique_docs": 0,
            "context_tokens_approx": 0,
            "latency_ms": 0
        }

    # Evaluation
    expected_docs = test_case.get("expected_docs", [])

    # Calculate retrieval metrics using unique doc names
    precision = calculate_precision_at_k(unique_doc_names, expected_docs, k=3)
    recall = calculate_recall_at_k(unique_doc_names, expected_docs, k=3)
    rr = calculate_reciprocal_rank(unique_doc_names, expected_docs)

    # Calculate true positives, false positives, false negatives
    expected_set = set(expected_docs)
    retrieved_set = set(unique_doc_names)

    true_positives = list(expected_set.intersection(retrieved_set))
    false_positives = list(retrieved_set - expected_set)
    false_negatives = list(expected_set - retrieved_set)

    result["evaluation"] = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(calculate_f1_score(precision, recall), 3),
        "reciprocal_rank": round(rr, 3),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

    return result


def save_detailed_result(result: Dict[str, Any], output_dir: Path):
   # Save detailed result for a single test case.

    detailed_dir = output_dir / "detailed"
    detailed_dir.mkdir(parents=True, exist_ok=True)

    filename = detailed_dir / f"{result['test_id']}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def save_summary_csv(results: List[Dict[str, Any]], output_dir: Path, timestamp: str):

#Save summary results to CSV - RETRIEVAL FOCUSED.

    csv_file = output_dir / f"summary_{timestamp}.csv"

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "test_id", "category", "difficulty", "query",
            "expected_docs", "retrieved_docs", "num_chunks",
            "precision", "recall", "f1_score", "mrr",
            "true_positives", "false_positives", "false_negatives",
            "context_tokens", "latency_ms"
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            # Get retrieved docs
            unique_docs = result["retrieval_trace"].get("unique_docs", [])
            docs_str = "; ".join(unique_docs) if unique_docs else "None"

            # Get expected docs from golden dataset
            expected = "; ".join(result.get("expected_docs", []))

            # True/false positives/negatives
            tp = "; ".join(result["evaluation"].get("true_positives", []))
            fp = "; ".join(result["evaluation"].get("false_positives", []))
            fn = "; ".join(result["evaluation"].get("false_negatives", []))

            writer.writerow({
                "test_id": result["test_id"],
                "category": result["category"],
                "difficulty": result["difficulty"],
                "query": result["query"],
                "expected_docs": expected if expected else "None",
                "retrieved_docs": docs_str,
                "num_chunks": result["retrieval_trace"].get("num_chunks_retrieved", 0),
                "precision": result["evaluation"]["precision"],
                "recall": result["evaluation"]["recall"],
                "f1_score": result["evaluation"]["f1_score"],
                "mrr": result["evaluation"]["reciprocal_rank"],
                "true_positives": tp if tp else "None",
                "false_positives": fp if fp else "None",
                "false_negatives": fn if fn else "None",
                "context_tokens": result["retrieval_trace"].get("context_tokens_approx", 0),
                "latency_ms": result["retrieval_trace"].get("latency_ms", 0)
            })

    print(f"\n✓ Summary CSV saved: {csv_file}")


def generate_report(results: List[Dict[str, Any]], output_dir: Path, timestamp: str, config):
    # Generate human-readable report.

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

    # Find failed queries (low recall)
    failed = [r for r in results if r["evaluation"]["recall"] < 0.5]

    # Generate report
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("RAG RETRIEVAL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Domain: {config.domain_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Queries: {overall_metrics['total_queries']}\n\n")

        # RAG Configuration
        f.write("-" * 60 + "\n")
        f.write("RAG CONFIGURATION\n")
        f.write("-" * 60 + "\n")
        sample_config = results[0]["rag_config"]
        f.write(f"Chunk Size:         {sample_config['chunk_size']}\n")
        f.write(f"Chunk Overlap:      {sample_config['chunk_overlap']}\n")
        f.write(f"Embedding Model:    {sample_config['embedding_model'].split('/')[-1]}\n")
        f.write(f"Text Splitter:      MarkdownHeaderTextSplitter\n")
        f.write(f"Vector Store:       {sample_config['vector_store']}\n")
        f.write(f"Retrieval K:        {sample_config['retrieval_k']}\n")
        f.write(f"Relevance Threshold: {sample_config['relevance_threshold']}\n\n")

        f.write("-" * 60 + "\n")
        f.write("RETRIEVAL METRICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Avg Precision@3:   {overall_metrics['avg_precision']:.3f}\n")
        f.write(f"Avg Recall@3:      {overall_metrics['avg_recall']:.3f}\n")
        f.write(f"Avg F1 Score:      {overall_metrics['avg_f1']:.3f}\n")
        f.write(f"Mean Reciprocal Rank: {overall_metrics['mrr']:.3f}\n")
        f.write(f"Success Rate:      {overall_metrics['success_rate']*100:.1f}%\n\n")

        f.write(f"Avg Chunks/Query:  {overall_metrics['avg_chunks_per_query']:.1f}\n")
        f.write(f"Avg Tokens/Query:  {overall_metrics['avg_tokens_per_query']:.0f}\n\n")

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

        f.write(f"Status: Precision {status(overall_metrics['avg_precision'], 0.8, 0.7)} | ")
        f.write(f"Recall {status(overall_metrics['avg_recall'], 0.8, 0.7)} | ")
        f.write(f"F1 {status(overall_metrics['avg_f1'], 0.8, 0.7)}\n\n")

        # By difficulty
        f.write("-" * 60 + "\n")
        f.write("BY DIFFICULTY\n")
        f.write("-" * 60 + "\n")
        for diff in ["easy", "medium", "hard"]:
            if diff in difficulty_metrics:
                m = difficulty_metrics[diff]
                f.write(f"{diff.capitalize()} ({len(by_difficulty[diff])} queries):\n")
                f.write(f"  Precision: {m['avg_precision']:.3f} | Recall: {m['avg_recall']:.3f} | ")
                f.write(f"F1: {m['avg_f1']:.3f} | MRR: {m['mrr']:.3f}\n")
        f.write("\n")

        # By category
        f.write("-" * 60 + "\n")
        f.write("BY CATEGORY\n")
        f.write("-" * 60 + "\n")
        for cat, m in category_metrics.items():
            f.write(f"{cat.capitalize()} ({len(by_category[cat])} queries):\n")
            f.write(f"  Precision: {m['avg_precision']:.3f} | Recall: {m['avg_recall']:.3f} | ")
            f.write(f"F1: {m['avg_f1']:.3f}\n")
        f.write("\n")

        # Failed queries
        if failed:
            f.write("-" * 60 + "\n")
            f.write(f"LOW RECALL QUERIES ({len(failed)})\n")
            f.write("-" * 60 + "\n")
            for i, r in enumerate(failed, 1):
                f.write(f"{i}. {r['test_id']} ({r['difficulty']}): {r['query'][:60]}...\n")
                f.write(f"   Recall: {r['evaluation']['recall']:.2f} | Precision: {r['evaluation']['precision']:.2f}\n")
                if r['evaluation']['false_negatives']:
                    f.write(f"   Missing docs: {', '.join(r['evaluation']['false_negatives'])}\n")
                if r['evaluation']['false_positives']:
                    f.write(f"   Irrelevant docs: {', '.join(r['evaluation']['false_positives'])}\n")
            f.write("\n")

        # Recommendations
        f.write("-" * 60 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 60 + "\n")

        if overall_metrics['avg_precision'] < 0.8:
            f.write("⚠ Precision below target (0.8) - too many irrelevant docs retrieved\n")
            f.write("  → Consider: Higher relevance threshold, better chunking strategy\n")
        if overall_metrics['avg_recall'] < 0.7:
            f.write("⚠ Recall below target (0.7) - missing relevant docs\n")
            f.write("  → Consider: Increase k, lower relevance threshold, improve embeddings\n")
        if overall_metrics['avg_f1'] < 0.75:
            f.write("⚠ F1 score below target (0.75) - balance between precision and recall needed\n")
        if overall_metrics['avg_tokens_per_query'] > 2000:
            f.write("⚠ High token usage per query - consider smaller chunks or stricter filtering\n")
        if overall_metrics['latency_p95'] > 1000:
            f.write("⚠ P95 latency > 1s - retrieval may be slow\n")

        if (overall_metrics['avg_precision'] >= 0.8 and
            overall_metrics['avg_recall'] >= 0.7 and
            overall_metrics['avg_f1'] >= 0.75):
            f.write("✓ Retrieval performance meets targets!\n")

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

    # Load domain config (check Phase 8C v2, then v1, then root configs/)
    try:
        try:
            config = load_domain_config(args.domain, config_dir=Path("configs/experiments_phase8c_v2"))
            print(f"✓ Loaded domain config: {args.domain} (from experiments_phase8c_v2/)")
        except FileNotFoundError:
            try:
                config = load_domain_config(args.domain, config_dir=Path("configs/experiments_phase8c"))
                print(f"✓ Loaded domain config: {args.domain} (from experiments_phase8c/)")
            except FileNotFoundError:
                config = load_domain_config(args.domain)
                print(f"✓ Loaded domain config: {args.domain}")
    except Exception as e:
        print(f"✗ Failed to load domain config: {e}")
        return 1

    # Load golden dataset from config
    if config.golden_dataset:
        dataset_path = config.golden_dataset
    else:
        # Fallback: extract base domain for experiments
        base_domain = args.domain.split('_exp')[0]
        dataset_path = Path(f"golden_datasets/{base_domain}_tests.json")

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
        # Success = retrieved at least 1 relevant doc (recall > 0)
        if result["evaluation"]["recall"] > 0:
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
