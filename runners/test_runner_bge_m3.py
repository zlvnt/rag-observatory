"""
BGE-M3 Multi-Functional Retrieval Test Runner

Usage:
    python runners/test_runner_bge_m3.py --config configs/experiments_phase8b/z3_agent_exp6_bge_full.yaml --output results/exp6_bge_full/
"""

import json
import csv
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import sys
from dotenv import load_dotenv
import yaml

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from z3_core.vector_bge_m3 import build_bge_m3_index, get_bge_m3_retriever
from evaluators.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    calculate_f1_score,
    aggregate_metrics,
    group_by_category,
)


class ProgressBar:
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


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    for key in ['knowledge_base_dir', 'vector_store_dir']:
        if key in config:
            config[key] = Path(config[key])

    config['golden_dataset'] = Path(config['golden_dataset'])

    print(f"✓ Loaded config: {config_path}")
    print(f"  Domain: {config['domain_name']}")
    print(f"  Embedding: {config['embedding_model']}")
    print(f"  k={config['retrieval_k']}, threshold={config['relevance_threshold']}")
    print(f"  Weights: dense={config['dense_weight']}, sparse={config['sparse_weight']}, colbert={config['colbert_weight']}")

    return config


def load_golden_dataset(dataset_path: Path) -> Dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"✓ Loaded golden dataset: {dataset_path}")
    print(f"  Domain: {data['metadata']['domain']}")
    print(f"  Version: {data['metadata']['version']}")
    print(f"  Test cases: {len(data['test_cases'])}")

    return data


def ensure_vector_store(config: Dict[str, Any]) -> bool:
    vector_dir = config['vector_store_dir']

    dense_path = vector_dir / "bge_m3_dense.npy"

    if dense_path.exists():
        print(f"✓ BGE-M3 vector store found: {vector_dir}")
        return True

    print(f"⚠ BGE-M3 vector store not found, building...")
    print(f"  Docs dir: {config['knowledge_base_dir']}")
    print(f"  Vector dir: {vector_dir}")

    try:
        build_bge_m3_index(
            docs_dir=config['knowledge_base_dir'],
            vector_dir=vector_dir,
            chunk_size=config['chunk_size'],
            chunk_overlap=config['chunk_overlap'],
            dense_weight=config['dense_weight'],
            sparse_weight=config['sparse_weight'],
            colbert_weight=config['colbert_weight']
        )
        print(f"✓ BGE-M3 vector store built successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to build vector store: {e}")
        raise


def run_single_test(
    test_case: Dict[str, Any],
    config: Dict[str, Any],
    retriever,
    verbose: bool = False
) -> Dict[str, Any]:
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
            "chunk_size": config['chunk_size'],
            "chunk_overlap": config['chunk_overlap'],
            "embedding_model": config['embedding_model'],
            "embedding_type": "bge_m3_multifunctional",
            "vector_store": "BGE-M3 Custom",
            "retrieval_k": config['retrieval_k'],
            "relevance_threshold": config['relevance_threshold'],
            "dense_weight": config['dense_weight'],
            "sparse_weight": config['sparse_weight'],
            "colbert_weight": config['colbert_weight']
        },
        "retrieval_trace": {},
        "evaluation": {}
    }

    start_time = time.time()
    try:
        retrieved_docs = retriever.retrieve(query, k=config['retrieval_k'])

        retrieval_latency = (time.time() - start_time) * 1000

        unique_doc_names = []
        seen = set()
        for doc in retrieved_docs:
            filename = doc.metadata.get('source', 'unknown').split("/")[-1]
            if filename not in seen:
                unique_doc_names.append(filename)
                seen.add(filename)

        combined_context = "\n\n[Docs]\n\n".join([doc.page_content for doc in retrieved_docs])
        context_tokens = len(combined_context) // 4

        result["retrieval_trace"] = {
            "unique_docs": unique_doc_names,
            "num_chunks_retrieved": len(retrieved_docs),
            "num_unique_docs": len(unique_doc_names),
            "context_length_chars": len(combined_context),
            "context_tokens_approx": context_tokens,
            "retrieved_context_preview": combined_context[:500] + "..." if len(combined_context) > 500 else combined_context,
            "latency_ms": round(retrieval_latency, 2)
        }

    except Exception as e:
        print(f"✗ Retrieval failed: {e}")
        unique_doc_names = []
        result["retrieval_trace"] = {
            "error": str(e),
            "unique_docs": [],
            "num_chunks_retrieved": 0,
            "num_unique_docs": 0,
            "context_tokens_approx": 0,
            "latency_ms": 0
        }

    expected_docs = test_case.get("expected_docs", [])

    precision = calculate_precision_at_k(unique_doc_names, expected_docs, k=3)
    recall = calculate_recall_at_k(unique_doc_names, expected_docs, k=3)
    rr = calculate_reciprocal_rank(unique_doc_names, expected_docs)

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
    detailed_dir = output_dir / "detailed"
    detailed_dir.mkdir(parents=True, exist_ok=True)

    filename = detailed_dir / f"{result['test_id']}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def save_summary_csv(results: List[Dict[str, Any]], output_dir: Path, timestamp: str):
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
            writer.writerow({
                "test_id": result["test_id"],
                "category": result["category"],
                "difficulty": result["difficulty"],
                "query": result["query"],
                "expected_docs": " | ".join(result["expected_docs"]),
                "retrieved_docs": " | ".join(result["retrieval_trace"]["unique_docs"]),
                "num_chunks": result["retrieval_trace"]["num_chunks_retrieved"],
                "precision": result["evaluation"]["precision"],
                "recall": result["evaluation"]["recall"],
                "f1_score": result["evaluation"]["f1_score"],
                "mrr": result["evaluation"]["reciprocal_rank"],
                "true_positives": " | ".join(result["evaluation"]["true_positives"]),
                "false_positives": " | ".join(result["evaluation"]["false_positives"]),
                "false_negatives": " | ".join(result["evaluation"]["false_negatives"]),
                "context_tokens": result["retrieval_trace"]["context_tokens_approx"],
                "latency_ms": result["retrieval_trace"]["latency_ms"]
            })

    print(f"✓ Summary saved: {csv_file}")


def generate_report(results: List[Dict[str, Any]], config: Dict[str, Any], output_dir: Path, timestamp: str):
    overall = aggregate_metrics(results)

    by_difficulty_grouped = group_by_category(results, "difficulty")
    by_difficulty = {k: aggregate_metrics(v) for k, v in by_difficulty_grouped.items()}

    by_category_grouped = group_by_category(results, "category")
    by_category = {k: aggregate_metrics(v) for k, v in by_category_grouped.items()}

    latencies = [r["retrieval_trace"]["latency_ms"] for r in results]
    tokens = [r["retrieval_trace"]["context_tokens_approx"] for r in results]

    report_file = output_dir / f"report_{timestamp}.txt"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("BGE-M3 MULTI-FUNCTIONAL RETRIEVAL EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Domain: {config['domain_name']}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Queries: {len(results)}\n\n")

        f.write("-" * 60 + "\n")
        f.write("RAG CONFIGURATION\n")
        f.write("-" * 60 + "\n")
        f.write(f"Chunk Size:         {config['chunk_size']}\n")
        f.write(f"Chunk Overlap:      {config['chunk_overlap']}\n")
        f.write(f"Embedding Model:    {config['embedding_model']}\n")
        f.write(f"Embedding Type:     BGE-M3 Multi-Functional\n")
        f.write(f"Vector Store:       Custom (Dense + Sparse + ColBERT)\n")
        f.write(f"Retrieval K:        {config['retrieval_k']}\n")
        f.write(f"Relevance Threshold: {config['relevance_threshold']}\n")
        f.write(f"Hybrid Weights:     Dense={config['dense_weight']}, Sparse={config['sparse_weight']}, ColBERT={config['colbert_weight']}\n\n")

        f.write("-" * 60 + "\n")
        f.write("RETRIEVAL METRICS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Avg Precision@3:   {overall['avg_precision']:.3f}\n")
        f.write(f"Avg Recall@3:      {overall['avg_recall']:.3f}\n")
        f.write(f"Avg F1 Score:      {overall['avg_f1']:.3f}\n")
        f.write(f"Mean Reciprocal Rank: {overall['mrr']:.3f}\n")
        f.write(f"Success Rate:      {overall['success_rate'] * 100:.1f}%\n\n")

        f.write(f"Avg Chunks/Query:  {overall['avg_chunks_per_query']:.1f}\n")
        f.write(f"Avg Tokens/Query:  {int(overall['avg_tokens_per_query'])}\n\n")

        f.write(f"Latency (avg):     {overall['latency_avg']:.0f}ms\n")
        f.write(f"Latency (P50):     {overall['latency_p50']:.0f}ms\n")
        f.write(f"Latency (P95):     {overall['latency_p95']:.0f}ms\n")
        f.write(f"Latency (P99):     {overall['latency_p99']:.0f}ms\n\n")

        precision_status = "✓" if overall['avg_precision'] >= 0.80 else "⚠"
        recall_status = "✓" if overall['avg_recall'] >= 0.90 else "⚠"
        f1_status = "✓" if overall['avg_f1'] >= 0.75 else "⚠"
        f.write(f"Status: Precision {precision_status} | Recall {recall_status} | F1 {f1_status}\n\n")

        f.write("-" * 60 + "\n")
        f.write("BY DIFFICULTY\n")
        f.write("-" * 60 + "\n")
        for difficulty, metrics in by_difficulty.items():
            f.write(f"{difficulty.capitalize()} ({metrics['total_queries']} queries):\n")
            f.write(f"  Precision: {metrics['avg_precision']:.3f} | Recall: {metrics['avg_recall']:.3f} | ")
            f.write(f"F1: {metrics['avg_f1']:.3f} | MRR: {metrics['mrr']:.3f}\n")
        f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("BY CATEGORY\n")
        f.write("-" * 60 + "\n")
        for category, metrics in by_category.items():
            f.write(f"{category.capitalize()} ({metrics['total_queries']} queries):\n")
            f.write(f"  Precision: {metrics['avg_precision']:.3f} | Recall: {metrics['avg_recall']:.3f} | ")
            f.write(f"F1: {metrics['avg_f1']:.3f}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"✓ Report saved: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="BGE-M3 Multi-Functional Retrieval Test Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("BGE-M3 MULTI-FUNCTIONAL RETRIEVAL EVALUATION")
    print("=" * 60 + "\n")

    config = load_config(config_path)

    dataset = load_golden_dataset(config['golden_dataset'])
    test_cases = dataset["test_cases"]

    print(f"\nEnsuring BGE-M3 vector store...")
    if not ensure_vector_store(config):
        print("✗ Failed to ensure vector store")
        return

    print(f"\nLoading BGE-M3 retriever...")
    retriever = get_bge_m3_retriever(
        vector_dir=config['vector_store_dir'],
        k=config['retrieval_k'],
        dense_weight=config['dense_weight'],
        sparse_weight=config['sparse_weight'],
        colbert_weight=config['colbert_weight']
    )

    print(f"\n{'='*60}")
    print(f"Running {len(test_cases)} test cases...")
    print(f"{'='*60}\n")

    progress = ProgressBar(len(test_cases), "Testing")
    results = []

    for i, test_case in enumerate(test_cases, 1):
        progress.update(i, f"{test_case['id']}")
        result = run_single_test(test_case, config, retriever, verbose=args.verbose)
        results.append(result)
        save_detailed_result(result, output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print("Generating reports...")
    print(f"{'='*60}\n")

    save_summary_csv(results, output_dir, timestamp)
    generate_report(results, config, output_dir, timestamp)

    print(f"\n{'='*60}")
    print("✓ BGE-M3 MULTI-FUNCTIONAL EVALUATION COMPLETE")
    print(f"{'='*60}\n")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
