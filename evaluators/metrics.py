"""Metrics calculation for RAG evaluation.

This module provides functions to calculate standard RAG metrics:
- Precision@K, Recall@K
- Mean Reciprocal Rank (MRR)
- Keyword Coverage
- Success Rate
"""

from typing import List, Set, Dict, Any


def calculate_precision_at_k(
    retrieved_docs: List[str],
    expected_docs: List[str],
    k: int = 3
) -> float:
    """Calculate Precision@K.

    Precision@K = (relevant docs retrieved) / (total docs retrieved)

    Args:
        retrieved_docs: List of retrieved document names
        expected_docs: List of expected relevant document names
        k: Number of top documents to consider

    Returns:
        Precision score (0.0 to 1.0)
    """
    if not retrieved_docs or k == 0:
        return 0.0

    # Take top-k retrieved docs
    top_k = retrieved_docs[:k]

    # Convert to sets for intersection
    retrieved_set = set(top_k)
    expected_set = set(expected_docs)

    # Count relevant docs in retrieved
    relevant_retrieved = len(retrieved_set.intersection(expected_set))

    return relevant_retrieved / len(top_k)


def calculate_recall_at_k(
    retrieved_docs: List[str],
    expected_docs: List[str],
    k: int = 3
) -> float:
    """Calculate Recall@K.

    Recall@K = (relevant docs retrieved) / (total relevant docs)

    Args:
        retrieved_docs: List of retrieved document names
        expected_docs: List of expected relevant document names
        k: Number of top documents to consider

    Returns:
        Recall score (0.0 to 1.0)
    """
    if not expected_docs:
        return 1.0  # No relevant docs to retrieve

    if not retrieved_docs:
        return 0.0

    # Take top-k retrieved docs
    top_k = retrieved_docs[:k]

    # Convert to sets for intersection
    retrieved_set = set(top_k)
    expected_set = set(expected_docs)

    # Count relevant docs in retrieved
    relevant_retrieved = len(retrieved_set.intersection(expected_set))

    return relevant_retrieved / len(expected_set)


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score (harmonic mean of precision and recall).

    Args:
        precision: Precision score
        recall: Recall score

    Returns:
        F1 score (0.0 to 1.0)
    """
    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def calculate_reciprocal_rank(
    retrieved_docs: List[str],
    expected_docs: List[str]
) -> float:
    """Calculate Reciprocal Rank for a single query.

    RR = 1 / (rank of first relevant document)

    Args:
        retrieved_docs: List of retrieved document names (in order)
        expected_docs: List of expected relevant document names

    Returns:
        Reciprocal rank (0.0 to 1.0)
    """
    if not retrieved_docs or not expected_docs:
        return 0.0

    expected_set = set(expected_docs)

    # Find rank of first relevant doc (1-indexed)
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in expected_set:
            return 1.0 / rank

    return 0.0  # No relevant doc found


def calculate_mrr(reciprocal_ranks: List[float]) -> float:
    """Calculate Mean Reciprocal Rank across multiple queries.

    Args:
        reciprocal_ranks: List of RR scores for each query

    Returns:
        Mean reciprocal rank (0.0 to 1.0)
    """
    if not reciprocal_ranks:
        return 0.0

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def calculate_keyword_coverage(
    generated_answer: str,
    expected_keywords: List[str]
) -> Dict[str, Any]:
    """Calculate keyword coverage in generated answer.

    Args:
        generated_answer: Generated answer text
        expected_keywords: List of expected keywords/phrases

    Returns:
        Dictionary with:
        - coverage: Ratio of found keywords (0.0 to 1.0)
        - found: List of keywords found
        - missing: List of keywords missing
    """
    if not expected_keywords:
        return {"coverage": 1.0, "found": [], "missing": []}

    answer_lower = generated_answer.lower()

    found = []
    missing = []

    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            found.append(keyword)
        else:
            missing.append(keyword)

    coverage = len(found) / len(expected_keywords) if expected_keywords else 1.0

    return {
        "coverage": coverage,
        "found": found,
        "missing": missing
    }


def calculate_success_rate(
    routing_correct: List[bool],
    has_retrieved_docs: List[bool]
) -> float:
    """Calculate overall success rate.

    A query is successful if routing is correct AND at least 1 doc retrieved.

    Args:
        routing_correct: List of boolean values for routing correctness
        has_retrieved_docs: List of boolean values for doc retrieval success

    Returns:
        Success rate (0.0 to 1.0)
    """
    if not routing_correct or len(routing_correct) != len(has_retrieved_docs):
        return 0.0

    successes = sum(
        1 for route_ok, has_docs in zip(routing_correct, has_retrieved_docs)
        if route_ok and has_docs
    )

    return successes / len(routing_correct)


def calculate_percentile(values: List[float], percentile: int) -> float:
    """Calculate percentile of a list of values.

    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0-100)

    Returns:
        Percentile value
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile / 100)
    index = min(index, len(sorted_values) - 1)

    return sorted_values[index]


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics from multiple test results.

    Args:
        results: List of individual test result dictionaries

    Returns:
        Aggregated metrics dictionary
    """
    if not results:
        return {}

    # Extract individual metrics
    routing_correct = [r["evaluation"]["routing_correct"] for r in results]
    precisions = [r["evaluation"]["precision"] for r in results]
    recalls = [r["evaluation"]["recall"] for r in results]
    keyword_coverages = [r["evaluation"]["keyword_coverage"] for r in results]
    latencies = [r["pipeline_trace"]["total_latency_ms"] for r in results]
    rrs = [r["evaluation"].get("reciprocal_rank", 0.0) for r in results]

    # Check if docs were retrieved
    has_docs = [
        r["pipeline_trace"]["retrieval"]["num_docs_final"] > 0
        for r in results
    ]

    # Calculate aggregates
    return {
        "total_queries": len(results),
        "routing_accuracy": sum(routing_correct) / len(routing_correct),
        "routing_correct_count": sum(routing_correct),
        "avg_precision": sum(precisions) / len(precisions),
        "avg_recall": sum(recalls) / len(recalls),
        "avg_f1": calculate_f1_score(
            sum(precisions) / len(precisions),
            sum(recalls) / len(recalls)
        ),
        "avg_keyword_coverage": sum(keyword_coverages) / len(keyword_coverages),
        "mrr": calculate_mrr(rrs),
        "success_rate": calculate_success_rate(routing_correct, has_docs),
        "latency_avg": sum(latencies) / len(latencies),
        "latency_p50": calculate_percentile(latencies, 50),
        "latency_p95": calculate_percentile(latencies, 95),
        "latency_p99": calculate_percentile(latencies, 99),
    }


def group_by_category(results: List[Dict[str, Any]], field: str) -> Dict[str, List[Dict[str, Any]]]:
    """Group results by a specific field (e.g., difficulty, category).

    Args:
        results: List of test results
        field: Field name to group by

    Returns:
        Dictionary mapping field values to lists of results
    """
    grouped = {}

    for result in results:
        value = result.get(field, "unknown")
        if value not in grouped:
            grouped[value] = []
        grouped[value].append(result)

    return grouped
