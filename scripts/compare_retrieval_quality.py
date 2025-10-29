"""
Compare retrieval quality between two experiments side-by-side.

Usage:
    python scripts/compare_retrieval_quality.py \
        --exp1 results/phase9a/exp9a1/detailed/ \
        --exp2 results/phase9a/exp9a2/detailed/ \
        --output results/phase9a/comparison_exp9a1_vs_exp9a2.csv
"""

import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List


def load_detailed_result(file_path: Path) -> Dict:
    """Load detailed result JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_retrieved_text(result: Dict) -> str:
    """Extract retrieved document text from result."""
    retrieval_trace = result.get('retrieval_trace', {})

    # Get the full retrieved context (this contains all chunks combined)
    context = retrieval_trace.get('retrieved_context_preview', '')

    if not context:
        return "[No context retrieved]"

    # The context already has [Docs] markers, return as-is
    return context


def create_comparison_csv(exp1_dir: Path, exp2_dir: Path, output_file: Path, exp1_name: str, exp2_name: str):
    """Create side-by-side comparison CSV."""

    # Get all JSON files from exp1
    exp1_files = sorted(exp1_dir.glob("*.json"))

    if not exp1_files:
        print(f"No JSON files found in {exp1_dir}")
        return

    print(f"Found {len(exp1_files)} test cases to compare")

    # Open CSV for writing
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'test_id',
            'query',
            'expected_docs',
            f'{exp1_name}_precision',
            f'{exp1_name}_recall',
            f'{exp1_name}_retrieved_docs',
            f'{exp1_name}_retrieved_text',
            f'{exp2_name}_precision',
            f'{exp2_name}_recall',
            f'{exp2_name}_retrieved_docs',
            f'{exp2_name}_retrieved_text',
            'precision_diff',
            'recall_diff'
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for exp1_file in exp1_files:
            test_id = exp1_file.stem

            # Load exp1 result
            exp1_result = load_detailed_result(exp1_file)

            # Load exp2 result (matching test_id)
            exp2_file = exp2_dir / f"{test_id}.json"
            if not exp2_file.exists():
                print(f"Warning: {test_id} not found in exp2, skipping")
                continue

            exp2_result = load_detailed_result(exp2_file)

            # Extract data
            query = exp1_result.get('query', '')
            expected_docs = "; ".join(exp1_result.get('expected_docs', []))

            # Exp1 metrics
            exp1_eval = exp1_result.get('evaluation', {})
            exp1_precision = exp1_eval.get('precision', 0)
            exp1_recall = exp1_eval.get('recall', 0)
            exp1_retrieved = "; ".join(exp1_result.get('retrieval_trace', {}).get('unique_docs', []))
            exp1_text = extract_retrieved_text(exp1_result)

            # Exp2 metrics
            exp2_eval = exp2_result.get('evaluation', {})
            exp2_precision = exp2_eval.get('precision', 0)
            exp2_recall = exp2_eval.get('recall', 0)
            exp2_retrieved = "; ".join(exp2_result.get('retrieval_trace', {}).get('unique_docs', []))
            exp2_text = extract_retrieved_text(exp2_result)

            # Calculate differences
            precision_diff = exp2_precision - exp1_precision
            recall_diff = exp2_recall - exp1_recall

            # Write row
            writer.writerow({
                'test_id': test_id,
                'query': query,
                'expected_docs': expected_docs,
                f'{exp1_name}_precision': f"{exp1_precision:.3f}",
                f'{exp1_name}_recall': f"{exp1_recall:.3f}",
                f'{exp1_name}_retrieved_docs': exp1_retrieved,
                f'{exp1_name}_retrieved_text': exp1_text,
                f'{exp2_name}_precision': f"{exp2_precision:.3f}",
                f'{exp2_name}_recall': f"{exp2_recall:.3f}",
                f'{exp2_name}_retrieved_docs': exp2_retrieved,
                f'{exp2_name}_retrieved_text': exp2_text,
                'precision_diff': f"{precision_diff:+.3f}",
                'recall_diff': f"{recall_diff:+.3f}"
            })

    print(f"\n✓ Comparison CSV created: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare retrieval quality between two experiments")
    parser.add_argument("--exp1", required=True, help="Path to exp1 detailed/ directory")
    parser.add_argument("--exp2", required=True, help="Path to exp2 detailed/ directory")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--exp1-name", default="Exp1", help="Name for exp1 (default: Exp1)")
    parser.add_argument("--exp2-name", default="Exp2", help="Name for exp2 (default: Exp2)")

    args = parser.parse_args()

    exp1_dir = Path(args.exp1)
    exp2_dir = Path(args.exp2)
    output_file = Path(args.output)

    # Validate paths
    if not exp1_dir.exists():
        print(f"Error: Exp1 directory not found: {exp1_dir}")
        return 1

    if not exp2_dir.exists():
        print(f"Error: Exp2 directory not found: {exp2_dir}")
        return 1

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create comparison
    create_comparison_csv(exp1_dir, exp2_dir, output_file, args.exp1_name, args.exp2_name)

    return 0


if __name__ == "__main__":
    exit(main())
