#!/usr/bin/env python3
"""
Create qualitative_analysis_exp6.csv from Exp6 detailed results
Extracts retrieved text for manual inspection
"""

import json
import csv
from pathlib import Path
from typing import List, Dict

def load_exp6_results() -> List[Dict]:
    """Load all Exp6 detailed JSON results"""
    exp6_dir = Path("results/exp6/detailed")
    results = []

    # Get all JSON files sorted by name
    json_files = sorted(exp6_dir.glob("*.json"))

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results.append(data)

    return results

def extract_chunks_preview(context_preview: str, num_chunks: int) -> List[str]:
    """
    Extract individual chunks from the combined preview
    Since preview shows combined text, we'll split by [Docs] marker
    """
    chunks = []

    # Split by [Docs] marker
    parts = context_preview.split("[Docs]")

    # Remove empty strings and clean up
    parts = [p.strip() for p in parts if p.strip()]

    # Take first 200 chars of each chunk for preview
    for i, part in enumerate(parts[:num_chunks]):  # Limit to num_chunks
        # Truncate to ~200 chars, break at word boundary
        if len(part) > 200:
            truncated = part[:200].rsplit(' ', 1)[0] + "..."
        else:
            truncated = part
        chunks.append(truncated)

    # Pad with empty strings if fewer chunks
    while len(chunks) < 3:
        chunks.append("")

    return chunks[:3]  # Always return exactly 3 chunks (or empty strings)

def create_qualitative_csv(results: List[Dict], output_path: str):
    """Create CSV with qualitative analysis columns"""

    csv_columns = [
        'query_id',
        'difficulty',
        'category',
        'query_text',
        'expected_docs',
        'retrieved_docs',
        'num_chunks',
        'precision',
        'recall',
        'f1_score',
        'mrr',
        'chunk_1_preview',
        'chunk_2_preview',
        'chunk_3_preview',
        'context_tokens',
        'inspection_notes'
    ]

    rows = []

    for result in results:
        # Extract chunk previews
        chunks = extract_chunks_preview(
            result['retrieval_trace']['retrieved_context_preview'],
            result['retrieval_trace']['num_chunks_retrieved']
        )

        row = {
            'query_id': result['test_id'],
            'difficulty': result['difficulty'],
            'category': result['category'],
            'query_text': result['query'],
            'expected_docs': ' | '.join(result['expected_docs']),
            'retrieved_docs': ' | '.join(result['retrieval_trace']['unique_docs']),
            'num_chunks': result['retrieval_trace']['num_chunks_retrieved'],
            'precision': f"{result['evaluation']['precision']:.3f}",
            'recall': f"{result['evaluation']['recall']:.3f}",
            'f1_score': f"{result['evaluation']['f1_score']:.3f}",
            'mrr': f"{result['evaluation']['reciprocal_rank']:.3f}",
            'chunk_1_preview': chunks[0],
            'chunk_2_preview': chunks[1],
            'chunk_3_preview': chunks[2],
            'context_tokens': result['retrieval_trace']['context_tokens_approx'],
            'inspection_notes': ''  # Empty for manual filling
        }

        rows.append(row)

    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Created {output_path}")
    print(f"   - Total queries: {len(rows)}")
    print(f"   - Columns: {len(csv_columns)}")
    print("\nColumn breakdown:")
    print("  - Query metadata: query_id, difficulty, category, query_text")
    print("  - Ground truth: expected_docs")
    print("  - Retrieval results: retrieved_docs, num_chunks")
    print("  - Metrics: precision, recall, f1_score, mrr")
    print("  - Text inspection: chunk_1/2/3_preview (first ~200 chars each)")
    print("  - Efficiency: context_tokens")
    print("  - Manual analysis: inspection_notes (for your observations)")

if __name__ == "__main__":
    # Load Exp6 results
    results = load_exp6_results()

    # Create qualitative CSV
    output_path = "results/report/qualitative_analysis_exp6.csv"
    create_qualitative_csv(results, output_path)

    print(f"\n📊 Next steps:")
    print(f"  1. Open {output_path}")
    print(f"  2. Read through chunk_1/2/3_preview columns")
    print(f"  3. Add notes in 'inspection_notes' for:")
    print(f"     - False positives (irrelevant chunks retrieved)")
    print(f"     - Ranking issues (relevant doc ranked too low)")
    print(f"     - Context problems (chunks cut mid-sentence, missing key info)")
    print(f"     - Multi-doc failures (expected 2+ docs, only got 1)")
