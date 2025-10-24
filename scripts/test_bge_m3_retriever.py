#!/usr/bin/env python3
"""
Test BGE-M3 multi-functional retriever implementation
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from z3_core.vector_bge_m3 import build_bge_m3_index, get_bge_m3_retriever

def test_build_index():
    print("=" * 60)
    print("TEST 1: Building BGE-M3 Multi-Functional Index")
    print("=" * 60)

    docs_dir = Path("docs")
    vector_dir = Path("data/vector_stores/test_bge_m3")

    if vector_dir.exists():
        import shutil
        shutil.rmtree(vector_dir)
        print(f"Cleaned up old index at {vector_dir}")

    build_bge_m3_index(
        docs_dir=docs_dir,
        vector_dir=vector_dir,
        chunk_size=500,
        chunk_overlap=50,
        dense_weight=0.4,
        sparse_weight=0.3,
        colbert_weight=0.3
    )

    print("\n✅ Index building complete!\n")
    return vector_dir


def test_retrieval(vector_dir: Path):
    print("=" * 60)
    print("TEST 2: Testing BGE-M3 Multi-Functional Retrieval")
    print("=" * 60)

    retriever = get_bge_m3_retriever(
        vector_dir=vector_dir,
        k=3,
        dense_weight=0.4,
        sparse_weight=0.3,
        colbert_weight=0.3
    )

    test_queries = [
        "Bagaimana cara return barang yang rusak?",
        "Berapa lama batas waktu return untuk produk elektronik?",
        "Nomor customer service TokoPedia berapa?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)

        results = retriever.retrieve(query, k=3)

        print(f"\nRetrieved {len(results)} documents:")
        for j, doc in enumerate(results, 1):
            source = doc.metadata.get('source', 'unknown')
            preview = doc.page_content[:150].replace('\n', ' ')
            print(f"\n{j}. Source: {source}")
            print(f"   Preview: {preview}...")

    print("\n✅ Retrieval test complete!\n")


if __name__ == "__main__":
    print("Testing BGE-M3 Multi-Functional Retriever Implementation\n")

    vector_dir = test_build_index()

    test_retrieval(vector_dir)

    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✅")
    print("="*60)
    print("\nNext steps:")
    print("1. Run full experiment with: python runners/test_runner.py --domain experiments_phase8b/z3_agent_exp6_bge_full")
    print("2. Compare results with Exp6 (MPNet) and Exp6_bge (BGE-M3 dense-only)")
