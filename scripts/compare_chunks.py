"""Compare text chunks between two configs for specific queries."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from z3_core.domain_config import load_domain_config
from z3_core.vector import get_retriever

def compare_query(query: str, config1_name: str, config2_name: str):
    """Compare retrieval for a single query across two configs."""

    # Load config 1 (chunk 500)
    config1 = load_domain_config(config1_name, config_dir=Path('configs/experiments_phase8b'))
    retriever1 = get_retriever(
        config1.vector_store_dir,
        config1.embedding_model,
        k=config1.retrieval_k
    )

    # Load config 2 (chunk 700)
    config2 = load_domain_config(config2_name, config_dir=Path('configs'))
    retriever2 = get_retriever(
        config2.vector_store_dir,
        config2.embedding_model,
        k=config2.retrieval_k
    )

    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}\n")

    # Retrieve from config 1
    print(f"--- CONFIG 1: {config1_name} (chunk {config1.chunk_size}, overlap {config1.chunk_overlap}) ---\n")
    docs1 = retriever1.invoke(query)
    for i, doc in enumerate(docs1[:3]):
        source = doc.metadata.get('source', 'unknown')
        print(f"Chunk {i+1} ({source}):")
        print(doc.page_content[:350])
        print("...\n")

    # Retrieve from config 2
    print(f"\n--- CONFIG 2: {config2_name} (chunk {config2.chunk_size}, overlap {config2.chunk_overlap}) ---\n")
    docs2 = retriever2.invoke(query)
    for i, doc in enumerate(docs2[:3]):
        source = doc.metadata.get('source', 'unknown')
        print(f"Chunk {i+1} ({source}):")
        print(doc.page_content[:350])
        print("...\n")

if __name__ == "__main__":
    # Test queries
    queries = [
        "Bagaimana cara return barang yang rusak?",
        "Kenapa OTP tidak masuk ke HP saya?",
        "Sudah bayar tapi status masih menunggu pembayaran, kenapa?",
    ]

    for query in queries:
        compare_query(query, "z3_agent_exp6_bge", "z3_agent_exp6_bge")
