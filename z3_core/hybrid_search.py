"""
Hybrid Search: Semantic (FAISS) + Keyword (BM25) retrieval.

Combines:
- Semantic similarity (dense embeddings via FAISS)
- Keyword matching (sparse BM25 scores)

Goal: Fix "meleset sedikit" problem by catching exact keyword matches.
"""

from pathlib import Path
from typing import List
from langchain.schema import Document


def build_bm25_index(docs_dir: Path, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Build BM25 index from documents.

    Args:
        docs_dir: Directory containing markdown/txt files
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks

    Returns:
        BM25Retriever instance
    """
    from langchain_community.retrievers import BM25Retriever
    from z3_core.vector import _load_raw_docs, _split_docs

    print(f"INFO: Building BM25 index from docs - docs_dir: {docs_dir}")

    # Load and split docs (same as FAISS)
    docs = _load_raw_docs(docs_dir)
    split_docs = _split_docs(docs, chunk_size, chunk_overlap, use_markdown_splitter=False)

    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(split_docs)

    print(f"INFO: BM25 index built - total docs: {len(split_docs)}")
    return bm25_retriever


def get_hybrid_retriever(
    vector_dir: Path,
    docs_dir: Path,
    embedding_model: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    k: int = 7,
    weights: List[float] = None
):
    """
    Create hybrid retriever combining FAISS (semantic) and BM25 (keyword).

    Args:
        vector_dir: Path to FAISS index
        docs_dir: Path to documents (for BM25)
        embedding_model: Embedding model name
        chunk_size: Chunk size for BM25
        chunk_overlap: Chunk overlap for BM25
        k: Number of results to retrieve per retriever
        weights: [semantic_weight, bm25_weight] (default: [0.5, 0.5])

    Returns:
        EnsembleRetriever combining FAISS and BM25
    """
    from langchain.retrievers import EnsembleRetriever
    from z3_core.vector import get_retriever

    if weights is None:
        weights = [0.5, 0.5]  # Equal weight by default

    print(f"INFO: Creating hybrid retriever - weights: {weights}")

    # 1. FAISS retriever (semantic)
    faiss_retriever = get_retriever(vector_dir, embedding_model, k=k)
    print(f"✓ FAISS retriever loaded (k={k})")

    # 2. BM25 retriever (keyword)
    bm25_retriever = build_bm25_index(docs_dir, chunk_size, chunk_overlap)
    bm25_retriever.k = k  # Set k for BM25
    print(f"✓ BM25 retriever built (k={k})")

    # 3. Ensemble retriever (combines both)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=weights
    )
    print(f"✓ Hybrid retriever ready - semantic {weights[0]:.1f} + BM25 {weights[1]:.1f}")

    return ensemble_retriever
