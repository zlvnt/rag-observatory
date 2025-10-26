"""
BGE Reranker for RAG Observatory.

Cross-encoder reranking to improve retrieval accuracy.
Uses BAAI/bge-reranker models for multilingual support.
"""

from typing import List, Tuple
from langchain.schema import Document


class BGEReranker:
    """Cross-encoder reranker using BAAI BGE models."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", use_fp16: bool = True):
        """
        Initialize BGE reranker.

        Args:
            model_name: HuggingFace model name (default: BAAI/bge-reranker-base)
            use_fp16: Use half precision for faster inference (default: True)
        """
        from FlagEmbedding import FlagReranker

        self.model_name = model_name
        self.use_fp16 = use_fp16

        print(f"Loading reranker model: {model_name} (fp16={use_fp16})...")
        self.reranker = FlagReranker(model_name, use_fp16=use_fp16)
        print(f"✓ Reranker loaded: {model_name}")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> List[Document]:
        """
        Rerank documents using cross-encoder model.

        Args:
            query: User query
            documents: List of retrieved documents (from bi-encoder)
            top_k: Number of top documents to return after reranking

        Returns:
            Top-k reranked documents (sorted by relevance score)
        """
        if not documents:
            return []

        # Prepare (query, doc) pairs for reranker
        pairs = [[query, doc.page_content] for doc in documents]

        # Get relevance scores from cross-encoder
        scores = self.reranker.compute_score(pairs)

        # Handle single document case (score is float, not list)
        if not isinstance(scores, list):
            scores = [scores]

        # Sort documents by score (descending)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Add reranker scores to document metadata
        for doc, score in doc_score_pairs:
            doc.metadata['reranker_score'] = float(score)

        # Return top-k documents
        top_docs = [doc for doc, score in doc_score_pairs[:top_k]]

        return top_docs

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents and return with scores.

        Args:
            query: User query
            documents: List of retrieved documents
            top_k: Number of top documents to return

        Returns:
            List of (document, score) tuples, sorted by score descending
        """
        if not documents:
            return []

        # Prepare pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Get scores
        scores = self.reranker.compute_score(pairs)

        # Handle single document
        if not isinstance(scores, list):
            scores = [scores]

        # Sort by score
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Add scores to metadata
        for doc, score in doc_score_pairs:
            doc.metadata['reranker_score'] = float(score)

        return doc_score_pairs[:top_k]
