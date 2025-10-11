from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional, TYPE_CHECKING
import re

if TYPE_CHECKING:
    from langchain.schema.retriever import BaseRetriever


def retrieve_context(
    query: str,
    retriever: Optional["BaseRetriever"] = None,
    vector_dir: Optional[Path] = None,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    mode: Literal["docs", "web", "all"] = "docs",
    k_docs: int = 3,
    k_web: int = 3,
    max_len: int = 2000,
    relevance_threshold: float = 0.8
) -> str:
    """Retrieve context from documents and/or web.

    Args:
        query: User query
        retriever: Pre-built retriever instance (optional)
        vector_dir: Directory with vector store (used if retriever is None)
        embedding_model: Model name for embeddings (used if retriever is None)
        mode: Retrieval mode - "docs", "web", or "all"
        k_docs: Number of documents to retrieve
        k_web: Number of web results to retrieve
        max_len: Maximum length of each context snippet
        relevance_threshold: Minimum relevance score (0-1)

    Returns:
        Retrieved context as formatted string
    """
    contexts = []

    if mode in {"docs", "all"}:
        # Build retriever on-demand if not provided
        if retriever is None:
            if vector_dir is None:
                raise ValueError("Either retriever or vector_dir must be provided for docs mode")
            from z3_core.vector import get_retriever
            retriever = get_retriever(vector_dir, embedding_model, k=k_docs)

        docs = retriever.get_relevant_documents(query)
        
        # Apply simple relevance filtering (threshold 0.8)
        if docs:
            # Filter docs based on simple content relevance
            filtered_docs = []
            query_words = set(query.lower().split())
            
            for doc in docs:
                content_words = set(doc.page_content.lower().split())
                if query_words and content_words:
                    # Simple word overlap score
                    overlap = len(query_words.intersection(content_words))
                    relevance_score = overlap / len(query_words)
                    
                    if relevance_score >= relevance_threshold:
                        filtered_docs.append(doc)
            
            # Use filtered docs if any pass threshold, otherwise use all
            final_docs = filtered_docs if filtered_docs else docs
            
            context_docs = "\n".join(
                f"[Docs] { _safe_content(d.page_content.strip(), max_len) }"
                for d in final_docs if d.page_content.strip()
            )
            contexts.append(context_docs)
            print(f"DEBUG: RAG.docs - found: {len(docs)}, filtered: {len(final_docs)}")

    if mode in {"web", "all"}:
        # Web search is optional - skip if not available
        try:
            from app.services.search import search_web
            snippets = search_web(query, k=k_web)
            if snippets:
                context_web = "\n".join(
                    f"[Web] { _safe_content(s.strip(), max_len) }"
                    for s in snippets if s.strip()
                )
                contexts.append(context_web)
            print(f"DEBUG: RAG.web - found: {len(snippets)}")
        except ImportError:
            print("INFO: Web search not available, skipping")
        except Exception as e:
            print(f"WARNING: Web search failed - error: {e}")

    if not contexts:
        print(f"WARNING: No RAG context found - query: {query}, mode: {mode}")
        return ""
    return "\n\n".join(contexts)


def rebuild_index(
    docs_dir: Path,
    vector_dir: Path,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
) -> None:
    """Rebuild vector index from documents.

    Args:
        docs_dir: Directory containing documents
        vector_dir: Directory where vector store will be saved
        embedding_model: HuggingFace model name for embeddings
    """
    from z3_core.vector import build_index
    build_index(docs_dir, vector_dir, embedding_model)

def _safe_content(text: str, max_len: int = 2_000) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"

if __name__ == "__main__":
    import sys

    print("Usage: Use domain_config.py to configure and build indexes")
    print("Example:")
    print("  from z3_core.rag import rebuild_index")
    print("  from pathlib import Path")
    print("  rebuild_index(Path('docs'), Path('data/vector_store'))")
