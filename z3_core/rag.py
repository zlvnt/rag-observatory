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
    relevance_threshold: float = 0.8,
    return_debug_info: bool = False,
    use_reranker: bool = False,
    reranker_model: str = "BAAI/bge-reranker-base",
    reranker_top_k: int = 3,
    reranker_use_fp16: bool = True,
    use_hybrid_search: bool = False,
    hybrid_weights: Optional[list] = None,
    docs_dir: Optional[Path] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50
):
    """Retrieve context from documents and/or web.

    Args:
        query: User query
        retriever: Pre-built retriever instance (optional)
        vector_dir: Directory with vector store (used if retriever is None)
        embedding_model: Model name for embeddings (used if retriever is None)
        mode: Retrieval mode - "docs", "web", or "all"
        k_docs: Number of documents to retrieve (before reranking if enabled)
        k_web: Number of web results to retrieve
        max_len: Maximum length of each context snippet
        relevance_threshold: Minimum relevance score (0-1)
        return_debug_info: Return tuple (context, debug_info) instead of just context
        use_reranker: Enable cross-encoder reranking (Phase 9A)
        reranker_model: HuggingFace model name for reranker
        reranker_top_k: Number of docs to return after reranking
        reranker_use_fp16: Use half precision for reranker (faster, slight accuracy loss)
        use_hybrid_search: Enable hybrid search (Semantic + BM25) (Phase 9B)
        hybrid_weights: Weights for [semantic, bm25] (default: [0.5, 0.5])
        docs_dir: Directory with documents (required for hybrid search)
        chunk_size: Chunk size for BM25 (required for hybrid search)
        chunk_overlap: Chunk overlap for BM25 (required for hybrid search)

    Returns:
        str: Retrieved context string (if return_debug_info=False)
        tuple: (context, debug_info) (if return_debug_info=True)
            debug_info contains:
            - docs_retrieved: List of doc metadata (source, score, relevance_score)
            - num_docs_initial: Number of docs before filtering/reranking
            - num_docs_final: Number of docs after filtering
            - retrieval_mode: Mode used for retrieval
    """
    contexts = []
    debug_data = {
        "docs_retrieved": [],
        "num_docs_initial": 0,
        "num_docs_final": 0,
        "retrieval_mode": mode
    }

    if mode in {"docs", "all"}:
        # Build retriever on-demand if not provided
        if retriever is None:
            if vector_dir is None:
                raise ValueError("Either retriever or vector_dir must be provided for docs mode")

            # Phase 9B: Hybrid search support
            if use_hybrid_search:
                if docs_dir is None:
                    raise ValueError("docs_dir is required for hybrid search")
                from z3_core.hybrid_search import get_hybrid_retriever
                retriever = get_hybrid_retriever(
                    vector_dir=vector_dir,
                    docs_dir=docs_dir,
                    embedding_model=embedding_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    k=k_docs,
                    weights=hybrid_weights
                )
                print(f"DEBUG: Using hybrid retriever (Semantic + BM25)")
            else:
                from z3_core.vector import get_retriever
                retriever = get_retriever(vector_dir, embedding_model, k=k_docs)

        docs = retriever.get_relevant_documents(query)

        debug_data["num_docs_initial"] = len(docs)

        # Apply reranker if enabled (Phase 9A)
        if use_reranker and docs:
            from z3_core.reranker import BGEReranker

            reranker = BGEReranker(model_name=reranker_model, use_fp16=reranker_use_fp16)
            # Get docs WITH reranker scores
            docs_with_scores = reranker.rerank(query, docs, top_k=reranker_top_k, return_scores=True)

            print(f"DEBUG: Reranker applied - initial: {debug_data['num_docs_initial']}, reranked top-{reranker_top_k}: {len(docs_with_scores)}")

            # Filter based on RERANKER SCORES (not word overlap!)
            filtered_docs = []

            for i, (doc, reranker_score) in enumerate(docs_with_scores):
                # Store doc metadata for debug
                doc_metadata = {
                    "source": doc.metadata.get("source", "unknown"),
                    "relevance_score": float(reranker_score),  # Now using reranker score!
                    "rank": i,
                    "passed_threshold": reranker_score >= relevance_threshold
                }
                debug_data["docs_retrieved"].append(doc_metadata)

                # Filter by reranker score
                if reranker_score >= relevance_threshold:
                    filtered_docs.append(doc)

            # Smart fallback: if none pass threshold, use top chunks based on score
            if filtered_docs:
                final_docs = filtered_docs
            else:
                # Adaptive fallback based on top score quality
                top_score = docs_with_scores[0][1] if docs_with_scores else 0

                if top_score >= 0.3:  # Decent score
                    # Use top 2 chunks
                    final_docs = [doc for doc, _ in docs_with_scores[:2]]
                    print(f"DEBUG: Fallback - top score {top_score:.3f}, using top 2 chunks")
                elif top_score >= 0.2:  # Low score
                    # Use only top 1 chunk
                    final_docs = [docs_with_scores[0][0]]
                    print(f"DEBUG: Fallback - low score {top_score:.3f}, using top 1 chunk")
                else:
                    # Very low scores, return empty or top 1
                    final_docs = [docs_with_scores[0][0]] if docs_with_scores else []
                    print(f"DEBUG: Fallback - very low score {top_score:.3f}, using top 1 chunk")

            debug_data["num_docs_final"] = len(final_docs)

            # Store individual chunk texts for qualitative analysis
            debug_data["retrieved_chunks_full"] = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "text": doc.page_content.strip()
                }
                for doc in final_docs if doc.page_content.strip()
            ]

            context_docs = "\n".join(
                f"[Docs] { _safe_content(d.page_content.strip(), max_len) }"
                for d in final_docs if d.page_content.strip()
            )
            contexts.append(context_docs)
            print(f"DEBUG: RAG.docs - found: {len(docs_with_scores)}, filtered: {len(final_docs)}")

        # If reranker NOT enabled, use old word overlap method (backward compatibility)
        elif docs:
            # Filter docs based on simple content relevance (word overlap)
            filtered_docs = []
            query_words = set(query.lower().split())

            for i, doc in enumerate(docs):
                content_words = set(doc.page_content.lower().split())
                if query_words and content_words:
                    # Simple word overlap score
                    overlap = len(query_words.intersection(content_words))
                    relevance_score = overlap / len(query_words)

                    # Store doc metadata for debug
                    doc_metadata = {
                        "source": doc.metadata.get("source", "unknown"),
                        "relevance_score": relevance_score,
                        "rank": i,
                        "passed_threshold": relevance_score >= relevance_threshold
                    }
                    debug_data["docs_retrieved"].append(doc_metadata)

                    if relevance_score >= relevance_threshold:
                        filtered_docs.append(doc)

            # Use filtered docs if any pass threshold, otherwise use all
            final_docs = filtered_docs if filtered_docs else docs
            debug_data["num_docs_final"] = len(final_docs)

            # Store individual chunk texts for qualitative analysis
            debug_data["retrieved_chunks_full"] = [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "text": doc.page_content.strip()
                }
                for doc in final_docs if doc.page_content.strip()
            ]

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
        result = ""
    else:
        result = "\n\n".join(contexts)

    if return_debug_info:
        return result, debug_data

    return result


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
