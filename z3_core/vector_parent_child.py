"""Parent-Child Markdown Splitter for Phase 8E.

Hybrid approach:
- Parent: Split on markdown headers (##) - full semantic sections
- Child: Further split large parents with RecursiveCharacterTextSplitter
- Store child chunks for retrieval with parent metadata for context
"""

from __future__ import annotations
from pathlib import Path
from typing import List, TYPE_CHECKING

from langchain.schema import Document

if TYPE_CHECKING:
    from langchain_community.vectorstores.faiss import FAISS


def _get_embeddings(model_name: str):
    """Get HuggingFace embeddings (reuse from vector.py)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    print(f"INFO: Using HuggingFace embeddings - model: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def _split_parent_child(
    docs: List[Document],
    parent_headers: List[str] = ["##"],
    child_chunk_size: int = 450,
    child_chunk_overlap: int = 50,
    parent_max_tokens: int = 1500
) -> List[Document]:
    """Split documents using parent-child markdown approach.

    Args:
        docs: Raw documents to split
        parent_headers: Markdown headers to split on (default: ["##"] for level 2)
        child_chunk_size: Size for child chunks if parent too large
        child_chunk_overlap: Overlap for child chunks
        parent_max_tokens: Max tokens for parent before child splitting

    Returns:
        List of child documents with parent metadata
    """
    from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

    # Prepare header mapping for MarkdownHeaderTextSplitter
    headers_to_split_on = [(h, f"Header{h.count('#')}") for h in parent_headers]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap
    )

    all_children = []

    for doc in docs:
        # Step 1: Split into parent sections using markdown headers
        parents = markdown_splitter.split_text(doc.page_content)

        # Preserve source metadata
        source = doc.metadata.get('source', 'unknown')

        for i, parent in enumerate(parents):
            parent_content = parent.page_content
            parent_metadata = parent.metadata

            # Estimate token count (rough: 1 token ≈ 4 chars)
            estimated_tokens = len(parent_content) // 4

            # Step 2: If parent too large, split into children
            if estimated_tokens > parent_max_tokens:
                # Split parent into children
                children = child_splitter.create_documents([parent_content])

                # Add parent context to each child
                for j, child in enumerate(children):
                    child.metadata = {
                        'source': source,
                        'parent_section': parent_metadata.get('Header2', f'Section_{i}'),
                        'child_index': j,
                        'parent_content': parent_content[:500] + '...',  # Store parent preview
                        **parent_metadata  # Include all parent headers
                    }
                    all_children.append(child)
            else:
                # Parent small enough, use as-is
                parent.metadata = {
                    'source': source,
                    'parent_section': parent_metadata.get('Header2', f'Section_{i}'),
                    'child_index': 0,
                    'is_complete_section': True,  # Flag for complete sections
                    **parent_metadata
                }
                all_children.append(parent)

    print(f"INFO: Parent-Child split complete - parents: {len(docs)}, children: {len(all_children)}")
    return all_children


def build_parent_child_index(
    docs_dir: Path,
    vector_dir: Path,
    embedding_model: str,
    parent_headers: List[str] = ["##"],
    child_chunk_size: int = 450,
    child_chunk_overlap: int = 50,
    parent_max_tokens: int = 1500
) -> None:
    """Build FAISS index using parent-child markdown splitter.

    Args:
        docs_dir: Directory containing markdown documents
        vector_dir: Directory to save FAISS index
        embedding_model: HuggingFace embedding model name
        parent_headers: Headers to split on for parents
        child_chunk_size: Size for child chunks
        child_chunk_overlap: Overlap for child chunks
        parent_max_tokens: Max tokens before child splitting
    """
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_community.vectorstores.faiss import FAISS

    print(f"INFO: Building parent-child index - docs_dir: {docs_dir}")

    # Load raw documents
    loaders = [
        DirectoryLoader(str(docs_dir), glob="**/*.md", loader_cls=TextLoader),
        DirectoryLoader(str(docs_dir), glob="**/*.txt", loader_cls=TextLoader),
    ]
    docs: List[Document] = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"WARNING: Doc load failed - loader: {loader}, error: {e}")

    print(f"INFO: Raw docs loaded - total: {len(docs)}")

    # Split using parent-child approach
    child_docs = _split_parent_child(
        docs,
        parent_headers=parent_headers,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        parent_max_tokens=parent_max_tokens
    )

    # Build FAISS index
    vectordb = FAISS.from_documents(child_docs, _get_embeddings(embedding_model))

    # Save index
    vector_dir.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(vector_dir))

    print(f"INFO: Parent-child index saved - path: {vector_dir}, total chunks: {len(child_docs)}")

    # Print sample metadata for verification
    if child_docs:
        print(f"INFO: Sample child metadata: {child_docs[0].metadata}")


def get_parent_child_retriever(
    vector_dir: Path,
    embedding_model: str,
    k: int = 3
):
    """Load FAISS retriever for parent-child index.

    Args:
        vector_dir: Directory containing FAISS index
        embedding_model: HuggingFace embedding model name
        k: Number of chunks to retrieve

    Returns:
        FAISS retriever
    """
    from langchain_community.vectorstores.faiss import FAISS

    try:
        vectordb = FAISS.load_local(
            str(vector_dir),
            _get_embeddings(embedding_model),
            allow_dangerous_deserialization=True
        )
        print(f"INFO: Parent-child index loaded - path: {vector_dir}")
        return vectordb.as_retriever(search_kwargs={"k": k})
    except Exception as e:
        print(f"ERROR: Failed to load parent-child index - error: {e}")
        raise FileNotFoundError(f"Parent-child index not found at {vector_dir}")
