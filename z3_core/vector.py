from __future__ import annotations
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

from langchain.schema import Document

if TYPE_CHECKING:
    from langchain_community.vectorstores.faiss import FAISS

def _index_exists(vector_dir: Path) -> bool:
    """Check if FAISS index exists in the given directory.

    Args:
        vector_dir: Directory where vector store should be located

    Returns:
        True if index exists, False otherwise
    """
    return (vector_dir / "index.faiss").exists()


def _get_embeddings(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """Get embedding model instance.

    Args:
        model_name: HuggingFace model name to use for embeddings

    Returns:
        Embedding model instance

    Note:
        Falls back to Gemini embeddings if HuggingFace fails
    """
    # Try HuggingFace embeddings first (best for customer service)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        print(f"INFO: Using HuggingFace embeddings - model: {model_name}")
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
            encode_kwargs={'normalize_embeddings': True}  # Better similarity scores
        )
    except Exception as e:
        print(f"WARNING: HuggingFace embeddings failed, falling back to Gemini - error: {e}")

    # Fallback to Gemini embeddings
    import os
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    print(f"INFO: Using Gemini embeddings")
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )

def _load_vectordb(vector_dir: Path, embedding_model: str) -> "FAISS":
    """Load FAISS vector store from disk.

    Args:
        vector_dir: Directory where vector store is saved
        embedding_model: Model name used for embeddings

    Returns:
        Loaded FAISS vector store
    """
    from langchain_community.vectorstores.faiss import FAISS
    vectordb = FAISS.load_local(
        str(vector_dir),
        _get_embeddings(embedding_model),
        allow_dangerous_deserialization=True  # Safe because we created the files
    )
    print(f"INFO: FAISS index loaded - path: {vector_dir}")
    return vectordb


def get_retriever(
    vector_dir: Path,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    k: int = 4
):
    """Get retriever from existing vector store or build new one.

    Args:
        vector_dir: Directory where vector store is/will be saved
        embedding_model: HuggingFace model name for embeddings
        k: Number of documents to retrieve

    Returns:
        Langchain retriever instance
    """
    try:
        vectordb = _load_vectordb(vector_dir, embedding_model)
    except Exception as e:
        if _index_exists(vector_dir):
            print(f"ERROR: Failed to load FAISS index - error: {e}")
            raise
        print(f"WARNING: Vector index not found at {vector_dir}, please build index first")
        raise FileNotFoundError(f"Vector index not found at {vector_dir}")
    return vectordb.as_retriever(search_kwargs={"k": k})

def build_index(
    docs_dir: Path,
    vector_dir: Path,
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> None:
    """Build FAISS vector index from documents.

    Args:
        docs_dir: Directory containing documents to index
        vector_dir: Directory where vector store will be saved
        embedding_model: HuggingFace model name for embeddings
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks

    Returns:
        None
    """
    print(f"INFO: Building vector index from docs - docs_dir: {docs_dir}")
    docs = _load_raw_docs(docs_dir)
    split_docs = _split_docs(docs, chunk_size, chunk_overlap)
    from langchain_community.vectorstores.faiss import FAISS

    vectordb = FAISS.from_documents(split_docs, _get_embeddings(embedding_model))

    vector_dir.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(vector_dir))
    print(f"INFO: Vector index saved - path: {vector_dir}, total: {len(split_docs)}")


def _load_raw_docs(docs_dir: Path) -> List[Document]:
    """Load raw documents from directory.

    Args:
        docs_dir: Directory containing documents

    Returns:
        List of loaded documents
    """
    from langchain_community.document_loaders import (
        DirectoryLoader,
        TextLoader,
    )

    # Use simple TextLoader for all file types (more reliable)
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
    print(f"INFO: Docs loaded - total: {len(docs)}")
    return docs

def _split_docs(docs: List[Document], chunk_size: int = 700, chunk_overlap: int = 100) -> List[Document]:
    """Split documents into chunks.

    Args:
        docs: Documents to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of split document chunks
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)
    print(f"INFO: Using RecursiveCharacterTextSplitter - chunks: {len(split_docs)}")
    return split_docs
