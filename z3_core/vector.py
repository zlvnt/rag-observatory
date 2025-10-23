from __future__ import annotations
from pathlib import Path
from typing import List, TYPE_CHECKING

from langchain.schema import Document

if TYPE_CHECKING:
    from langchain_community.vectorstores.faiss import FAISS

def _index_exists(vector_dir: Path) -> bool:
    return (vector_dir / "index.faiss").exists()


def _get_embeddings(model_name: str):
    from langchain_huggingface import HuggingFaceEmbeddings
    print(f"INFO: Using HuggingFace embeddings - model: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def _load_vectordb(vector_dir: Path, embedding_model: str) -> "FAISS":
    from langchain_community.vectorstores.faiss import FAISS
    vectordb = FAISS.load_local(
        str(vector_dir),
        _get_embeddings(embedding_model),
        allow_dangerous_deserialization=True
    )
    print(f"INFO: FAISS index loaded - path: {vector_dir}")
    return vectordb


def get_retriever(
    vector_dir: Path,
    embedding_model: str,
    k: int = 4
):
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
    embedding_model: str,
    chunk_size: int = 700,
    chunk_overlap: int = 100
) -> None:
    print(f"INFO: Building vector index from docs - docs_dir: {docs_dir}")
    docs = _load_raw_docs(docs_dir)
    split_docs = _split_docs(docs, chunk_size, chunk_overlap)
    from langchain_community.vectorstores.faiss import FAISS

    vectordb = FAISS.from_documents(split_docs, _get_embeddings(embedding_model))

    vector_dir.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(vector_dir))
    print(f"INFO: Vector index saved - path: {vector_dir}, total: {len(split_docs)}")


def _load_raw_docs(docs_dir: Path) -> List[Document]:
    from langchain_community.document_loaders import (
        DirectoryLoader,
        TextLoader,
    )

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
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(docs)
    print(f"INFO: Using RecursiveCharacterTextSplitter - chunks: {len(split_docs)}")
    return split_docs
