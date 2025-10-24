from __future__ import annotations
from pathlib import Path
from typing import List

from langchain.schema import Document

def build_bge_m3_index(
    docs_dir: Path,
    vector_dir: Path,
    chunk_size: int = 700,
    chunk_overlap: int = 100,
    dense_weight: float = 0.4,
    sparse_weight: float = 0.3,
    colbert_weight: float = 0.3
) -> None:
    print(f"INFO: Building BGE-M3 multi-functional index from docs - docs_dir: {docs_dir}")

    docs = _load_raw_docs(docs_dir)
    split_docs = _split_docs(docs, chunk_size, chunk_overlap)

    from z3_core.bge_m3_retriever import BGEM3Retriever

    retriever = BGEM3Retriever(
        model_name="BAAI/bge-m3",
        vector_store_dir=vector_dir,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        colbert_weight=colbert_weight,
        use_fp16=True
    )

    retriever.build_index(split_docs)

    print(f"INFO: BGE-M3 multi-functional index built - total chunks: {len(split_docs)}")


def get_bge_m3_retriever(
    vector_dir: Path,
    k: int = 4,
    dense_weight: float = 0.4,
    sparse_weight: float = 0.3,
    colbert_weight: float = 0.3
):
    from z3_core.bge_m3_retriever import BGEM3Retriever

    if not vector_dir.exists():
        raise FileNotFoundError(f"BGE-M3 vector store not found at {vector_dir}")

    retriever = BGEM3Retriever(
        model_name="BAAI/bge-m3",
        vector_store_dir=vector_dir,
        k=k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        colbert_weight=colbert_weight,
        use_fp16=True
    )

    print(f"INFO: BGE-M3 multi-functional retriever loaded - k={k}, weights=[{dense_weight}, {sparse_weight}, {colbert_weight}]")

    return retriever


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
