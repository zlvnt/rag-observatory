from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, TYPE_CHECKING
import numpy as np
import pickle

if TYPE_CHECKING:
    from langchain.schema import Document

class BGEM3Retriever:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        vector_store_dir: Path = None,
        k: int = 4,
        dense_weight: float = 0.4,
        sparse_weight: float = 0.3,
        colbert_weight: float = 0.3,
        use_fp16: bool = True
    ):
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self.vector_store_dir = vector_store_dir
        self.k = k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.colbert_weight = colbert_weight

        self.dense_embeddings = None
        self.sparse_embeddings = None
        self.colbert_embeddings = None
        self.documents = None
        self.metadata = None

        if vector_store_dir and vector_store_dir.exists():
            self._load_embeddings()

    def _load_embeddings(self):
        print(f"INFO: Loading BGE-M3 multi-functional embeddings from {self.vector_store_dir}")

        dense_path = self.vector_store_dir / "bge_m3_dense.npy"
        sparse_path = self.vector_store_dir / "bge_m3_sparse.pkl"
        colbert_path = self.vector_store_dir / "bge_m3_colbert.pkl"
        docs_path = self.vector_store_dir / "bge_m3_documents.pkl"

        if dense_path.exists():
            self.dense_embeddings = np.load(dense_path)
            print(f"  - Dense embeddings loaded: {self.dense_embeddings.shape}")

        if sparse_path.exists():
            with open(sparse_path, 'rb') as f:
                self.sparse_embeddings = pickle.load(f)
            print(f"  - Sparse embeddings loaded: {len(self.sparse_embeddings)} chunks")

        if colbert_path.exists():
            with open(colbert_path, 'rb') as f:
                self.colbert_embeddings = pickle.load(f)
            print(f"  - ColBERT embeddings loaded: {len(self.colbert_embeddings)} chunks")

        if docs_path.exists():
            with open(docs_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
            print(f"  - Documents loaded: {len(self.documents)} chunks")

    def build_index(self, documents: List[Document]):
        print(f"INFO: Building BGE-M3 multi-functional index for {len(documents)} documents")

        texts = [doc.page_content for doc in documents]

        print("  - Encoding with BGE-M3 (dense + sparse + colbert)...")
        embeddings = self.model.encode(
            texts,
            batch_size=12,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )

        self.dense_embeddings = embeddings['dense_vecs']
        self.sparse_embeddings = embeddings['lexical_weights']
        self.colbert_embeddings = embeddings['colbert_vecs']
        self.documents = documents
        self.metadata = [doc.metadata for doc in documents]

        print(f"  - Dense embeddings: {self.dense_embeddings.shape}")
        print(f"  - Sparse embeddings: {len(self.sparse_embeddings)} chunks")
        print(f"  - ColBERT embeddings: {len(self.colbert_embeddings)} chunks")

        self.vector_store_dir.mkdir(parents=True, exist_ok=True)

        np.save(self.vector_store_dir / "bge_m3_dense.npy", self.dense_embeddings)

        with open(self.vector_store_dir / "bge_m3_sparse.pkl", 'wb') as f:
            pickle.dump(self.sparse_embeddings, f)

        with open(self.vector_store_dir / "bge_m3_colbert.pkl", 'wb') as f:
            pickle.dump(self.colbert_embeddings, f)

        with open(self.vector_store_dir / "bge_m3_documents.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)

        print(f"INFO: BGE-M3 index saved to {self.vector_store_dir}")

    def _compute_dense_scores(self, query_embedding: np.ndarray) -> np.ndarray:
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = self.dense_embeddings / np.linalg.norm(self.dense_embeddings, axis=1, keepdims=True)
        scores = np.dot(doc_norms, query_norm)
        return scores

    def _compute_sparse_scores(self, query_sparse: Dict[str, float]) -> np.ndarray:
        scores = np.zeros(len(self.sparse_embeddings))

        for idx, doc_sparse in enumerate(self.sparse_embeddings):
            score = 0.0
            for token, query_weight in query_sparse.items():
                if token in doc_sparse:
                    score += query_weight * doc_sparse[token]
            scores[idx] = score

        return scores

    def _compute_colbert_scores(self, query_colbert: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(self.colbert_embeddings))

        for idx, doc_colbert in enumerate(self.colbert_embeddings):
            score = self.model.colbert_score(query_colbert, doc_colbert).item()
            scores[idx] = score

        return scores

    def retrieve(self, query: str, k: int = None) -> List[Document]:
        if k is None:
            k = self.k

        print(f"INFO: BGE-M3 multi-functional retrieval for: '{query[:50]}...'")

        query_embeddings = self.model.encode(
            [query],
            batch_size=1,
            max_length=8192,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True
        )

        dense_scores = self._compute_dense_scores(query_embeddings['dense_vecs'][0])
        sparse_scores = self._compute_sparse_scores(query_embeddings['lexical_weights'][0])
        colbert_scores = self._compute_colbert_scores(query_embeddings['colbert_vecs'][0])

        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
        sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-8)
        colbert_scores = (colbert_scores - colbert_scores.min()) / (colbert_scores.max() - colbert_scores.min() + 1e-8)

        hybrid_scores = (
            self.dense_weight * dense_scores +
            self.sparse_weight * sparse_scores +
            self.colbert_weight * colbert_scores
        )

        top_k_indices = np.argsort(hybrid_scores)[::-1][:k]

        print(f"  - Dense scores (top-3): {dense_scores[top_k_indices[:3]]}")
        print(f"  - Sparse scores (top-3): {sparse_scores[top_k_indices[:3]]}")
        print(f"  - ColBERT scores (top-3): {colbert_scores[top_k_indices[:3]]}")
        print(f"  - Hybrid scores (top-3): {hybrid_scores[top_k_indices[:3]]}")

        results = [self.documents[idx] for idx in top_k_indices]

        return results
