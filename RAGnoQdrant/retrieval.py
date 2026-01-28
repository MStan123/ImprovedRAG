from pathlib import Path
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from qdrant_client.models import Filter, FieldCondition, MatchValue
from bm25 import bm25_search
# from vector_store import client
from config import COLLECTION_SUMMARIES
# from embeddings import embeddings
from documents import summary_documents
from langchain_core.documents import Document
from reranker import summary_compressor
from logger_setup import setup_logger

logger = setup_logger()

def hybrid_summary_search(query: str, top_k: int = 15, category: str = None) -> list[Document]:
    """
    Упрощённый поиск БЕЗ Qdrant:
    BM25 → FlashRank
    """

    # 1. BM25 широкий поиск
    bm25_docs = bm25_search(query, top_k=top_k)

    if not bm25_docs:
        return []

    # 2. Фильтрация по категории (опционально)
    if category:
        bm25_docs = [
            doc for doc in bm25_docs
            if doc.metadata.get("category") == category
        ]

    return bm25_docs[:top_k]

__all__ = ["hybrid_summary_search"]
