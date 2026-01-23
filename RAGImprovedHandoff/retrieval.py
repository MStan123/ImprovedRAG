from pathlib import Path
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from qdrant_client.models import Filter, FieldCondition, MatchValue
from bm25 import bm25_search
from vector_store import client
from config import COLLECTION_SUMMARIES
from embeddings import embeddings
from documents import summary_documents
from reranker import summary_compressor
from logger_setup import setup_logger

logger = setup_logger()

def hybrid_summary_search(query, top_k=30, category=None):
    # --- BM25 ---
    bm25_docs = bm25_search(query, top_k=top_k)

    # --- Qdrant semantic search ---
    query_embedding = embeddings.embed_query(query)
    filter_condition = None
    if category:
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="metadata.category",
                    match=MatchValue(value=category)
                )
            ]
        )

    search_result = client.query_points(
        collection_name=COLLECTION_SUMMARIES,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
        query_filter=filter_condition,
    )

    qdrant_docs = []
    for hit in search_result.points:
        file_name = hit.payload.get("file")
        for doc in summary_documents:
            if doc.metadata["file"] == file_name:
                qdrant_docs.append(doc)
                break

    # --- Объединяем и уникализируем по файлу ---
    combined_docs = bm25_docs + qdrant_docs
    seen = set()
    final_docs = []
    for doc in combined_docs:
        file_key = doc.metadata["file"]
        if file_key not in seen:
            final_docs.append(doc)
            seen.add(file_key)

    return final_docs[:top_k]

__all__ = ["hybrid_summary_search"]