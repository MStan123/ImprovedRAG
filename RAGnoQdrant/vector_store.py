# from qdrant_client import QdrantClient
# from langchain_qdrant import QdrantVectorStore
# from config import QDRANT_URL, QDRANT_PORT, COLLECTION_SUMMARIES, COLLECTION_CACHE
# from embeddings import embeddings
# from documents import summary_documents
# from qdrant_client.models import VectorParams, Distance
# from logger_setup import setup_logger
#
# logger = setup_logger()
#
# client = QdrantClient(url=QDRANT_URL, port=QDRANT_PORT)
#
# # Основное хранилище саммари
#
# if not client.collection_exists(COLLECTION_SUMMARIES):
#     client.create_collection(
#         collection_name=COLLECTION_SUMMARIES,
#         vectors_config=VectorParams(size=384, distance=Distance.COSINE),
#     )
#     summary_store = QdrantVectorStore(
#         client=client,
#         collection_name=COLLECTION_SUMMARIES,
#         embedding=embeddings,
#     )
#     summary_store.add_documents(summary_documents)
#     logger.info(f"Created and populated collection '{COLLECTION_SUMMARIES}'")
# else:
#     logger.info(f"Using existing collection '{COLLECTION_SUMMARIES}'")
#
# # Кэш-хранилище
# if not client.collection_exists(COLLECTION_CACHE):
#     client.create_collection(
#         collection_name=COLLECTION_CACHE,
#         vectors_config=VectorParams(size=384, distance=Distance.COSINE),
#     )
#     logger.info(f"Created cache collection '{COLLECTION_CACHE}'")
# else:
#     logger.info(f"Using existing collection '{COLLECTION_CACHE}'")
#
# cache_vector_store = QdrantVectorStore(
#     client=client,
#     collection_name=COLLECTION_CACHE,
#     embedding=embeddings,
# )
#
# __all__ = ["summary_store", "cache_vector_store", "client"]