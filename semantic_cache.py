from langchain_core.documents import Document
from vector_store import cache_vector_store
from logger_setup import setup_logger
from langdetect import detect

logger = setup_logger()

class RAGSemanticCache:
    def __init__(self, vector_store, threshold: float = 0.7):
        self.vector_store = vector_store
        self.threshold = threshold

    def retrieve_cached_response(self, query: str):
        """Возвращает Document с кэшированным ответом или None"""
        query_lang = detect(query)
        results = self.vector_store.similarity_search_with_score(
            query,
            k=1,
            score_threshold=self.threshold
        )
        if results:
            best, score = results[0]
            if best.metadata.get("language") == query_lang:
                return best

    def store_response(self, query: str, response: str, tokens: int):
        language = detect(query)
        doc = Document(
            page_content=query,  # эмбеддится именно запрос
            metadata={
                "response": response,
                "tokens": tokens,
                "language": language
            }
        )
        self.vector_store.add_documents([doc])

semantic_cache = RAGSemanticCache(cache_vector_store, threshold=0.7)

__all__ = ["semantic_cache"]