from langchain_community.document_compressors import FlashrankRerank
from logger_setup import setup_logger

logger = setup_logger()

FlashrankRerank.model_rebuild()

summary_compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=7)

logger.info("FlashRank rerankers initialized")

__all__ = ["summary_compressor"]
