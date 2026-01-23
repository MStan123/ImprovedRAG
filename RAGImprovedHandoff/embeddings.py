from logger_setup import setup_logger
from langchain_huggingface import HuggingFaceEmbeddings

logger = setup_logger()

logger.info("Embedding model Loading")

try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    logger.info("Embeddings model loaded")

except Exception as e:
    logger.error(f"Embeddings model failed: {e}", exc_info=True)
    logger.error(f"Error type: {type(e).__name__}")

    raise



__all__ = ["embeddings"]