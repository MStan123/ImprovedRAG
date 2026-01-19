import logging

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("rag_assistant.log", mode='a', encoding='utf-8')
        ]
    )
    return logging.getLogger("RAG_Assistant")