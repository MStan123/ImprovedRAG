import json
from langchain_core.documents import Document
from logger_setup import setup_logger
from pathlib import Path

logger = setup_logger()

def load_summary_documents(index_path: str = "output1.json") -> list[Document]:
    with open(index_path, 'r', encoding='utf-8') as f:
        index_data = json.load(f)

    logger.info(f"Loaded {len(index_data['chunks'])} summary chunks from {index_path}")

    documents = []
    for i, chunk_info in enumerate(index_data["chunks"]):
        content = f"{chunk_info['summary']}\n" + "\n".join(chunk_info.get('questions', []))
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "file": chunk_info["file"],
                    "summary": chunk_info["summary"],
                    "chunk_id": i + 1
                }
            )
        )

    logger.info(f"Created {len(documents)} summary Document objects")
    return documents

def load_detailed_chunks_cache(chunks_dir: str = "/home/user/PyCharmMiscProject/RAG/chunks") -> dict[str, str]:
    """
    Загружает все детальные чанки в память при старте.
    Возвращает словарь: {filename: content}
    """
    chunks_path = Path(chunks_dir)

    if not chunks_path.exists():
        logger.warning(f"Chunks directory not found: {chunks_dir}")
        return {}

    cache = {}

    # Загружаем все файлы чанков
    for chunk_file in chunks_path.iterdir():
        if chunk_file.is_file():
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    cache[chunk_file.name] = f.read()
            except Exception as e:
                logger.error(f"Failed to load chunk {chunk_file.name}: {e}")

    logger.info(f"✅ Cached {len(cache)} detailed chunks in memory ({sum(len(v) for v in cache.values()) // 1024} KB)")
    return cache


# Инициализация при импорте модуля
summary_documents = load_summary_documents()
detailed_chunks_cache = load_detailed_chunks_cache(chunks_dir="/home/user/PyCharmMiscProject/RAG/chunks")  # Укажите ваш путь

__all__ = ["summary_documents", "load_summary_documents", "detailed_chunks_cache"]
