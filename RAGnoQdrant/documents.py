import json
from langchain_core.documents import Document
from logger_setup import setup_logger

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


summary_documents = load_summary_documents()

__all__ = ["summary_documents", "load_summary_documents"]