from sklearn.feature_extraction.text import TfidfVectorizer
from documents import summary_documents
from logger_setup import setup_logger

logger = setup_logger()

documents_texts = [doc.page_content for doc in summary_documents]
vectorizer = TfidfVectorizer()
X_sparse = vectorizer.fit_transform(documents_texts)

logger.info("TF-IDF vectorizer fitted on summaries")

def bm25_search(query: str, top_k: int = 30) -> list:
    query_vec = vectorizer.transform([query])
    scores = (X_sparse @ query_vec.T).toarray().flatten()
    top_idx = scores.argsort()[::-1][:top_k]
    return [summary_documents[i] for i in top_idx]

__all__ = ["bm25_search", "vectorizer"]