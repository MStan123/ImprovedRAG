from dataclasses import dataclass
from logger_setup import setup_logger

logger = setup_logger()

@dataclass
class CostStats:
    llm_calls: int = 0
    cache_hits: int = 0
    spent_tokens: int = 0
    saved_tokens: int = 0
    handoff_count: int = 0
    cached_responses: int = 0

stats = CostStats()

def print_cost_report():
    total_queries = stats.llm_calls + stats.cache_hits
    potential_tokens = stats.spent_tokens + stats.saved_tokens

    logger.info("💰 COST REPORT")
    logger.info(f"Total queries: {total_queries}")
    logger.info(f"LLM calls: {stats.llm_calls}")
    logger.info(f"Cache hits: {stats.cache_hits}")
    logger.info(f"Cache hit rate: {(stats.cache_hits / total_queries * 100) if total_queries else 0:.1f}%")
    logger.info(f"Spent tokens: {stats.spent_tokens}")
    logger.info(f"Saved tokens: {stats.saved_tokens}")
    logger.info(f"Potential tokens without cache: {potential_tokens}")
    logger.info(f"Token savings: {(stats.saved_tokens / potential_tokens * 100) if potential_tokens else 0:.1f}%")
    logger.info(f"Human handoffs: {stats.handoff_count}")

__all__ = ["stats", "print_cost_report"]