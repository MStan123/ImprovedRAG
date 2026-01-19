from langchain_openai import AzureChatOpenAI
from config import (
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_DEPLOYMENT_FALLBACK, AZURE_OPENAI_API_VERSION_FALLBACK
)
from logger_setup import setup_logger

logger = setup_logger()

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_version=AZURE_OPENAI_API_VERSION,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=1,
    max_retries=3,
    max_tokens=3000,
    timeout=30.0
)

fallback_llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_FALLBACK,
    api_version=AZURE_OPENAI_API_VERSION_FALLBACK,
    api_key=AZURE_OPENAI_API_KEY,
    temperature=1,
    max_retries=3,
    max_tokens=3000,
    timeout=30.0
)

logger.info("Azure OpenAI clients initialized (primary + fallback)")

__all__ = ["llm", "fallback_llm"]