"""
configuration settings for the rag application.
@sebieire [https://github.com/sebieire/] (2024 / updated 2025)
"""

# ---------------------------------------------------------------------------

from pathlib import Path

from pydantic_settings import BaseSettings
from typing import Optional, Literal


# ---------------------------------------------------------------------------
# - - = SETTINGS CLASS = - -
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    
    # ---------------------------------------------------------------------------
    # - - = API KEYS = - -
    # ---------------------------------------------------------------------------
    openai_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    
    
    # ---------------------------------------------------------------------------
    # - - = MODELS = - -
    # ---------------------------------------------------------------------------
    embedding_type: Literal["openai", "local", "cohere"] = "openai" # "local"

    # @sebieire updated 2025
    embedding_model: str = "text-embedding-3-small" # e.g. OpenAI embedding model
    local_embedding_model: str = "BAAI/bge-small-en-v1.5" # e.g. huggingface model
    cohere_embedding_model: str = "embed-english-v3.0" # e.g. Cohere embedding model

    llm_model: str = "gpt-4o-mini" # OpenAI LLM model GPT-4 (optional GPT-5 - not required)
    llm_temperature: float = 0.7
    reranker_model: str = "rerank-english-v3.0"
    use_cohere_reranker: bool = True
    
    
    # ---------------------------------------------------------------------------
    # - - = VECTOR STORE = - -
    # ---------------------------------------------------------------------------
    vector_store_type: Literal["chroma", "qdrant"] = "chroma"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "rag_documents"
    
    # ---------------------------------------------------------------------------
    # - - = CHUNKING = - -
    # ---------------------------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list = ["\n\n", "\n", ". ", " ", ""]
    
    # ---------------------------------------------------------------------------
    # - - = RETRIEVAL = - -
    # ---------------------------------------------------------------------------
    top_k: int = 10
    rerank_top_k: int = 5
    similarity_threshold: float = 0.7
    use_mmr: bool = True
    mmr_diversity_score: float = 0.3
    
    # ---------------------------------------------------------------------------
    # - - = HYBRID SEARCH = - -
    # ---------------------------------------------------------------------------
    enable_hybrid_search: bool = True
    bm25_weight: float = 0.5
    
    # ---------------------------------------------------------------------------
    # - - = ADVANCED FEATURES = - -
    # ---------------------------------------------------------------------------
    use_hyde: bool = False
    use_multi_query: bool = True
    use_reranking: bool = True
    streaming: bool = False
    return_sources: bool = True
    verbose: bool = False
    
    # ---------------------------------------------------------------------------
    # - - = PATHS = - -
    # ---------------------------------------------------------------------------
    data_dir: Path = Path("./data")
    upload_dir: Path = Path("./uploads")
    
    # ---------------------------------------------------------------------------
    # - - = LOGGING = - -
    # ---------------------------------------------------------------------------
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # ignore extra fields in .env

# ---------------------------------------------------------------------------
# - - = SETTINGS INSTANCE = - -
# ---------------------------------------------------------------------------

settings = Settings()