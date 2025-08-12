# ---------------------------------------------------------------------------

import numpy as np
from abc import ABC, abstractmethod
from loguru import logger

from typing import List, Optional, Union

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_cohere import CohereEmbeddings

from src.config import settings

# ---------------------------------------------------------------------------
# - - = ABSTRACT BASE = - -
# ---------------------------------------------------------------------------

class EmbeddingStrategy(ABC):
    """abstract base class for embedding strategies."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

# ---------------------------------------------------------------------------
# - - = OPENAI EMBEDDING STRATEGY = - -
# ---------------------------------------------------------------------------

class OpenAIEmbeddingStrategy(EmbeddingStrategy):
    """OpenAI embeddings with caching and batch processing."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=settings.openai_api_key
        )
        self._cache = {}
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """embed documents with batching."""
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
            
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """embed query with caching."""
        if text in self._cache:
            return self._cache[text]
        
        embedding = self.embeddings.embed_query(text)
        self._cache[text] = embedding
        return embedding

# ---------------------------------------------------------------------------
# - - = LOCAL EMBEDDING STRATEGY = - -
# ---------------------------------------------------------------------------

class LocalEmbeddingStrategy(EmbeddingStrategy):
    """local embeddings using sentence-transformers."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

# ---------------------------------------------------------------------------
# - - = COHERE EMBEDDING STRATEGY = - -
# ---------------------------------------------------------------------------

class CohereEmbeddingStrategy(EmbeddingStrategy):
    """
    cohere embeddings with input type specification.
    """
    
    def __init__(self, model_name: str = "embed-english-v3.0"):
        self.embeddings = CohereEmbeddings(
            model=model_name,
            cohere_api_key=settings.cohere_api_key
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

# ---------------------------------------------------------------------------
# - - = HYBRID EMBEDDINGS = - -
# ---------------------------------------------------------------------------

class HybridEmbeddings:
    """
    hybrid embeddings combining multiple embedding models.
    useful for ensemble approaches.
    """
    
    def __init__(self, strategies: List[EmbeddingStrategy], weights: Optional[List[float]] = None):
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        
        if len(self.weights) != len(self.strategies):
            raise ValueError("Number of weights must match number of strategies")
        
        if abs(sum(self.weights) - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """combine embeddings from multiple strategies."""
        all_embeddings = []
        
        for strategy, weight in zip(self.strategies, self.weights):
            embeddings = strategy.embed_documents(texts)
            weighted_embeddings = [[v * weight for v in emb] for emb in embeddings]
            all_embeddings.append(weighted_embeddings)
        
        # combine embeddings
        combined = []
        for i in range(len(texts)):
            combined_embedding = []
            for strategy_embeddings in all_embeddings:
                if not combined_embedding:
                    combined_embedding = strategy_embeddings[i]
                else:
                    combined_embedding = [
                        a + b for a, b in zip(combined_embedding, strategy_embeddings[i])
                    ]
            combined.append(combined_embedding)
        
        return combined
    
    def embed_query(self, text: str) -> List[float]:
        """combine query embeddings from multiple strategies."""
        combined_embedding = []
        
        for strategy, weight in zip(self.strategies, self.weights):
            embedding = strategy.embed_query(text)
            weighted_embedding = [v * weight for v in embedding]
            
            if not combined_embedding:
                combined_embedding = weighted_embedding
            else:
                combined_embedding = [
                    a + b for a, b in zip(combined_embedding, weighted_embedding)
                ]
        
        return combined_embedding

# ---------------------------------------------------------------------------
# - - = EMBEDDING FACTORY = - -
# ---------------------------------------------------------------------------

class EmbeddingFactory:
    """factory for creating embedding strategies."""
    
    @staticmethod
    def create_embeddings(
        embedding_type: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> EmbeddingStrategy:
        """create embedding strategy based on type."""
        
        # use settings if not provided
        embedding_type = embedding_type or settings.embedding_type
        
        strategies = {
            "openai": (OpenAIEmbeddingStrategy, settings.embedding_model),
            "local": (LocalEmbeddingStrategy, settings.local_embedding_model),
            "cohere": (CohereEmbeddingStrategy, settings.cohere_embedding_model),
        }
        
        if embedding_type not in strategies:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
        
        strategy_class, default_model = strategies[embedding_type]
        model = model_name or default_model
        
        logger.info(f"Creating {embedding_type} embeddings with model {model}")
        return strategy_class(model)