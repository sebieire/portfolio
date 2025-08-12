# ---------------------------------------------------------------------------


import uuid
import chromadb

from loguru import logger

from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

from chromadb.config import Settings as ChromaSettings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# langchain imports
from langchain.schema import Document
from langchain_community.vectorstores import Chroma, Qdrant
from langchain.vectorstores.base import VectorStore

# local imports
from src.config import settings
from src.embeddings import EmbeddingStrategy

# ---------------------------------------------------------------------------
# - - = ABSTRACT BASE STRATEGY = - -
# ---------------------------------------------------------------------------

class VectorStoreStrategy(ABC):
    """abstract base class for vector store strategies."""
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        pass
    
    @abstractmethod
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        pass
    
    @abstractmethod
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        pass
    
    @abstractmethod
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        pass

# ---------------------------------------------------------------------------
# - - = CHROMA VECTOR STORE = - -
# ---------------------------------------------------------------------------

class ChromaVectorStore(VectorStoreStrategy):
    """
    ChromaDB vector store implementation
    """
    
    def __init__(self, embedding_strategy: EmbeddingStrategy, persist_directory: str = "./chroma_db"):
        self.embedding_strategy = embedding_strategy
        self.persist_directory = persist_directory
        
        # initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # create or get collection
        self.collection_name = settings.collection_name
        try:
            self.collection = self.client.get_collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        # initialize Langchain Chroma
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embedding_strategy
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """add documents with deduplication."""
        # check for existing documents
        existing_hashes = set()
        if self.collection.count() > 0:
            results = self.collection.get()
            if results['metadatas']:
                existing_hashes = {
                    m.get('file_hash', '') for m in results['metadatas']
                }
        
        # filter out duplicates
        new_documents = [
            doc for doc in documents
            if doc.metadata.get('file_hash') not in existing_hashes
        ]
        
        if not new_documents:
            logger.info("No new documents to add (all duplicates)")
            return []
        
        # add documents
        ids = [str(uuid.uuid4()) for _ in new_documents]
        self.vectorstore.add_documents(new_documents, ids=ids)
        
        logger.info(f"Added {len(new_documents)} new documents to ChromaDB")
        return ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """perform similarity search."""
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """perform similarity search with scores."""
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """mmr search for diversity."""
        return self.vectorstore.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )

# ---------------------------------------------------------------------------
# - - = QDRANT VECTOR STORE = - -
# ---------------------------------------------------------------------------

class QdrantVectorStore(VectorStoreStrategy):
    """
    Qdrant vector store implementation
    """
    
    def __init__(self, embedding_strategy: EmbeddingStrategy):
        self.embedding_strategy = embedding_strategy
        
        # initialize Qdrant client
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        
        self.collection_name = settings.collection_name
        
        # create collection if it doesn't exist
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Using existing Qdrant collection: {self.collection_name}")
        except:
            # get embedding dimension
            sample_embedding = embedding_strategy.embed_query("sample text")
            vector_size = len(sample_embedding)
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created new Qdrant collection: {self.collection_name}")
        
        # initialize Langchain Qdrant
        self.vectorstore = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embedding_strategy
        )
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """add documents to Qdrant."""
        ids = [str(uuid.uuid4()) for _ in documents]
        self.vectorstore.add_documents(documents, ids=ids)
        logger.info(f"Added {len(documents)} documents to Qdrant")
        return ids
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """perform similarity search."""
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """perform similarity search with scores."""
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5
    ) -> List[Document]:
        """mmr search for diversity."""
        return self.vectorstore.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )

# ---------------------------------------------------------------------------
# - - = VECTOR STORE FACTORY = - -
# ---------------------------------------------------------------------------

class VectorStoreFactory:
    """factory for creating vector stores."""
    
    @staticmethod
    def create_vector_store(
        store_type: str,
        embedding_strategy: EmbeddingStrategy
    ) -> VectorStoreStrategy:
        """create vector store based on type."""
        
        stores = {
            "chroma": ChromaVectorStore,
            "qdrant": QdrantVectorStore,
        }
        
        if store_type not in stores:
            raise ValueError(f"Unknown vector store type: {store_type}")
        
        store_class = stores[store_type]
        logger.info(f"Creating {store_type} vector store")
        
        if store_type == "chroma":
            return store_class(embedding_strategy, settings.chroma_persist_dir)
        else:
            return store_class(embedding_strategy)