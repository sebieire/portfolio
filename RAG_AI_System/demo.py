#!/usr/bin/env python3
"""
demo script for the modern RAG system
demonstrates the system's capabilities without needing the API
"""

# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------
import os
from loguru import logger

from pathlib import Path
from dotenv import load_dotenv
from src.rag_chain import RAGChain, ConversationalRAG
from src.document_processor import DocumentProcessor


# load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# - - = SAMPLE DOC CREATION = - -
# ---------------------------------------------------------------------------
def create_sample_documents():
    """create sample documents for testing"""
    sample_dir = Path("./data/samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # sample doc 1: RAG concepts (basic) -----> text is AI generated
    rag_content = """
    RAG combines retrieval with generation to provide accurate, grounded responses.
    
    The system processes documents by splitting them into chunks, converting each chunk 
    into vector embeddings, and storing them in a vector database like ChromaDB.
    
    When a query comes in, it searches for similar chunks using both semantic similarity 
    and keyword matching (hybrid search). The retrieved chunks provide context for the LLM 
    to generate accurate answers.
    
    Key features implemented:
    - Hybrid search combining vector and BM25
    - Multi-query expansion for better recall
    - Reranking with Cohere for precision
    - HyDE for complex queries
    """
    
    (sample_dir / "rag_basics.md").write_text(rag_content)
    
    # sample doc 2: implementation details ---> text is AI generated
    implementation_content = """
    This RAG implementation uses ChromaDB for vector storage with HNSW indexing for 
    fast similarity search. Documents are chunked with 1000 token size and 200 token 
    overlap to maintain context.
    
    The hybrid retrieval combines semantic search with BM25 keyword matching using 
    configurable weights (default 50/50 split). Retrieved documents are reranked 
    using Cohere's cross-encoder model for improved relevance.
    
    Conversation memory maintains the last 10 exchanges and includes the most recent 
    5 in the prompt for context-aware responses.
    """
    
    (sample_dir / "implementation.md").write_text(implementation_content)
    
    logger.info(f"Created sample documents in {sample_dir}")
    return sample_dir

# ---------------------------------------------------------------------------
# - - = BASIC RAG TESTING = - -
# ---------------------------------------------------------------------------
def test_basic_rag():
    """test basic RAG functionality"""
    print("\n" + "="*50)
    print("Testing Basic RAG System")
    print("="*50)
    
    # initialize RAG
    rag = RAGChain(
        vector_store_type="chroma",
        embedding_type="openai" if os.getenv("OPENAI_API_KEY") else "local"
    )
    
    # create and ingest sample documents
    sample_dir = create_sample_documents()
    result = rag.ingest_documents([
        str(sample_dir / "rag_basics.md"),
        str(sample_dir / "implementation.md")
    ])
    
    print(f"\nIngested {result['documents_processed']} documents")
    print(f"Created {result['chunks_created']} chunks")
    
    # test queries
    test_queries = [
        "How does hybrid search work?",
        "What is the chunk size used?",
        "How does reranking improve results?",
        "Explain the HyDE technique"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        response = rag.query(query, k=3, return_sources=True)
        
        print(f"Answer: {response['answer'][:300]}...")
        print(f"\nSources:")
        for source in response['sources']:
            print(f"   - {source['file']} (relevance: {source.get('relevance_score', 0):.2f})")

# ---------------------------------------------------------------------------
# - - = CONVERSATIONAL RAG TESTING = - -
# ---------------------------------------------------------------------------
def test_conversational_rag():
    """test conversational RAG with memory"""
    print("\n" + "="*50)
    print("Testing Conversational RAG System")
    print("="*50)
    
    # initialize conversational RAG
    rag = ConversationalRAG(
        vector_store_type="chroma",
        embedding_type="openai" if os.getenv("OPENAI_API_KEY") else "local"
    )
    
    # ingest documents
    sample_dir = create_sample_documents()
    rag.ingest_documents([
        str(sample_dir / "rag_basics.md"),
        str(sample_dir / "implementation.md")
    ])
    
    # simulate conversation
    conversation = [
        "What is hybrid search?",
        "How much overlap is used between chunks?",  # follows up on chunking
        "What model does the reranker use?",
        "How many conversations are stored in memory?"  # tests memory feature
    ]
    
    for i, query in enumerate(conversation, 1):
        print(f"\nTurn {i} - User: {query}")
        print("-" * 40)
        
        response = rag.query(query, k=3)
        print(f"Assistant: {response['answer'][:300]}...")
        print(f"(Conversation length: {response['metadata']['conversation_length']})")

# ---------------------------------------------------------------------------
# - - = ADVANCED FEATURES TESTING = - -
# ---------------------------------------------------------------------------
def test_advanced_features():
    """test advanced RAG features"""
    print("\n" + "="*50)
    print("Testing Advanced RAG Features")
    print("="*50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping advanced features test (requires OpenAI API key)")
        return
    
    # initialize with advanced features
    rag = RAGChain(
        vector_store_type="chroma",
        embedding_type="openai"
    )
    
    # update settings for advanced features
    from src.config import settings
    settings.use_hyde = True
    settings.use_multi_query = True
    settings.use_reranking = True
    settings.enable_hybrid_search = True
    
    # reinitialize retriever
    from src.retriever import HybridRetriever
    rag.retriever = HybridRetriever(
        vector_store=rag.vector_store,
        embedding_strategy=rag.embedding_strategy,
        use_hyde=True,
        use_multi_query=True,
        use_reranking=True
    )
    
    # ingest documents
    sample_dir = create_sample_documents()
    rag.ingest_documents([
        str(sample_dir / "rag_basics.md"),
        str(sample_dir / "implementation.md")
    ])
    
    # test with complex query
    complex_query = "Explain how the hybrid retrieval system combines BM25 and vector search with reranking"
    
    print(f"\nComplex Query: {complex_query}")
    print("-" * 40)
    
    response = rag.query(complex_query, k=5, verbose=True)
    
    print(f"Answer: {response['answer'][:500]}...")
    print(f"\nMetadata:")
    print(f"   - Documents retrieved: {response['metadata']['documents_retrieved']}")
    print(f"   - Model used: {response['metadata']['model_used']}")
    
    if response.get('retrieved_documents'):
        print(f"\nRetrieved chunks preview:")
        for i, doc in enumerate(response['retrieved_documents'][:2], 1):
            print(f"   Chunk {i}: {doc['content'][:100]}...")

# ---------------------------------------------------------------------------
# - - = MAIN EXECUTION = - -
# ---------------------------------------------------------------------------
def main():
    """run all tests"""
    print("\nModern RAG System Test Suite")
    print("=" * 50)
    
    # check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Using local embeddings.")
        print("For best results, set your OpenAI API key in .env file")
    
    try:
        # run tests
        test_basic_rag()
        test_conversational_rag()
        test_advanced_features()
        
        print("\n" + "="*50)
        print("All tests completed successfully!")
        print("="*50)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        logger.exception("Test failed")

if __name__ == "__main__":
    main()