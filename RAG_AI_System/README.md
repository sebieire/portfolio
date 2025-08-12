# Modern RAG System (2024-2025)

A Retrieval-Augmented Generation (RAG) system using techniques in document retrieval and generation (as of 2025).

## 📖 Summary

This is effectively a boilerplate project which has started in 2024 (and been mostly rewritten or modified in 2025).
The original purpose is the integration of some of it's components and code logic into a large proprietary project that I am currently working on.

This repo was recently updated and cleaned to be used as part of the portfolio.
Among other technologies it utilises:
- **LangChain** (pre-built loaders, chunking)
- multiple (optionals with fallbacks) **embedding** technologies (OpenAI, local, Cohere)
- using **vector DB** ChromaDB (local) with alternative option for production
- **hybrid retrieval** (vector and BM25) with weighted scoring (default: 0.5)
- **re-ranking** (Cohere)
- max marginal relevance (**MMR**)
- parent-child retrieval & hypothetical document embeddings (**HyDE**)
- **fast API** and **Streamlit** integrations

Note: check comprehensive feature list below for further details.
The build integrates a provisional front-end GUI with a chat like interface.
The separation between a backend API and frontend system offers easy extensibility.

Note: the functionality for (local) semantic chunks (_semantic_chunking) is not yet implemented in this version.

The demo.py file (create_sample_documents function) contains AI generated test samples (same with sample data content).
Review the 'src/config.py' file for all configurations.
Enjoy!



## 🚀 Comprehensive Feature List

### Core RAG Capabilities
- **Hybrid Search**: Combines semantic (vector) and keyword (BM25) search with configurable weighted scoring
- **Multi-Query Expansion**: Automatically generates 3+ query variations for improved recall
- **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical answers to improve semantic matching
- **Advanced Reranking**: Uses Cohere's cross-encoder (rerank-english-v3.0) with GPT-4 fallback
- **MMR (Maximum Marginal Relevance)**: Reduces redundancy while maintaining relevance (configurable lambda)
- **Conversational Memory**: Maintains context across 10 queries, includes last 5 in prompt
- **Parent-Child Chunk Retrieval**: Each chunk aware of neighbors with 200-char previews for context expansion
- **Smart Chunking**: RecursiveCharacterTextSplitter with 1000 tokens/200 overlap, respects natural boundaries

### Embedding & Vector Store Architecture
- **Multiple Embedding Strategies**: 
  - OpenAI (text-embedding-3-small) - 1536 dimensions
  - Local models (BAAI/bge-small-en-v1.5) via Sentence Transformers
  - Cohere (embed-english-v3.0) for multilingual support
- **Vector Databases**: 
  - ChromaDB with HNSW indexing for development
  - Qdrant support for production scale
- **Configurable via settings.py**: All models and parameters centralized

### Technical Stack
- **LLM Models**: OpenAI GPT-5/GPT-4o-mini/GPT-4 (configurable temperature)
- **Orchestration**: LangChain for component integration
- **Backend**: FastAPI with auto-generated OpenAPI docs, CORS support
- **Frontend**: Streamlit for rapid prototyping with chat interface
- **Document Processing**: PyPDF, python-docx, Unstructured, BeautifulSoup4
- **Search Algorithms**: BM25Okapi for keyword matching, cosine similarity for vectors
- **Data Validation**: Pydantic for type safety and settings management

### Advanced Features
- **Source Attribution**: Every answer includes document citations with metadata
- **Similarity Threshold Filtering**: Configurable relevance cutoff (default 0.7)
- **Batch Processing**: Efficient embedding generation for large document sets
- **Token Counting**: tiktoken integration for OpenAI token management
- **Evaluation Metrics**: RAGAS integration for RAG quality assessment


## 📋 Prerequisites

- Python 3.11+
- OpenAI API key
- (Optional) Cohere API key for reranking
- (Optional) Docker for containerized deployment


## 🛠️ Installation

### Local Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd modern-rag-system
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Docker Setup

```bash
docker-compose up -d
```

## 🚀 Quick Start

### 1. Start the API server:
```bash
uvicorn src.api:app --reload --port 8000
```

### 2. Start the Streamlit UI:
```bash
streamlit run streamlit_app.py
```

### 3. Access the application:
- API: http://localhost:8000
- UI: http://localhost:8501
- API Docs: http://localhost:8000/docs


## 📖 Usage

### Run Demo Script

Test the system without starting the API:
```bash
python demo.py
```

This will:
- Create sample documents
- Test basic RAG functionality
- Test conversational memory
- Test advanced features (if API keys are configured)

### Via Streamlit UI

1. Upload documents using the sidebar
2. Ask questions in the chat interface
3. View sources and relevance scores
4. Enable verbose mode to see retrieved chunks

### Via API

```python
import requests

# Upload document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload",
        files={"file": f}
    )

# Query
response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "What is RAG?",
        "k": 5,
        "return_sources": True
    }
)
print(response.json())
```

### Programmatic Usage

```python
from src.rag_chain import RAGChain

# Initialize
rag = RAGChain(
    vector_store_type="chroma",
    embedding_type="openai"
)

# Ingest documents
rag.ingest_documents(["path/to/document.pdf"])

# Query
result = rag.query(
    question="What are the main components of RAG?",
    k=5
)
print(result["answer"])
```


## 🎯 Advanced Configuration

### Retrieval Settings
```python
# In .env or settings
EMBEDDING_TYPE=openai    # Options: openai, local, cohere
CHUNK_SIZE=1000          # Token size for chunks
CHUNK_OVERLAP=200        # Overlap between chunks
TOP_K=10                 # Documents to retrieve
RERANK_TOP_K=5          # Documents after reranking
USE_HYDE=false          # Enable HyDE (default: false)
USE_MULTI_QUERY=true    # Enable multi-query
USE_RERANKING=true      # Enable reranking
STREAMING=false         # Enable streaming responses
```

### Hybrid Search Tuning
```python
ENABLE_HYBRID_SEARCH=true
BM25_WEIGHT=0.5  # Balance between BM25 and vector search
```


## 📊 Evaluation

The system includes evaluation metrics:
- Context Precision
- Context Recall
- Answer Relevancy
- Retrieval Quality

Run evaluation:
```python
from src.rag_chain import RAGChain

rag = RAGChain()
test_queries = [
    {"question": "What is RAG?", "expected_answer": "..."},
    # Add more test cases
]
results = rag.evaluate_retrieval(test_queries)
```


## 🏗️ Architecture

```
Modern RAG System
├── Document Processing
│   ├── Smart Chunking
│   ├── Metadata Extraction
│   └── Contextual Enhancement
├── Retrieval Layer
│   ├── Vector Search (Semantic)
│   ├── BM25 Search (Keyword)
│   ├── Hybrid Fusion
│   └── MMR Diversification
├── Enhancement Layer
│   ├── Multi-Query Expansion
│   ├── HyDE Generation
│   └── Reranking (Cohere/LLM)
└── Generation Layer
    ├── Context Formation
    ├── Prompt Engineering
    └── Response Generation
```


## 🔧 Customization

### Add Custom Embeddings
```python
from src.embeddings import EmbeddingStrategy

class CustomEmbedding(EmbeddingStrategy):
    def embed_documents(self, texts):
        # Your implementation
        pass
    
    def embed_query(self, text):
        # Your implementation
        pass
```

### Add Custom Vector Store
```python
from src.vector_store import VectorStoreStrategy

class CustomVectorStore(VectorStoreStrategy):
    def add_documents(self, documents):
        # Your implementation
        pass
    
    def similarity_search(self, query, k=4):
        # Your implementation
        pass
```


## 📈 Performance Tips

1. **Chunking**: Adjust chunk size based on your documents
2. **Caching**: Enable embedding caching for repeated queries
3. **Batch Processing**: Use batch ingestion for large document sets
4. **Vector Store**: Use Qdrant for production scale
5. **Reranking**: Use Cohere for best accuracy

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce chunk_size or batch_size
2. **Slow Retrieval**: Enable caching, reduce top_k
3. **Poor Results**: Tune chunk_overlap, enable reranking
4. **API Errors**: Check API keys and rate limits


## 📝 License

MIT License - See LICENSE file for details


## Author

[sebieire](https://github.com/sebieire/)


## 📚 References

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [ChromaDB](https://www.trychroma.com/)
- [Qdrant](https://qdrant.tech/)
- [Cohere Rerank](https://cohere.com/rerank)



## 🏆 Advanced Techniques Implemented

- **Semantic Chunking**: Context-aware document splitting
- **Parent-Child Retrieval**: Maintains document context
- **Cross-Encoder Reranking**: Improves precision
- **Ensemble Embeddings**: Combines multiple models
- **Adaptive Retrieval**: Adjusts strategy based on query type
- **Conversation Memory**: Maintains chat context

---

Built with ❤️ for AI apps (2024-2025) and a sprinkle of salt and pepper :P