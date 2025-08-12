# ---------------------------------------------------------------------------
import os
import shutil
from pathlib import Path

from loguru import logger # logs

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


from src.rag_chain import RAGChain, ConversationalRAG
from src.config import settings

# ---------------------------------------------------------------------------
# - - = FASTAPI APP SETUP = - -
# ---------------------------------------------------------------------------

app = FastAPI(title="Modern RAG System API", version="1.0.0")

# cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# initialize rag chain (uses settings.py defaults)
rag_chain = ConversationalRAG()

# ---------------------------------------------------------------------------
# - - = REQUEST/RESPONSE MODELS = - -
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = 5
    return_sources: Optional[bool] = True
    verbose: Optional[bool] = False

class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]]
    metadata: Dict[str, Any]
    retrieved_documents: Optional[List[Dict[str, Any]]]

class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int
    chunks_stored: int

class HealthResponse(BaseModel):
    status: str
    vector_store: str
    embedding_model: str
    llm_model: str

@app.get("/", response_model=HealthResponse)
async def health_check():
    """health check endpoint."""
    return HealthResponse(
        status="healthy",
        vector_store=settings.vector_store_type,
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model
    )

@app.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    """upload and process a document."""
    
    # validate file type
    allowed_extensions = ['.pdf', '.txt', '.md', '.docx']
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not supported. Allowed: {allowed_extensions}"
        )
    
    # save uploaded file
    upload_path = settings.upload_dir / file.filename
    settings.upload_dir.mkdir(exist_ok=True)
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # process document
        result = rag_chain.ingest_documents([str(upload_path)])
        
        logger.info(f"Successfully processed {file.filename}")
        return IngestResponse(**result)
    
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # cleanup
        if upload_path.exists():
            upload_path.unlink()

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """query the rag system."""
    try:
        result = rag_chain.query(
            question=request.question,
            k=request.k,
            return_sources=request.return_sources,
            verbose=request.verbose
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Error querying RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-history")
async def clear_conversation_history():
    """clear conversation history."""
    rag_chain.clear_history()
    return {"status": "success", "message": "Conversation history cleared"}

@app.post("/bulk-ingest", response_model=IngestResponse)
async def bulk_ingest(directory: str):
    """ingest all documents from a directory."""
    path = Path(directory)
    
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Directory {directory} not found")
    
    # find all supported files
    file_paths = []
    for ext in ['.pdf', '.txt', '.md', '.docx']:
        file_paths.extend(path.rglob(f"*{ext}"))
    
    if not file_paths:
        raise HTTPException(status_code=404, detail="No supported files found in directory")
    
    try:
        result = rag_chain.ingest_documents([str(f) for f in file_paths])
        return IngestResponse(**result)
    
    except Exception as e:
        logger.error(f"Error in bulk ingest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)