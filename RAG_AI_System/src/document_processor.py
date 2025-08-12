# ---------------------------------------------------------------------------

import hashlib
import tiktoken

from loguru import logger

from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (PyPDFLoader, TextLoader, UnstructuredMarkdownLoader, Docx2txtLoader)
from langchain.schema import Document


from src.config import settings

# ---------------------------------------------------------------------------
# DOCUMENT PROCESSOR CLASS 
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """advanced document processing with smart chunking and metadata extraction."""
    
    def __init__(self):
        # ---------------------------------------------------------------------------
        # INITIALIZATION 
        # ---------------------------------------------------------------------------
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=settings.separators,
            length_function=self._tiktoken_len,
        )
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    def _tiktoken_len(self, text: str) -> int:
        """calculate token length using tiktoken."""
        return len(self.encoding.encode(text))
    
    def _get_file_hash(self, file_path: str) -> str:
        """generate unique hash for file content."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    # ---------------------------------------------------------------------------
    # DOCUMENT LOADING 
    # ---------------------------------------------------------------------------
    
    def load_document(self, file_path: str) -> List[Document]:
        """load document based on file type."""
        path = Path(file_path)
        
        loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
            '.docx': Docx2txtLoader,
        }
        
        loader_class = loaders.get(path.suffix.lower())
        if not loader_class:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        
        loader = loader_class(str(path))
        documents = loader.load()
        
        # add metadata
        file_hash = self._get_file_hash(str(path))
        for doc in documents:
            doc.metadata.update({
                'source': str(path),
                'file_name': path.name,
                'file_type': path.suffix,
                'file_hash': file_hash,
            })
        
        return documents
    
    # ---------------------------------------------------------------------------
    # DOCUMENT CHUNKING 
    # ---------------------------------------------------------------------------
    
    def chunk_documents(
        self, 
        documents: List[Document],
        use_semantic_chunking: bool = False
    ) -> List[Document]:
        """
        advanced chunking with multiple strategies.
        """
        if use_semantic_chunking:
            return self._semantic_chunking(documents)
        
        chunks = []
        for doc in documents:
            doc_chunks = self.text_splitter.split_documents([doc])
            
            # add chunk metadata
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(doc_chunks),
                    'chunk_size': self._tiktoken_len(chunk.page_content),
                })
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def _semantic_chunking(self, documents: List[Document]) -> List[Document]:
        """
        semantic chunking -> sentence embeddings similarity
        for now just a placeholder (for me TODO: implement functionality later)
        """
        # for now only a fall back
        # use sentence transformers in production later on
        return self.chunk_documents(documents, use_semantic_chunking=False)
    
    # ---------------------------------------------------------------------------
    # DIRECTORY PROCESSING 
    # ---------------------------------------------------------------------------
    
    def process_directory(self, directory: str) -> List[Document]:
        """process all supported documents in a directory."""
        path = Path(directory)
        all_chunks = []
        
        supported_extensions = ['.pdf', '.txt', '.md', '.docx']
        
        for ext in supported_extensions:
            for file_path in path.rglob(f"*{ext}"):
                try:
                    logger.info(f"Processing {file_path}")
                    documents = self.load_document(str(file_path))
                    chunks = self.chunk_documents(documents)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        return all_chunks
    
    # ---------------------------------------------------------------------------
    # CONTEXTUAL CHUNKING 
    # ---------------------------------------------------------------------------
    
    def add_contextual_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        add context from neighboring chunks (parent-child retrieval).
        """
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            # group chunks by source document
            same_doc_chunks = [
                c for c in chunks 
                if c.metadata.get('file_hash') == chunk.metadata.get('file_hash')
            ]
            
            chunk_index = chunk.metadata.get('chunk_index', 0)
            
            # add references to parent and child chunks
            chunk.metadata['has_parent'] = chunk_index > 0
            chunk.metadata['has_child'] = chunk_index < len(same_doc_chunks) - 1
            
            if chunk_index > 0:
                chunk.metadata['parent_content_preview'] = same_doc_chunks[chunk_index - 1].page_content[:200]
            
            if chunk_index < len(same_doc_chunks) - 1:
                chunk.metadata['child_content_preview'] = same_doc_chunks[chunk_index + 1].page_content[:200]
            
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks