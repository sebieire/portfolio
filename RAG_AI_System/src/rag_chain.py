# ---------------------------------------------------------------------------

import json
from loguru import logger

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# local imports
from src.config import settings
from src.retriever import HybridRetriever
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingFactory
from src.vector_store import VectorStoreFactory

# ---------------------------------------------------------------------------
# RAG CHAIN 
# ---------------------------------------------------------------------------

class RAGChain:
    """main rag chain combining all components."""
    
    def __init__(
        self,
        vector_store_type: Optional[str] = None,
        embedding_type: Optional[str] = None,
        streaming: Optional[bool] = None
    ):
        # use settings if not provided
        vector_store_type = vector_store_type or settings.vector_store_type
        embedding_type = embedding_type or settings.embedding_type
        streaming = streaming if streaming is not None else settings.streaming
        
        # initialize embeddings
        self.embedding_strategy = EmbeddingFactory.create_embeddings(embedding_type)
        
        # initialize vector store
        self.vector_store = VectorStoreFactory.create_vector_store(
            vector_store_type,
            self.embedding_strategy
        )
        
        # initialize retriever
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            embedding_strategy=self.embedding_strategy,
            use_hyde=settings.use_hyde,
            use_multi_query=settings.use_multi_query,
            use_reranking=settings.use_reranking
        )
        
        # initialize document processor
        self.document_processor = DocumentProcessor()
        
        # initialize llm
        callbacks = [StreamingStdOutCallbackHandler()] if streaming else []
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key,
            streaming=streaming,
            callbacks=callbacks
        )
        
        # create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided context.
            Use the following pieces of context to answer the question.
            If you don't know the answer based on the context, say you don't know.
            Always cite which document your answer comes from.
            
            Context:
            {context}
            """),
            ("human", "{question}")
        ])
        
        # create chain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def ingest_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """ingest documents into the vector store."""
        all_chunks = []
        
        for file_path in file_paths:
            try:
                # load and process document
                documents = self.document_processor.load_document(file_path)
                chunks = self.document_processor.chunk_documents(documents)
                
                # add contextual information
                chunks = self.document_processor.add_contextual_chunks(chunks)
                
                all_chunks.extend(chunks)
                logger.info(f"Processed {file_path}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # add to vector store
        ids = self.vector_store.add_documents(all_chunks)
        
        # index for bm25 if hybrid search is enabled
        if settings.enable_hybrid_search:
            self.retriever.index_for_bm25(all_chunks)
        
        return {
            "status": "success",
            "documents_processed": len(file_paths),
            "chunks_created": len(all_chunks),
            "chunks_stored": len(ids)
        }
    
    def query(
        self,
        question: str,
        k: int = 5,
        return_sources: Optional[bool] = None,
        verbose: Optional[bool] = None
    ) -> Dict[str, Any]:
        # use settings if not provided
        return_sources = return_sources if return_sources is not None else settings.return_sources
        verbose = verbose if verbose is not None else settings.verbose
        """query the rag system."""
        
        # retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(
            query=question,
            k=k,
            use_mmr=settings.use_mmr
        )
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "metadata": {"documents_retrieved": 0}
            }
        
        # format context
        context = self._format_context(retrieved_docs)
        
        # generate answer
        response = self.chain.invoke({
            "context": context,
            "question": question
        })
        
        # prepare result
        result = {
            "answer": response["text"],
            "metadata": {
                "documents_retrieved": len(retrieved_docs),
                "model_used": settings.llm_model
            }
        }
        
        if return_sources:
            result["sources"] = self._format_sources(retrieved_docs)
        
        if verbose:
            result["retrieved_documents"] = [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                for doc in retrieved_docs
            ]
        
        return result
    
    def _format_context(self, documents: List[Document]) -> str:
        """format documents as context for the llm."""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            
            context_parts.append(
                f"Document {i} (Source: {source}):\n{content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """format sources for the response."""
        sources = []
        seen_sources = set()
        
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            if source not in seen_sources:
                seen_sources.add(source)
                sources.append({
                    "file": source,
                    "type": doc.metadata.get('file_type', 'unknown'),
                    "relevance_score": doc.metadata.get('rerank_score', 
                                     doc.metadata.get('combined_score', 0))
                })
        
        return sorted(sources, key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    def evaluate_retrieval(self, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """evaluate retrieval quality using test queries."""
        from ragas.metrics import (
            context_precision,
            context_recall,
            context_relevancy,
            answer_relevancy
        )
        
        results = []
        
        for test_case in test_queries:
            question = test_case["question"]
            expected_answer = test_case.get("expected_answer", "")
            
            # get retrieval results
            response = self.query(question, return_sources=True, verbose=True)
            
            # calculate metrics
            metrics = {
                "question": question,
                "answer": response["answer"],
                "expected": expected_answer,
                "num_sources": len(response.get("sources", [])),
                "retrieval_count": response["metadata"]["documents_retrieved"]
            }
            
            results.append(metrics)
        
        return {
            "test_cases": len(test_queries),
            "results": results,
            "summary": {
                "avg_retrieval_count": sum(r["retrieval_count"] for r in results) / len(results),
                "avg_sources": sum(r["num_sources"] for r in results) / len(results)
            }
        }

# ---------------------------------------------------------------------------
# CONVERSATIONAL RAG 
# ---------------------------------------------------------------------------

class ConversationalRAG(RAGChain):
    """rag with conversation memory."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history = []
        
        # update prompt for conversational context
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided context.
            Use the following pieces of context to answer the question.
            Consider the conversation history when providing your answer.
            If you don't know the answer based on the context, say you don't know.
            
            Context:
            {context}
            
            Conversation History:
            {history}
            """),
            ("human", "{question}")
        ])
        
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def query(
        self,
        question: str,
        k: int = 5,
        return_sources: Optional[bool] = None,
        verbose: Optional[bool] = None
    ) -> Dict[str, Any]:
        # use settings if not provided
        return_sources = return_sources if return_sources is not None else settings.return_sources
        verbose = verbose if verbose is not None else settings.verbose
        """query with conversation history."""
        
        # retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(
            query=question,
            k=k,
            use_mmr=settings.use_mmr
        )
        
        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "metadata": {"documents_retrieved": 0}
            }
        
        # format context and history
        context = self._format_context(retrieved_docs)
        history = self._format_history()
        
        # generate answer
        response = self.chain.invoke({
            "context": context,
            "history": history,
            "question": question
        })
        
        # update conversation history
        self.conversation_history.append({
            "question": question,
            "answer": response["text"]
        })
        
        # keep only last 10 exchanges
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # prepare result
        result = {
            "answer": response["text"],
            "metadata": {
                "documents_retrieved": len(retrieved_docs),
                "model_used": settings.llm_model,
                "conversation_length": len(self.conversation_history)
            }
        }
        
        if return_sources:
            result["sources"] = self._format_sources(retrieved_docs)
        
        if verbose:
            result["retrieved_documents"] = [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                for doc in retrieved_docs
            ]
        
        return result
    
    def _format_history(self) -> str:
        """format conversation history."""
        if not self.conversation_history:
            return "No previous conversation."
        
        history_parts = []
        for exchange in self.conversation_history[-5:]:  # Last 5 exchanges
            history_parts.append(
                f"Q: {exchange['question']}\nA: {exchange['answer']}"
            )
        
        return "\n\n".join(history_parts)
    
    def clear_history(self):
        """clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")