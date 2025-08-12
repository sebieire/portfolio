# ---------------------------------------------------------------------------

import cohere
import numpy as np
from loguru import logger

from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereRerank

from rank_bm25 import BM25Okapi

# local imports
from src.config import settings
from src.vector_store import VectorStoreStrategy
from src.embeddings import EmbeddingStrategy

# ---------------------------------------------------------------------------
# - - = HYDE RETRIEVER = - -
# ---------------------------------------------------------------------------

class HyDERetriever:    
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate_hypothetical_document(self, query: str) -> str:
        
        prompt = f"""Given the following question, write a detailed paragraph that would perfectly answer it.
        Write as if you're writing a documentation or textbook entry.
        
        Question: {query}
        
        Hypothetical Answer:"""
        
        response = self.llm.invoke(prompt)
        return response.content

# ---------------------------------------------------------------------------
# - - = MULTI QUERY RETRIEVER = - -
# ---------------------------------------------------------------------------

class MultiQueryRetriever:
    """ 
    multiple query variations (better recall)
    """
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate_queries(self, original_query: str, num_queries: int = 3) -> List[str]:
        """generate multiple query variations."""
        prompt = f"""Generate {num_queries} different versions of the following question.
        Each version should ask for the same information but with different wording.
        
        Original question: {original_query}
        
        Provide the questions as a numbered list:"""
        
        response = self.llm.invoke(prompt)
        
        # parse the response
        queries = [original_query]
        for line in response.content.split('\n'):
            if line.strip() and any(line.startswith(str(i)) for i in range(1, num_queries + 1)):
                # remove numbering and clean
                query = line.split('.', 1)[-1].strip()
                queries.append(query)
        
        return queries[:num_queries + 1]

# ---------------------------------------------------------------------------
# - - = RERANKER = - -
# ---------------------------------------------------------------------------

class Reranker:
    """
    reranking retrieved documents
    """
    
    def __init__(self, use_cohere: bool = True):
        self.use_cohere = use_cohere and settings.cohere_api_key
        
        if self.use_cohere:
            self.cohere_client = cohere.Client(settings.cohere_api_key)
        else:
            self.llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=0,
                openai_api_key=settings.openai_api_key
            )
    
    def rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 5
    ) -> List[Document]:
        """rerank documents based on relevance to query."""
        
        if not documents:
            return []
        
        if self.use_cohere:
            return self._cohere_rerank(query, documents, top_k)
        else:
            return self._llm_rerank(query, documents, top_k)
    
    def _cohere_rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int
    ) -> List[Document]:
        """use Cohere's reranking api."""
        
        # prepare documents for reranking
        doc_texts = [doc.page_content for doc in documents]
        
        # rerank
        results = self.cohere_client.rerank(
            model=settings.reranker_model,
            query=query,
            documents=doc_texts,
            top_n=min(top_k, len(documents))
        )
        
        # return reranked documents
        reranked_docs = []
        for result in results.results:
            doc = documents[result.index]
            doc.metadata['rerank_score'] = result.relevance_score
            reranked_docs.append(doc)
        
        return reranked_docs
    
    def _llm_rerank(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int
    ) -> List[Document]:
        """use llm for reranking."""
        
        scored_docs = []
        
        for doc in documents:
            prompt = f"""Score the relevance of the following document to the query on a scale of 0-10.
            Only respond with a number.
            
            Query: {query}
            
            Document: {doc.page_content[:500]}
            
            Score:"""
            
            try:
                response = self.llm.invoke(prompt)
                score = float(response.content.strip())
                doc.metadata['rerank_score'] = score / 10.0
                scored_docs.append((score, doc))
            except:
                doc.metadata['rerank_score'] = 0.5
                scored_docs.append((5.0, doc))
        
        # sort by score and return top k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]

# ---------------------------------------------------------------------------
# - - = HYBRID RETRIEVER = - -
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    hybrid retrieval
    """
    
    def __init__(
        self,
        vector_store: VectorStoreStrategy,
        embedding_strategy: EmbeddingStrategy,
        use_hyde: bool = False,
        use_multi_query: bool = True,
        use_reranking: bool = True
    ):
        self.vector_store = vector_store
        self.embedding_strategy = embedding_strategy
        
        # initialize llm
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            openai_api_key=settings.openai_api_key
        )
        
        # initialize components
        self.hyde = HyDERetriever(self.llm) if use_hyde else None
        self.multi_query = MultiQueryRetriever(self.llm) if use_multi_query else None
        self.reranker = Reranker() if use_reranking else None
        
        # bm25 for hybrid search
        self.bm25 = None
        self.corpus = []
    
    def index_for_bm25(self, documents: List[Document]):
        """index documents for bm25 search."""
        self.corpus = documents
        tokenized_docs = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"Indexed {len(documents)} documents for BM25")
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        use_mmr: bool = True,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        main retrieval method  (combines all strategies)
        """
        all_documents = []
        
        # 1. multi-query expansion
        if self.multi_query:
            queries = self.multi_query.generate_queries(query, num_queries=2)
            logger.info(f"Generated {len(queries)} query variations")
        else:
            queries = [query]
        
        # 2. hyde transformation
        if self.hyde:
            hyde_doc = self.hyde.generate_hypothetical_document(queries[0])
            queries.append(hyde_doc)
            logger.info("Generated HyDE document")
        
        # 3. vector search for each query
        vector_docs = []
        for q in queries:
            if use_mmr:
                docs = self.vector_store.max_marginal_relevance_search(
                    query=q,
                    k=k,
                    fetch_k=settings.top_k,
                    lambda_mult=settings.mmr_diversity_score
                )
            else:
                docs = self.vector_store.similarity_search(
                    query=q,
                    k=settings.top_k,
                    filter=filter
                )
            vector_docs.extend(docs)
        
        # remove duplicates
        seen = set()
        unique_vector_docs = []
        for doc in vector_docs:
            doc_id = doc.metadata.get('file_hash', '') + doc.page_content[:100]
            if doc_id not in seen:
                seen.add(doc_id)
                unique_vector_docs.append(doc)
        
        # 4. bm25 search if enabled
        if settings.enable_hybrid_search and self.bm25:
            bm25_docs = self._bm25_search(query, k=settings.top_k)
            
            # combine with vector results
            all_documents = self._combine_results(
                unique_vector_docs,
                bm25_docs,
                vector_weight=1 - settings.bm25_weight,
                bm25_weight=settings.bm25_weight
            )
        else:
            all_documents = unique_vector_docs
        
        # 5. reranking
        if self.reranker and len(all_documents) > k:
            all_documents = self.reranker.rerank(
                query=query,
                documents=all_documents,
                top_k=k
            )
            logger.info(f"Reranked to top {k} documents")
        else:
            all_documents = all_documents[:k]
        
        # 6. filter by similarity threshold
        if hasattr(all_documents[0].metadata, 'score'):
            all_documents = [
                doc for doc in all_documents
                if doc.metadata.get('score', 0) >= settings.similarity_threshold
            ]
        
        logger.info(f"Retrieved {len(all_documents)} final documents")
        return all_documents
    
    def _bm25_search(self, query: str, k: int) -> List[Document]:
        """perform bm25 search."""
        if not self.bm25 or not self.corpus:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # return corresponding documents
        bm25_docs = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.corpus[idx]
                doc.metadata['bm25_score'] = float(scores[idx])
                bm25_docs.append(doc)
        
        return bm25_docs
    
    def _combine_results(
        self,
        vector_docs: List[Document],
        bm25_docs: List[Document],
        vector_weight: float,
        bm25_weight: float
    ) -> List[Document]:
        """combine and score results from vector and bm25 search"""
        
        combined_scores = {}
        all_docs = {}
        
        # score vector results
        for i, doc in enumerate(vector_docs):
            doc_id = id(doc)
            all_docs[doc_id] = doc
            # higher rank = lower score (inverse ranking)
            combined_scores[doc_id] = vector_weight * (1.0 - i / len(vector_docs))
        
        # score bm25 results
        for i, doc in enumerate(bm25_docs):
            doc_id = id(doc)
            if doc_id in combined_scores:
                combined_scores[doc_id] += bm25_weight * (1.0 - i / len(bm25_docs))
            else:
                all_docs[doc_id] = doc
                combined_scores[doc_id] = bm25_weight * (1.0 - i / len(bm25_docs))
        
        # sort by combined score
        sorted_docs = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # return documents with scores
        result_docs = []
        for doc_id, score in sorted_docs:
            doc = all_docs[doc_id]
            doc.metadata['combined_score'] = score
            result_docs.append(doc)
        
        return result_docs