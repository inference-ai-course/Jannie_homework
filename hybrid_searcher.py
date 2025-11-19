import sqlite3
import faiss
import numpy as np
from typing import List, Dict, Any
from hybrid_database import HybridDatabase

class HybridSearcher:
    def __init__(self, faiss_index_path: str, db_path: str, embedding_model):
        self.faiss_index = faiss.read_index(faiss_index_path)
        self.db = HybridDatabase(db_path)
        self.embedding_model = embedding_model
    
    def vector_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Your existing FAISS vector search"""
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:
                similarity = 1.0 / (1.0 + distance)
                results.append({
                    'chunk_id': int(idx),
                    'vector_score': float(similarity),
                    'rank': i
                })
        return results
    
    def hybrid_search(self, query: str, k: int = 5, method: str = "rrf") -> List[Dict[str, Any]]:
        """Combine vector and keyword search"""
        vector_results = self.vector_search(query, k * 2)
        keyword_results = self.db.keyword_search(query, k * 2)
        
        fused_scores = {}
        
        for rank, result in enumerate(vector_results):
            chunk_id = result['chunk_id']
            fused_scores[chunk_id] = 1.0 / (60 + rank + 1)
        
        for rank, result in enumerate(keyword_results):
            chunk_id = result['chunk_id']
            if chunk_id in fused_scores:
                fused_scores[chunk_id] += 1.0 / (60 + rank + 1)
            else:
                fused_scores[chunk_id] = 1.0 / (60 + rank + 1)
        
        top_chunk_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Get full document details
        results = self._get_document_details([chunk_id for chunk_id, score in top_chunk_ids])
        
        # Add hybrid scores to results
        for result in results:
            chunk_id = result['chunk_id']
            result['hybrid_score'] = fused_scores.get(chunk_id, 0.0)
        
        return results
    
    def _get_document_details(self, chunk_ids: List[int]) -> List[Dict[str, Any]]:
        """Retrieve full document details from database"""
        if not chunk_ids:
            return []
        
        conn = sqlite3.connect(self.db.db_path)
        placeholders = ','.join('?' * len(chunk_ids))
        
        results = conn.execute(f'''
            SELECT chunk_id, content, paper_title, authors, year, arxiv_id, source_pdf
            FROM documents
            WHERE chunk_id IN ({placeholders})
        ''', chunk_ids).fetchall()
        
        conn.close()
        
        documents = []
        for chunk_id, content, paper_title, authors, year, arxiv_id, source_pdf in results:
            documents.append({
                'chunk_id': chunk_id,
                'content': content,
                'paper_title': paper_title,
                'authors': authors,
                'year': year,
                'arxiv_id': arxiv_id,
                'source_pdf': source_pdf
            })
        
        return documents
    
    def compare_searches(self, query: str, k: int = 3):
        """Compare vector-only vs keyword-only vs hybrid search"""
        vector_results = self.vector_search(query, k)
        keyword_results = self.db.keyword_search(query, k)
        hybrid_results = self.hybrid_search(query, k)
        
        return {
            'query': query,
            'vector_only': [r['content'][:200] + '...' for r in vector_results],
            'keyword_only': [r['content'][:200] + '...' for r in keyword_results],
            'hybrid': [r['content'][:200] + '...' for r in hybrid_results]
        }
