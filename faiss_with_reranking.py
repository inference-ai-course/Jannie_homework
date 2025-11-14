import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Dict
import time

class EnhancedRetrievalSystem:
    def __init__(self, 
                 bi_encoder_model='all-MiniLM-L6-v2',
                 cross_encoder_model='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Enhanced retrieval with bi-encoder + cross-encoder reranking
        """
        print("🚀 Initializing Enhanced Retrieval System...")
        
        # First-stage: Bi-encoder for fast retrieval
        self.bi_encoder = SentenceTransformer(bi_encoder_model)
        
        # Second-stage: Cross-encoder for precise reranking
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        # Load FAISS index and texts
        self.index = faiss.read_index("embeddings/faiss.index")
        with open("embeddings/id2text.pkl", "rb") as f:
            self.texts = pickle.load(f)
        
        print(f"✅ System ready: {self.index.ntotal} documents loaded")
        print(f"   First-stage: {bi_encoder_model}")
        print(f"   Second-stage: {cross_encoder_model}")
    
    def search(self, 
               query: str, 
               first_stage_k: int = 20,  # Get more candidates initially
               final_k: int = 3,         # Return top N after reranking
               rerank_top_k: int = 10    # Rerank top N from first stage
               ) -> Dict:
        """
        Two-stage search: FAISS retrieval + cross-encoder reranking
        """
        start_time = time.time()
        
        # Stage 1: Fast retrieval with bi-encoder
        stage1_start = time.time()
        query_embedding = self.bi_encoder.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, first_stage_k)
        stage1_time = time.time() - stage1_start
        
        # Get candidate texts
        candidates = []
        candidate_info = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.texts):
                candidates.append(self.texts[idx])
                candidate_info.append({
                    'text': self.texts[idx],
                    'faiss_distance': float(dist),
                    'faiss_rank': len(candidates) + 1
                })
        
        # Stage 2: Rerank top candidates with cross-encoder
        stage2_start = time.time()
        candidates_to_rerank = candidates[:rerank_top_k]
        
        if candidates_to_rerank:
            # Create query-candidate pairs
            pairs = [(query, candidate) for candidate in candidates_to_rerank]
            
            # Get cross-encoder scores
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Combine with original info
            reranked_results = []
            for i, (candidate, cross_score) in enumerate(zip(candidates_to_rerank, cross_scores)):
                original_info = candidate_info[i]
                reranked_results.append({
                    'text': candidate,
                    'cross_encoder_score': float(cross_score),
                    'faiss_distance': original_info['faiss_distance'],
                    'faiss_rank': original_info['faiss_rank'],
                    'final_rank': i + 1
                })
            
            # Sort by cross-encoder score (higher is better)
            reranked_results.sort(key=lambda x: x['cross_encoder_score'], reverse=True)
            final_results = reranked_results[:final_k]
        else:
            final_results = []
        
        stage2_time = time.time() - stage2_start
        total_time = time.time() - start_time
        
        return {
            'query': query,
            'results': final_results,
            'metrics': {
                'total_time': total_time,
                'stage1_time': stage1_time,
                'stage2_time': stage2_time,
                'candidates_retrieved': len(candidates),
                'candidates_reranked': len(candidates_to_rerank),
                'final_results': len(final_results)
            }
        }

def compare_search_methods():
    """Compare standard FAISS vs enhanced reranking"""
    print("🧪 COMPARING SEARCH METHODS")
    print("=" * 50)
    
    # Initialize enhanced system
    enhanced_system = EnhancedRetrievalSystem()
    
    test_queries = [
        "machine learning applications",
        "neural network architecture", 
        "natural language processing techniques",
        "deep learning challenges"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 40)
        
        # Enhanced search with reranking
        enhanced_result = enhanced_system.search(query)
        
        print("🎯 Enhanced Search (with reranking):")
        for i, result in enumerate(enhanced_result['results']):
            improvement = f"(↑{result['faiss_rank'] - (i+1)} ranks)" if result['faiss_rank'] > (i+1) else ""
            print(f"  {i+1}. Cross-score: {result['cross_encoder_score']:.4f} {improvement}")
            print(f"     {result['text'][:100]}...")
        
        metrics = enhanced_result['metrics']
        print(f"⏱️  Timing: FAISS={metrics['stage1_time']:.3f}s, Rerank={metrics['stage2_time']:.3f}s")
        print(f"📊 Stats: {metrics['candidates_retrieved']} → {metrics['candidates_reranked']} → {metrics['final_results']}")

if __name__ == "__main__":
    compare_search_methods()
