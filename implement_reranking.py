from sentence_transformers import CrossEncoder
import json
import numpy as np
from tqdm import tqdm

class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initialize cross-encoder model for reranking
        Common models:
        - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good balance)
        - cross-encoder/ms-marco-MiniLM-L-12-v2 (slower, better)
        - cross-encoder/ms-marco-electra-base (even better but slower)
        """
        print(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
    
    def rerank(self, query: str, candidates: list, top_k: int = 5):
        """
        Rerank candidates using cross-encoder
        Returns: sorted list of (candidate, score, original_rank)
        """
        if not candidates:
            return []
        
        # Create query-candidate pairs for cross-encoder
        pairs = [(query, candidate) for candidate in candidates]
        
        # Get scores from cross-encoder
        scores = self.model.predict(pairs)
        
        # Combine candidates with their scores and original ranks
        ranked_results = []
        for i, (candidate, score) in enumerate(zip(candidates, scores)):
            ranked_results.append({
                'text': candidate,
                'rerank_score': float(score),
                'original_rank': i + 1
            })
        
        # Sort by rerank score (descending - higher is better)
        ranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        return ranked_results[:top_k]

def test_reranking():
    """Test the reranking functionality"""
    reranker = Reranker()
    
    # Test query and candidates
    query = "What is machine learning?"
    candidates = [
        "Deep learning is a subset of machine learning that uses neural networks.",
        "Machine learning is a field of artificial intelligence that enables computers to learn without being explicitly programmed.",
        "Natural language processing is a subfield of linguistics and computer science.",
        "Machine learning algorithms build models based on sample data to make predictions.",
        "Computer vision deals with how computers can understand visual information."
    ]
    
    print("🔍 Testing Reranking:")
    print(f"Query: {query}")
    print(f"\nOriginal candidates:")
    for i, candidate in enumerate(candidates):
        print(f"  {i+1}. {candidate}")
    
    # Rerank
    reranked = reranker.rerank(query, candidates, top_k=3)
    
    print(f"\n🎯 After reranking:")
    for i, result in enumerate(reranked):
        print(f"  {i+1}. (Score: {result['rerank_score']:.4f}, Was: #{result['original_rank']})")
        print(f"     {result['text']}")

if __name__ == "__main__":
    test_reranking()
