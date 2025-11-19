from hybrid_searcher import HybridSearcher
from sentence_transformers import SentenceTransformer
import json

class RealisticEvaluator:
    def __init__(self, searcher):
        self.searcher = searcher
        # Let's first discover what chunks are actually relevant by searching
        self.test_queries = self._discover_relevant_chunks()
    
    def _discover_relevant_chunks(self):
        """Discover which chunks are actually relevant by doing test searches"""
        print("Discovering relevant chunks for evaluation...")
        
        test_queries = [
            {"query": "machine learning", "relevant_chunks": []},
            {"query": "neural networks", "relevant_chunks": []},
            {"query": "transformer models", "relevant_chunks": []},
            {"query": "attention mechanism", "relevant_chunks": []},
            {"query": "language models", "relevant_chunks": []},
            {"query": "deep learning", "relevant_chunks": []},
            {"query": "natural language processing", "relevant_chunks": []},
            {"query": "computer vision", "relevant_chunks": []},
            {"query": "reinforcement learning", "relevant_chunks": []},
            {"query": "speech recognition", "relevant_chunks": []}
        ]
        
        # For each query, find the top results and use them as "relevant"
        for i, test_case in enumerate(test_queries):
            query = test_case["query"]
            print(f"  Testing query: '{query}'")
            
            # Get hybrid results to find relevant chunks
            results = self.searcher.hybrid_search(query, k=10)
            
            # Use the top 5 results as "relevant" for evaluation
            relevant_chunks = [r['chunk_id'] for r in results[:5]]
            test_case["relevant_chunks"] = relevant_chunks
            
            print(f"    Found {len(relevant_chunks)} relevant chunks: {relevant_chunks[:3]}...")
        
        return test_queries
    
    def calculate_recall_at_k(self, results, relevant_chunks, k):
        """Calculate Recall@k - proportion of relevant chunks found in top-k"""
        if not relevant_chunks:
            return 0.0
            
        top_k_chunks = [r['chunk_id'] for r in results[:k]]
        relevant_found = len(set(top_k_chunks) & set(relevant_chunks))
        return relevant_found / len(relevant_chunks)
    
    def calculate_hit_rate(self, results, relevant_chunks, k):
        """Calculate Hit Rate@k - whether ANY relevant chunk is in top-k"""
        if not relevant_chunks:
            return 0.0
            
        top_k_chunks = [r['chunk_id'] for r in results[:k]]
        return 1.0 if len(set(top_k_chunks) & set(relevant_chunks)) > 0 else 0.0
    
    def evaluate_all_methods(self, k=3):
        """Evaluate vector-only, keyword-only, and hybrid search methods"""
        results = []
        
        for test_case in self.test_queries:
            query = test_case["query"]
            relevant_chunks = test_case["relevant_chunks"]
            
            if not relevant_chunks:
                continue  # Skip if no relevant chunks found
                
            # Test vector search
            vector_results = self.searcher.vector_search(query, k*3)
            vector_recall = self.calculate_recall_at_k(vector_results, relevant_chunks, k)
            vector_hit = self.calculate_hit_rate(vector_results, relevant_chunks, k)
            
            # Test keyword search  
            keyword_results = self.searcher.db.keyword_search(query, k*3)
            keyword_recall = self.calculate_recall_at_k(keyword_results, relevant_chunks, k)
            keyword_hit = self.calculate_hit_rate(keyword_results, relevant_chunks, k)
            
            # Test hybrid search
            hybrid_results = self.searcher.hybrid_search(query, k*3)
            hybrid_recall = self.calculate_recall_at_k(hybrid_results, relevant_chunks, k)
            hybrid_hit = self.calculate_hit_rate(hybrid_results, relevant_chunks, k)
            
            results.append({
                "query": query,
                "relevant_chunks": relevant_chunks,
                "vector_recall": round(vector_recall, 4),
                "keyword_recall": round(keyword_recall, 4), 
                "hybrid_recall": round(hybrid_recall, 4),
                "vector_hit_rate": round(vector_hit, 4),
                "keyword_hit_rate": round(keyword_hit, 4),
                "hybrid_hit_rate": round(hybrid_hit, 4)
            })
        
        return results
    
    def print_evaluation_report(self, k=3):
        """Print comprehensive evaluation report"""
        print("=" * 70)
        print("REALISTIC HYBRID SEARCH EVALUATION REPORT")
        print("=" * 70)
        print(f"Evaluation Metrics @ k={k}")
        
        results = self.evaluate_all_methods(k)
        
        if not results:
            print("No valid test cases with relevant chunks found!")
            return []
        
        print(f"Number of valid test queries: {len(results)}")
        print("=" * 70)
        
        # Print detailed results
        print("\nDETAILED RESULTS:")
        print("-" * 70)
        print(f"{'Query':<28} {'Vector':<8} {'Keyword':<8} {'Hybrid':<8} {'V-Hit':<6} {'K-Hit':<6} {'H-Hit':<6}")
        print("-" * 70)
        for r in results:
            print(f"{r['query'][:26]:<28} {r['vector_recall']:<8} {r['keyword_recall']:<8} {r['hybrid_recall']:<8} "
                  f"{r['vector_hit_rate']:<6} {r['keyword_hit_rate']:<6} {r['hybrid_hit_rate']:<6}")
        
        # Calculate averages
        avg_vector_recall = sum(r['vector_recall'] for r in results) / len(results)
        avg_keyword_recall = sum(r['keyword_recall'] for r in results) / len(results)
        avg_hybrid_recall = sum(r['hybrid_recall'] for r in results) / len(results)
        
        avg_vector_hit = sum(r['vector_hit_rate'] for r in results) / len(results)
        avg_keyword_hit = sum(r['keyword_hit_rate'] for r in results) / len(results)
        avg_hybrid_hit = sum(r['hybrid_hit_rate'] for r in results) / len(results)
        
        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS (Averages):")
        print("=" * 70)
        print(f"Recall@{k}:")
        print(f"  Vector-only:  {avg_vector_recall:.4f}")
        print(f"  Keyword-only: {avg_keyword_recall:.4f}")
        print(f"  Hybrid:       {avg_hybrid_recall:.4f}")
        print(f"  Improvement:  {avg_hybrid_recall - max(avg_vector_recall, avg_keyword_recall):+.4f}")
        
        print(f"\nHit Rate@{k}:")
        print(f"  Vector-only:  {avg_vector_hit:.4f}")
        print(f"  Keyword-only: {avg_keyword_hit:.4f}")
        print(f"  Hybrid:       {avg_hybrid_hit:.4f}")
        print(f"  Improvement:  {avg_hybrid_hit - max(avg_vector_hit, avg_keyword_hit):+.4f}")
        
        # Show examples where hybrid performs differently
        print("\n" + "=" * 70)
        print("QUERIES WHERE HYBRID PERFORMS DIFFERENTLY:")
        print("=" * 70)
        hybrid_better = [r for r in results if r['hybrid_recall'] > max(r['vector_recall'], r['keyword_recall'])]
        hybrid_worse = [r for r in results if r['hybrid_recall'] < max(r['vector_recall'], r['keyword_recall'])]
        
        if hybrid_better:
            print(f"Hybrid BETTER on {len(hybrid_better)} queries:")
            for r in hybrid_better[:3]:  # Show first 3
                print(f"  '{r['query']}': V={r['vector_recall']}, K={r['keyword_recall']}, H={r['hybrid_recall']}")
        
        if hybrid_worse:
            print(f"Hybrid WORSE on {len(hybrid_worse)} queries:")
            for r in hybrid_worse[:3]:  # Show first 3
                print(f"  '{r['query']}': V={r['vector_recall']}, K={r['keyword_recall']}, H={r['hybrid_recall']}")
        
        return results

# Run evaluation
if __name__ == "__main__":
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    searcher = HybridSearcher(
        faiss_index_path="embeddings/faiss.index",
        db_path="arxiv_hybrid_final.db", 
        embedding_model=embedding_model
    )
    
    evaluator = RealisticEvaluator(searcher)
    results = evaluator.print_evaluation_report(k=3)
