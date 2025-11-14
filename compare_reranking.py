import requests
import json
import time

def test_reranking_comparison():
    """Compare search results with and without reranking"""
    
    test_queries = [
        "machine learning applications in healthcare",
        "transformer model architecture details",
        "neural network training optimization methods",
        "natural language processing recent advances"
    ]
    
    print("🔬 RERANKING PERFORMANCE COMPARISON")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)
        
        # Test without reranking
        start_time = time.time()
        response_no_rerank = requests.post(
            "http://localhost:8000/search",
            params={"query": query, "k": 3, "use_reranking": False}
        )
        time_no_rerank = time.time() - start_time
        
        # Test with reranking
        start_time = time.time()
        response_rerank = requests.post(
            "http://localhost:8000/search", 
            params={"query": query, "k": 3, "use_reranking": True}
        )
        time_rerank = time.time() - start_time
        
        if response_no_rerank.status_code == 200 and response_rerank.status_code == 200:
            no_rerank_data = response_no_rerank.json()
            rerank_data = response_rerank.json()
            
            print("📋 Without Reranking:")
            for i, result in enumerate(no_rerank_data['results']):
                print(f"  {i+1}. (FAISS: {result['score']:.4f}) {result['text'][:80]}...")
            
            print("🎯 With Reranking:")
            for i, result in enumerate(rerank_data['results']):
                cross_info = f", Cross: {result['cross_score']:.4f}" if result.get('cross_score') else ""
                print(f"  {i+1}. (FAISS: {result['score']:.4f}{cross_info}) {result['text'][:80]}...")
            
            print(f"⏱️  Timing: No rerank={time_no_rerank:.3f}s, With rerank={time_rerank:.3f}s")
            
            # Show ranking changes
            if len(no_rerank_data['results']) == len(rerank_data['results']):
                changes = []
                for i, (no_r, with_r) in enumerate(zip(no_rerank_data['results'], rerank_data['results'])):
                    if no_r['text'] != with_r['text']:
                        changes.append(f"Position {i+1} changed")
                if changes:
                    print(f"🔄 Ranking changes: {', '.join(changes)}")
        
        else:
            print("❌ API error - make sure the server is running")

if __name__ == "__main__":
    # Start the reranking API first, then run this
    print("Make sure to start the API first:")
    print("cd app && python main_reranking.py")
    print("\nThen run this script to compare results.")
    
    # Uncomment to auto-test if API is running
    # test_reranking_comparison()
