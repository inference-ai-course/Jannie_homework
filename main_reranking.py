from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from fastapi.middleware.cors import CORSMiddleware
import time

INDEX_PATH = "../embeddings/faiss.index"
ID2TEXT_PATH = "../embeddings/id2text.pkl"
BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

app = FastAPI(title="RAG API with Reranking")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchResult(BaseModel):
    id: str
    score: float
    cross_score: Optional[float] = None
    text: str
    method: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    metrics: dict

class RerankRequest(BaseModel):
    query: str
    candidates: List[str]

class RerankResult(BaseModel):
    text: str
    score: float
    rank: int

class RerankResponse(BaseModel):
    query: str
    reranked_results: List[RerankResult]
    metrics: dict

@app.on_event("startup")
async def startup_event():
    try:
        print("Loading FAISS index...")
        app.index = faiss.read_index(INDEX_PATH)
        
        print("Loading text mappings...")
        with open(ID2TEXT_PATH, "rb") as f:
            app.texts = pickle.load(f)
            
        print("Loading bi-encoder model...")
        app.bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
        
        print("Loading cross-encoder model for reranking...")
        app.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        
        print(f"✅ API ready! {app.index.ntotal} chunks with reranking capability")
        
    except Exception as e:
        print(f"❌ Startup error: {e}")
        import traceback
        traceback.print_exc()
        raise e

@app.post("/search", response_model=SearchResponse)
async def search(query: str, k: int = 3, use_reranking: bool = True):
    start_time = time.time()
    
    try:
        # Stage 1: FAISS retrieval
        query_embedding = app.bi_encoder.encode([query]).astype('float32')
        
        # Get more candidates for reranking
        first_stage_k = 20 if use_reranking else k
        distances, indices = app.index.search(query_embedding, first_stage_k)
        
        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(app.texts):
                candidates.append({
                    'text': app.texts[idx],
                    'faiss_distance': float(dist),
                    'index': idx
                })
        
        if not use_reranking or len(candidates) <= k:
            # Simple FAISS-only results
            results = []
            for i, candidate in enumerate(candidates[:k]):
                results.append(SearchResult(
                    id=f"result_{candidate['index']}",
                    score=candidate['faiss_distance'],
                    text=candidate['text'],
                    method="faiss_only"
                ))
            
            metrics = {
                'method': 'faiss_only',
                'total_time': time.time() - start_time,
                'candidates_considered': len(candidates[:k])
            }
            
        else:
            # Stage 2: Reranking
            rerank_start = time.time()
            candidates_to_rerank = candidates[:10]  # Rerank top 10
            
            # Create pairs for cross-encoder
            pairs = [(query, candidate['text']) for candidate in candidates_to_rerank]
            cross_scores = app.cross_encoder.predict(pairs)
            
            # Combine scores
            for i, candidate in enumerate(candidates_to_rerank):
                candidate['cross_score'] = float(cross_scores[i])
            
            # Sort by cross-encoder score
            candidates_to_rerank.sort(key=lambda x: x['cross_score'], reverse=True)
            
            results = []
            for i, candidate in enumerate(candidates_to_rerank[:k]):
                results.append(SearchResult(
                    id=f"result_{candidate['index']}",
                    score=candidate['faiss_distance'],
                    cross_score=candidate['cross_score'],
                    text=candidate['text'],
                    method="faiss_reranked"
                ))
            
            metrics = {
                'method': 'faiss_reranked',
                'total_time': time.time() - start_time,
                'reranking_time': time.time() - rerank_start,
                'candidates_retrieved': len(candidates),
                'candidates_reranked': len(candidates_to_rerank)
            }
        
        return SearchResponse(query=query, results=results, metrics=metrics)
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/rerank", response_model=RerankResponse)
async def rerank_search(request: RerankRequest):
    """
    Standalone reranking endpoint for existing search results
    """
    start_time = time.time()
    
    try:
        query = request.query
        candidates = request.candidates
        
        if not query or not candidates:
            return RerankResponse(
                query=query,
                reranked_results=[],
                metrics={"error": "Missing query or candidates"}
            )
        
        # Create query-candidate pairs for cross-encoder
        pairs = [(query, candidate) for candidate in candidates]
        
        # Get similarity scores from cross-encoder
        cross_scores = app.cross_encoder.predict(pairs)
        
        # Combine candidates with their scores
        scored_candidates = [
            {"text": text, "score": float(score)}
            for text, score in zip(candidates, cross_scores)
        ]
        
        # Sort by cross-encoder score (descending)
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        
        # Add rank information
        reranked_results = []
        for rank, candidate in enumerate(scored_candidates, 1):
            reranked_results.append(RerankResult(
                text=candidate["text"],
                score=candidate["score"],
                rank=rank
            ))
        
        metrics = {
            'total_time': time.time() - start_time,
            'candidates_reranked': len(candidates),
            'method': 'standalone_reranking'
        }
        
        return RerankResponse(
            query=query,
            reranked_results=reranked_results,
            metrics=metrics
        )
        
    except Exception as e:
        return RerankResponse(
            query=request.query,
            reranked_results=[],
            metrics={"error": str(e)}
        )

@app.get("/")
async def root():
    return {"message": "RAG API with Reranking - Use /search or /rerank endpoints"}

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "index_size": app.index.ntotal,
        "models": {
            "bi_encoder": BI_ENCODER_MODEL,
            "cross_encoder": CROSS_ENCODER_MODEL
        },
        "endpoints": {
            "/search": "Search with optional reranking",
            "/rerank": "Standalone reranking of existing results"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)