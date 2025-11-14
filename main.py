from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os

# Use correct relative paths - go up one level from app/ to project root
INDEX_PATH = "../embeddings/faiss.index"
ID2TEXT_PATH = "../embeddings/id2text.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

app = FastAPI(title="arXiv RAG API")

# Load models at startup
@app.on_event("startup")
async def startup_event():
    try:
        print(f"Loading FAISS index from: {INDEX_PATH}")
        print(f"File exists: {os.path.exists(INDEX_PATH)}")
        
        app.index = faiss.read_index(INDEX_PATH)
        
        print(f"Loading text mappings from: {ID2TEXT_PATH}")
        print(f"File exists: {os.path.exists(ID2TEXT_PATH)}")
        
        with open(ID2TEXT_PATH, "rb") as f:
            app.id2text = pickle.load(f)
            
        print("Loading embedding model...")
        app.model = SentenceTransformer(MODEL_NAME)
        
        print(f"✅ API ready! Index size: {app.index.ntotal} chunks")
        
    except Exception as e:
        print(f"❌ Startup error: {e}")
        import traceback
        traceback.print_exc()
        raise e

class SearchResult(BaseModel):
    id: str
    score: float
    text: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

@app.get("/")
async def root():
    return {"message": "arXiv RAG API - Use /search endpoint"}

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "index_size": app.index.ntotal,
        "model": MODEL_NAME
    }

@app.post("/search", response_model=SearchResponse)
async def search(query: str, k: int = 3):
    try:
        # Encode query
        query_embedding = app.model.encode([query]).astype('float32')
        
        # Search
        distances, indices = app.index.search(query_embedding, k)
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(app.id2text):
                results.append(SearchResult(
                    id=f"result_{idx}",
                    score=float(dist),
                    text=app.id2text[idx]
                ))
        
        return SearchResponse(query=query, results=results)
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
