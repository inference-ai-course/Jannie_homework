import sqlite3
import json
import faiss
import numpy as np
from typing import List, Dict, Any

class HybridDatabase:
    def __init__(self, db_path: str = "hybrid_search.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with FTS5 for hybrid search"""
        conn = sqlite3.connect(self.db_path)
        
        # Create main documents table (extending your existing chunks)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                chunk_id INTEGER PRIMARY KEY,
                content TEXT,
                paper_title TEXT,
                authors TEXT,
                year INTEGER,
                arxiv_id TEXT,
                source_pdf TEXT,
                embedding_id INTEGER
            )
        ''')
        
        # Create FTS5 virtual table for keyword search
        conn.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts 
            USING fts5(content, paper_title, authors, tokenize="porter")
        ''')
        
        conn.commit()
        conn.close()
        print(f"Hybrid database initialized at: {self.db_path}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to both main table and FTS index"""
        conn = sqlite3.connect(self.db_path)
        
        for doc in documents:
            # Insert into main table
            conn.execute('''
                INSERT OR REPLACE INTO documents 
                (chunk_id, content, paper_title, authors, year, arxiv_id, source_pdf, embedding_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc['chunk_id'],
                doc['content'],
                doc.get('paper_title', ''),
                doc.get('authors', ''),
                doc.get('year', ''),
                doc.get('arxiv_id', ''),
                doc.get('source_pdf', ''),
                doc.get('embedding_id', doc['chunk_id'])
            ))
            
            # Insert into FTS5 table
            conn.execute('''
                INSERT OR REPLACE INTO documents_fts 
                (rowid, content, paper_title, authors)
                VALUES (?, ?, ?, ?)
            ''', (
                doc['chunk_id'],
                doc['content'],
                doc.get('paper_title', ''),
                doc.get('authors', '')
            ))
        
        conn.commit()
        conn.close()
        print(f"Added {len(documents)} documents to hybrid database")
    
    def keyword_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform keyword search using FTS5"""
        conn = sqlite3.connect(self.db_path)
        
        results = conn.execute('''
            SELECT d.chunk_id, d.content, d.paper_title, d.authors, 
                   bm25(documents_fts) as score
            FROM documents d
            JOIN documents_fts f ON d.chunk_id = f.rowid
            WHERE documents_fts MATCH ?
            ORDER BY score
            LIMIT ?
        ''', (query, k)).fetchall()
        
        conn.close()
        
        formatted_results = []
        for chunk_id, content, paper_title, authors, score in results:
            similarity = 1.0 / (1.0 + abs(score)) if score < 0 else 1.0
            formatted_results.append({
                'chunk_id': chunk_id,
                'content': content,
                'paper_title': paper_title,
                'authors': authors,
                'keyword_score': similarity
            })
        
        return formatted_results
