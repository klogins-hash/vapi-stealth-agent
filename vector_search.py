"""
Vector Search Agent
Uses Cohere v4 embed model for semantic search and document retrieval
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
import numpy as np
import cohere

app = Flask(__name__)

class VectorSearchAgent:
    """
    Vector search agent using Cohere v4 embed model
    """
    
    def __init__(self):
        self.cohere_api_key = os.environ.get('COHERE_API_KEY')
        self.db_orchestrator_url = os.environ.get('DB_ORCHESTRATOR_URL', 'http://database-orchestrator:8000')
        
        if not self.cohere_api_key:
            print("⚠️ COHERE_API_KEY not found in environment")
            self.cohere_client = None
        else:
            try:
                self.cohere_client = cohere.Client(self.cohere_api_key)
                print("✅ Cohere client initialized with v4 embed model")
            except Exception as e:
                print(f"❌ Failed to initialize Cohere client: {e}")
                self.cohere_client = None
        
        self.vector_store = {}  # In-memory store for demo
        self.document_metadata = {}
        
    async def embed_text(self, text: str, input_type: str = "search_document") -> List[float]:
        """
        Generate embeddings using Cohere v4 embed model
        """
        if not self.cohere_client:
            raise Exception("Cohere client not initialized")
        
        try:
            response = self.cohere_client.embed(
                texts=[text],
                model="embed-english-v3.0",  # Latest available model
                input_type=input_type,  # search_document, search_query, classification, clustering
                embedding_types=["float"]
            )
            
            return response.embeddings.float[0]
            
        except Exception as e:
            print(f"❌ Embedding failed: {e}")
            raise e
    
    async def add_document(self, doc_id: str, content: str, metadata: Dict = None) -> Dict:
        """
        Add document to vector store
        """
        try:
            # Generate embedding
            embedding = await self.embed_text(content, "search_document")
            
            # Store in vector store
            self.vector_store[doc_id] = {
                'embedding': embedding,
                'content': content,
                'created_at': datetime.now().isoformat()
            }
            
            # Store metadata
            if metadata:
                self.document_metadata[doc_id] = metadata
            
            # Optionally persist to database via orchestrator
            await self._persist_to_db(doc_id, content, embedding, metadata)
            
            return {
                'doc_id': doc_id,
                'status': 'added',
                'embedding_dimension': len(embedding),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'doc_id': doc_id,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def search_similar(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Dict]:
        """
        Search for similar documents using vector similarity
        """
        try:
            # Generate query embedding
            query_embedding = await self.embed_text(query, "search_query")
            
            # Calculate similarities
            similarities = []
            for doc_id, doc_data in self.vector_store.items():
                doc_embedding = doc_data['embedding']
                
                # Cosine similarity
                similarity = self._cosine_similarity(query_embedding, doc_embedding)
                
                if similarity >= threshold:
                    similarities.append({
                        'doc_id': doc_id,
                        'similarity': similarity,
                        'content': doc_data['content'][:200] + '...' if len(doc_data['content']) > 200 else doc_data['content'],
                        'metadata': self.document_metadata.get(doc_id, {}),
                        'created_at': doc_data['created_at']
                    })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            return {
                'query': query,
                'results': similarities[:top_k],
                'total_matches': len(similarities),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'query': query,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def _persist_to_db(self, doc_id: str, content: str, embedding: List[float], metadata: Dict = None):
        """
        Persist document and embedding to database via orchestrator
        """
        try:
            # Create vector table if needed
            create_table_query = """
            CREATE TABLE IF NOT EXISTS vector_documents (
                id VARCHAR(255) PRIMARY KEY,
                content TEXT,
                embedding FLOAT8[],
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            async with aiohttp.ClientSession() as session:
                # Create table
                create_payload = {
                    'database': 'postgresql',
                    'query': create_table_query
                }
                
                async with session.post(f"{self.db_orchestrator_url}/query", json=create_payload) as response:
                    create_result = await response.json()
                
                # Insert document
                insert_query = """
                INSERT INTO vector_documents (id, content, embedding, metadata) 
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET 
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    created_at = CURRENT_TIMESTAMP
                """
                
                insert_payload = {
                    'database': 'postgresql',
                    'query': insert_query,
                    'params': [doc_id, content, embedding, json.dumps(metadata or {})]
                }
                
                async with session.post(f"{self.db_orchestrator_url}/query", json=insert_payload) as response:
                    insert_result = await response.json()
                    
        except Exception as e:
            print(f"⚠️ Failed to persist to database: {e}")
    
    def get_stats(self) -> Dict:
        """
        Get vector store statistics
        """
        return {
            'total_documents': len(self.vector_store),
            'cohere_client_status': 'connected' if self.cohere_client else 'disconnected',
            'embedding_model': 'embed-english-v3.0',
            'db_orchestrator_url': self.db_orchestrator_url,
            'last_updated': datetime.now().isoformat()
        }

# Initialize vector search agent
vector_agent = VectorSearchAgent()

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        'service': 'Vector Search Agent',
        'status': 'operational',
        'version': '1.0.0',
        'embedding_model': 'Cohere embed-english-v3.0',
        'endpoints': [
            '/health',
            '/embed',
            '/add-document',
            '/search',
            '/stats'
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'cohere_client': 'connected' if vector_agent.cohere_client else 'disconnected',
        'documents_stored': len(vector_agent.vector_store),
        'db_orchestrator': vector_agent.db_orchestrator_url,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/embed', methods=['POST'])
def embed_text():
    """Generate embeddings for text"""
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    input_type = data.get('input_type', 'search_document')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        embedding = loop.run_until_complete(vector_agent.embed_text(text, input_type))
        return jsonify({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'embedding_dimension': len(embedding),
            'input_type': input_type,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        loop.close()

@app.route('/add-document', methods=['POST'])
def add_document():
    """Add document to vector store"""
    data = request.get_json()
    
    if not data or 'content' not in data:
        return jsonify({'error': 'No content provided'}), 400
    
    doc_id = data.get('doc_id', f"doc_{int(datetime.now().timestamp())}")
    content = data['content']
    metadata = data.get('metadata', {})
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(vector_agent.add_document(doc_id, content, metadata))
        return jsonify(result)
    finally:
        loop.close()

@app.route('/search', methods=['POST'])
def search():
    """Search for similar documents"""
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400
    
    query = data['query']
    top_k = data.get('top_k', 5)
    threshold = data.get('threshold', 0.7)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        results = loop.run_until_complete(vector_agent.search_similar(query, top_k, threshold))
        return jsonify(results)
    finally:
        loop.close()

@app.route('/stats', methods=['GET'])
def stats():
    """Get vector store statistics"""
    return jsonify(vector_agent.get_stats())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8002))
    app.run(host='0.0.0.0', port=port, debug=False)
