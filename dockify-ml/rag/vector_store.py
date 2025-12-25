import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from typing import List, Dict, Tuple
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenGaussVectorStore:
    """
    Vector store using openGauss (PostgreSQL-compatible) with pgvector extension
    """
    
    def __init__(self, db_config: Dict):
        """
        Initialize connection to openGauss
        
        Args:
            db_config: Database configuration dictionary
        """
        self.db_config = db_config
        self.conn = None
        self.embedding_dim = None
        self._connect()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.conn.autocommit = False
            logger.info("Connected to openGauss database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def setup_database(self, embedding_dim: int):
        """
        Set up database schema with vector extension
        
        Args:
            embedding_dim: Dimension of embedding vectors
        """
        self.embedding_dim = embedding_dim
        
        with self.conn.cursor() as cur:
            # Enable pgvector extension (if openGauss supports it)
            # Note: openGauss may have different vector extension syntax
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except Exception as e:
                logger.warning(f"Could not create vector extension: {e}")
                logger.info("Attempting to use native vector support...")
            
            # Create documents table with vector column
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source TEXT,
                    metadata JSONB,
                    embedding vector({embedding_dim}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create index for faster similarity search
            # Using IVFFlat or HNSW depending on what's available
            try:
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                    ON documents USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
            except Exception as e:
                logger.warning(f"Could not create IVFFlat index: {e}")
                # Try HNSW as fallback
                try:
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS documents_embedding_idx 
                        ON documents USING hnsw (embedding vector_cosine_ops);
                    """)
                except Exception as e2:
                    logger.warning(f"Could not create HNSW index: {e2}")
            
            self.conn.commit()
            logger.info("Database schema created successfully")
    
    def insert_documents(self, documents: List[Dict], batch_size: int = 100):
        """
        Insert documents with embeddings into the database
        
        Args:
            documents: List of document dictionaries with 'embedding' key
            batch_size: Number of documents to insert per batch
        """
        logger.info(f"Inserting {len(documents)} documents")
        
        with self.conn.cursor() as cur:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                values = [
                    (
                        doc['id'],
                        doc['text'],
                        doc.get('source', ''),
                        json.dumps(doc.get('metadata', {})),
                        doc['embedding'].tolist()  # Convert numpy array to list
                    )
                    for doc in batch
                ]
                
                execute_values(
                    cur,
                    """
                    INSERT INTO documents (id, text, source, metadata, embedding)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        text = EXCLUDED.text,
                        source = EXCLUDED.source,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding;
                    """,
                    values
                )
                
                self.conn.commit()
                logger.info(f"Inserted batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        logger.info("All documents inserted successfully")
    
    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar documents using cosine similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        with self.conn.cursor() as cur:
            # Using cosine distance (1 - cosine similarity)
            cur.execute(
                """
                SELECT id, text, source, metadata, 
                       1 - (embedding <=> %s::vector) as similarity
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_embedding.tolist(), query_embedding.tolist(), top_k)
            )
            
            results = []
            for row in cur.fetchall():
                results.append({
                    'id': row[0],
                    'text': row[1],
                    'source': row[2],
                    'metadata': row[3],
                    'similarity': float(row[4])
                })
            
            return results
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

