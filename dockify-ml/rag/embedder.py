from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings for text using sentence transformers"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedder
        
        Args:
            model_name: Name of the sentence transformer model
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Embed multiple texts in batches
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings
        """
        logger.info(f"Embedding {len(texts)} texts")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def embed_documents(self, documents: List[dict], text_key: str = "text") -> List[dict]:
        """
        Add embeddings to document dictionaries
        
        Args:
            documents: List of document dictionaries
            text_key: Key in dictionary containing text to embed
            
        Returns:
            Documents with added 'embedding' key
        """
        texts = [doc[text_key] for doc in documents]
        embeddings = self.embed_batch(texts)
        
        for doc, embedding in zip(documents, embeddings):
            doc['embedding'] = embedding
        
        logger.info(f"Added embeddings to {len(documents)} documents")
        return documents

