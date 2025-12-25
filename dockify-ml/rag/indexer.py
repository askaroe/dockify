"""
Script to load, embed, and index documents into the vector store
"""

import logging
from data_loader import DataLoader
from embedder import Embedder
from vector_store import OpenGaussVectorStore
from config import DB_CONFIG, EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main indexing pipeline"""
    
    # Step 1: Load data
    logger.info("=" * 50)
    logger.info("Step 1: Loading data")
    logger.info("=" * 50)
    loader = DataLoader()
    documents = loader.load_all_data()
    
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return
    
    # Step 2: Generate embeddings
    logger.info("=" * 50)
    logger.info("Step 2: Generating embeddings")
    logger.info("=" * 50)
    embedder = Embedder(model_name=EMBEDDING_MODEL)
    documents = embedder.embed_documents(documents)
    
    # Step 3: Set up vector store
    logger.info("=" * 50)
    logger.info("Step 3: Setting up vector store")
    logger.info("=" * 50)
    vector_store = OpenGaussVectorStore(DB_CONFIG)
    vector_store.setup_database(embedding_dim=embedder.embedding_dim)
    
    # Step 4: Insert documents
    logger.info("=" * 50)
    logger.info("Step 4: Inserting documents into vector store")
    logger.info("=" * 50)
    vector_store.insert_documents(documents)
    
    # Clean up
    vector_store.close()
    
    logger.info("=" * 50)
    logger.info("Indexing completed successfully!")
    logger.info(f"Total documents indexed: {len(documents)}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

