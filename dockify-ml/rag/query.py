"""
Script to query the RAG system
"""

import logging
from embedder import Embedder
from vector_store import OpenGaussVectorStore
from llm_client import DeepSeekClient
from config import DB_CONFIG, EMBEDDING_MODEL, OPENROUTER_API_KEY, DEEPSEEK_MODEL, TOP_K_RESULTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """Complete RAG system for querying"""
    
    def __init__(self):
        """Initialize RAG components"""
        logger.info("Initializing RAG system...")
        
        # Initialize embedder
        self.embedder = Embedder(model_name=EMBEDDING_MODEL)
        
        # Initialize vector store
        self.vector_store = OpenGaussVectorStore(DB_CONFIG)
        
        # Initialize LLM client
        self.llm_client = DeepSeekClient(
            api_key=OPENROUTER_API_KEY,
            model=DEEPSEEK_MODEL
        )
        
        logger.info("RAG system initialized successfully")
    
    def query(self, question: str, top_k: int = TOP_K_RESULTS, verbose: bool = True) -> str:
        """
        Query the RAG system
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            verbose: Whether to print retrieval details
            
        Returns:
            Generated answer
        """
        # Step 1: Embed the question
        if verbose:
            logger.info(f"Question: {question}")
            logger.info("Embedding question...")
        
        query_embedding = self.embedder.embed_text(question)
        
        # Step 2: Retrieve relevant documents
        if verbose:
            logger.info(f"Retrieving top {top_k} relevant documents...")
        
        retrieved_docs = self.vector_store.similarity_search(query_embedding, top_k=top_k)
        
        if verbose:
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            for i, doc in enumerate(retrieved_docs, 1):
                logger.info(f"  Doc {i}: Similarity={doc['similarity']:.3f}, Source={doc['source']}")
        
        # Step 3: Generate answer using LLM
        if verbose:
            logger.info("Generating answer with DeepSeek...")
        
        answer = self.llm_client.generate_answer(question, retrieved_docs)
        
        return answer
    
    def close(self):
        """Clean up resources"""
        self.vector_store.close()


def main():
    """Main query interface"""
    
    # Initialize RAG system
    rag = RAGSystem()
    
    print("=" * 70)
    print("Medical RAG System - Powered by DeepSeek")
    print("=" * 70)
    print("Type 'quit' or 'exit' to stop")
    print()
    
    try:
        while True:
            # Get user question
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\n" + "-" * 70)
            
            # Query the system
            answer = rag.query(question, verbose=False)
            
            print("\nAnswer:")
            print(answer)
            print("-" * 70)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    
    finally:
        rag.close()


if __name__ == "__main__":
    main()

