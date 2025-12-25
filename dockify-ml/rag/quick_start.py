"""
Quick start example for testing the RAG system with sample data
"""

from embedder import Embedder
from vector_store import OpenGaussVectorStore
from llm_client import DeepSeekClient
import numpy as np

# Sample medical documents for testing
SAMPLE_DOCS = [
    {
        "id": "doc1",
        "text": "Diabetes is a chronic condition that affects how your body processes blood sugar (glucose). The two main types are Type 1, where the body doesn't produce insulin, and Type 2, where the body doesn't use insulin properly.",
        "source": "medical_knowledge",
        "metadata": {"topic": "diabetes"}
    },
    {
        "id": "doc2",
        "text": "Common symptoms of diabetes include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, blurred vision, and slow-healing sores.",
        "source": "medical_knowledge",
        "metadata": {"topic": "diabetes_symptoms"}
    },
    {
        "id": "doc3",
        "text": "High blood pressure (hypertension) is a condition where the force of blood against artery walls is too high. It can lead to heart disease, stroke, and kidney problems if left untreated.",
        "source": "medical_knowledge",
        "metadata": {"topic": "hypertension"}
    },
]

def test_rag_system():
    """Test the RAG system with sample data"""
    
    print("=" * 70)
    print("RAG System Quick Start Test")
    print("=" * 70)
    
    # 1. Initialize embedder
    print("\n1. Initializing embedder...")
    embedder = Embedder()
    
    # 2. Generate embeddings for sample docs
    print("2. Generating embeddings for sample documents...")
    docs_with_embeddings = embedder.embed_documents(SAMPLE_DOCS)
    
    # 3. Set up vector store
    print("3. Setting up vector store...")
    db_config = {
        "host": "localhost",
        "port": "5432",
        "database": "medical_rag",
        "user": "your_username",
        "password": "your_password"
    }
    
    try:
        vector_store = OpenGaussVectorStore(db_config)
        vector_store.setup_database(embedding_dim=embedder.embedding_dim)
        
        # 4. Insert documents
        print("4. Inserting documents...")
        vector_store.insert_documents(docs_with_embeddings)
        
        # 5. Test retrieval
        print("\n5. Testing retrieval...")
        query = "What are the symptoms of diabetes?"
        print(f"Query: {query}")
        
        query_embedding = embedder.embed_text(query)
        results = vector_store.similarity_search(query_embedding, top_k=2)
        
        print(f"\nFound {len(results)} relevant documents:")
        for i, doc in enumerate(results, 1):
            print(f"\n  Document {i} (Similarity: {doc['similarity']:.3f}):")
            print(f"  {doc['text'][:100]}...")
        
        # 6. Test LLM generation (requires API key)
        print("\n6. Testing LLM generation...")
        print("Note: This requires a valid OPENROUTER_API_KEY")
        
        try:
            from config import OPENROUTER_API_KEY
            if OPENROUTER_API_KEY:
                llm = DeepSeekClient(OPENROUTER_API_KEY)
                answer = llm.generate_answer(query, results)
                print(f"\nGenerated Answer:\n{answer}")
            else:
                print("Skipping LLM test (no API key configured)")
        except Exception as e:
            print(f"Skipping LLM test: {e}")
        
        # Clean up
        vector_store.close()
        
        print("\n" + "=" * 70)
        print("Test completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError during test: {e}")
        print("\nMake sure:")
        print("1. openGauss/PostgreSQL is running")
        print("2. Database credentials are correct in config.py")
        print("3. Vector extension is installed")

if __name__ == "__main__":
    test_rag_system()

