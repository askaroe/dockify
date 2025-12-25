from openai import OpenAI
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSeekClient:
    """
    Client for DeepSeek model via OpenRouter API
    """
    
    def __init__(self, api_key: str, model: str = "deepseek/deepseek-chat"):
        """
        Initialize DeepSeek client
        
        Args:
            api_key: OpenRouter API key
            model: Model name on OpenRouter
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.model = model
        logger.info(f"Initialized DeepSeek client with model: {model}")
    
    def generate_answer(self, query: str, context_docs: List[Dict], 
                       system_prompt: str = None) -> str:
        """
        Generate answer using retrieved context
        
        Args:
            query: User question
            context_docs: List of retrieved documents
            system_prompt: Optional system prompt
            
        Returns:
            Generated answer
        """
        # Build context from retrieved documents
        context = self._build_context(context_docs)
        
        # Default system prompt for medical RAG
        if system_prompt is None:
            system_prompt = """You are a helpful medical assistant. Answer questions based on the provided context.
If the context doesn't contain relevant information, say so clearly.
Always prioritize accuracy and mention if you're uncertain."""
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Context information:
{context}

Question: {query}

Please provide a helpful answer based on the context above."""}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            logger.info("Successfully generated answer")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def _build_context(self, docs: List[Dict]) -> str:
        """
        Build context string from retrieved documents
        
        Args:
            docs: List of document dictionaries
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Document {i}]")
            context_parts.append(doc['text'])
            context_parts.append(f"(Source: {doc.get('source', 'unknown')}, Similarity: {doc.get('similarity', 0):.3f})")
            context_parts.append("")
        
        return "\n".join(context_parts)

