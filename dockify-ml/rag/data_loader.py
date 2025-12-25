import pandas as pd
from datasets import load_dataset
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare medical data from various sources"""
    
    def __init__(self):
        self.documents = []
    
    def load_csv_data(self, csv_path: str) -> List[Dict]:
        """
        Load data from CSV file
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            List of document dictionaries
        """
        logger.info(f"Loading CSV data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        documents = []
        for idx, row in df.iterrows():
            # Adjust column names based on your CSV structure
            # For ai_medical_chatbot.csv, typically has 'question' and 'answer' columns
            doc = {
                "id": f"csv_{idx}",
                "text": f"Question: {row.get('question', row.get('Question', ''))}\\n\\nAnswer: {row.get('answer', row.get('Answer', ''))}",
                "source": "ai_medical_chatbot",
                "metadata": row.to_dict()
            }
            documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents from CSV")
        return documents
    
    def load_huggingface_dataset(self, dataset_name: str, split: str = "train") -> List[Dict]:
        """
        Load data from Hugging Face dataset
        
        Args:
            dataset_name: Name of the HF dataset
            split: Dataset split to load
            
        Returns:
            List of document dictionaries
        """
        logger.info(f"Loading HuggingFace dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        
        documents = []
        for idx, item in enumerate(dataset):
            # Adjust based on dataset structure
            # MedText typically has 'text' or similar fields
            text = item.get('text', '') or item.get('content', '') or str(item)
            
            doc = {
                "id": f"hf_{idx}",
                "text": text,
                "source": dataset_name,
                "metadata": item
            }
            documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents from HuggingFace")
        return documents
    
    def load_all_data(self) -> List[Dict]:
        """Load all configured data sources"""
        all_documents = []
        
        # Load CSV data
        try:
            csv_docs = self.load_csv_data('/kaggle/input/ai-medical-chatbot/ai_medical_chatbot.csv')
            all_documents.extend(csv_docs)
        except Exception as e:
            logger.warning(f"Could not load CSV data: {e}")
        
        # Load HuggingFace dataset
        try:
            hf_docs = self.load_huggingface_dataset("BI55/MedText", split="train")
            all_documents.extend(hf_docs)
        except Exception as e:
            logger.warning(f"Could not load HuggingFace data: {e}")
        
        # Optional: Load gym exercise data
        # try:
        #     gym_docs = self.load_csv_data('/kaggle/input/gym-exercise-data/gym_exercise_data.csv')
        #     all_documents.extend(gym_docs)
        # except Exception as e:
        #     logger.warning(f"Could not load gym data: {e}")
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents

