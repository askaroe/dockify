# Dockify - AI-Powered Health & Medical Intelligence System

An integrated AI system combining classical machine learning, advanced LLM fine-tuning, and retrieval-augmented generation for comprehensive health analysis and medical question answering.

## Technology Stack

**Huawei Technologies:**
- **MindSpore** - Deep learning framework for ML model training
- **openGauss** - Enterprise-grade vector database for RAG system

**AI & ML:**
- Classical ML models for health prediction
- DeepSeek LLM with LoRA fine-tuning
- Sentence Transformers for embeddings
- RAG (Retrieval-Augmented Generation) architecture

## Project Structure

```
ICT/
├── data/              # Raw and processed datasets
├── models/            # Trained MindSpore models and prediction scripts
├── notebooks/         # Training and preprocessing notebooks
│   ├── sleep_train.ipynb
│   └── deepseek_lora_finetuning.ipynb
└── rag/               # Medical RAG system with openGauss
    ├── indexer.py     # Load and index medical data
    ├── query.py       # Query interface with DeepSeek
    └── ...
```

---

## 1. Classical ML Models (MindSpore)

Five specialized health prediction models trained using **MindSpore**:

| Model | Type | Purpose | Output |
|-------|------|---------|--------|
| **1A: Sleep Quality Scorer** | Regression | Predicts overall sleep quality | Score 0-100 |
| **1B: Sleep Stage Classifier** | Classification | Identifies sleep stage | deep/light/rem/restless |
| **2A: Lifestyle Classifier** | Classification | Categorizes lifestyle | sedentary/active/athletic |
| **2B: Activity Predictor** | Regression | Predicts next-day calories | Calories burned |
| **2C: Health Risk Scorer** | Regression | Assesses health risk | Risk score 0-100 |

**Quick Start:**
```bash
cd models
python main.py  # Run predictions for all 5 models
```

### Key Features
- Built with MindSpore for optimal performance
- Z-score normalized inputs for consistency
- Handles missing values automatically (uses training means)
- Trained on real-world sleep and lifestyle data

**Input Features Include:**
- Sleep metrics (duration, efficiency, regularity)
- Physiological data (heart rate, body composition)
- Lifestyle factors (workout frequency, nutrition, stress)
- Temporal patterns (day of week, time of day)

---

## 2. DeepSeek LoRA Fine-tuning

Advanced language model fine-tuning for medical and health domain expertise.

### Overview
- **Base Model:** DeepSeek-R1-Distill-Qwen-1.5B (1.5 billion parameters)
- **Technique:** LoRA (Low-Rank Adaptation) - parameter-efficient fine-tuning
- **Framework:** MindNLP + MindSpore
- **Dataset:** AI Medical Chatbot + MedText dataset

### Why LoRA Fine-tuning?

**Parameter Efficiency:**
- Trains only ~0.1% of model parameters (16.7M out of 1.5B)
- 4-bit quantization reduces memory footprint by 75%
- Enables training large models on consumer hardware

**Domain Adaptation:**
- Specializes the model for medical/health queries
- Learns medical terminology and reasoning patterns
- Improves accuracy for health-related questions

**Cost-Effective:**
- Faster training (3 epochs vs dozens for full fine-tuning)
- Lower compute requirements (GPU-friendly)
- Easy to update with new medical knowledge

### Technical Details

**LoRA Configuration:**
```python
- Rank (r): 16
- Alpha: 32
- Target modules: All attention and MLP layers
- Dropout: 0.05
- Quantization: 4-bit with NF4
```

**Training:**
- 1,000 medical Q&A examples
- 3 epochs with cosine learning rate schedule
- Batch size: 16 (4 × 4 gradient accumulation)
- Max sequence length: 512 tokens

**Notebook:** `notebooks/deepseek_lora_finetuning.ipynb`

---

## 3. Medical RAG System (openGauss + DeepSeek)

A production-ready Retrieval-Augmented Generation system for medical question answering, leveraging **Huawei's openGauss** database.

### Architecture

```
User Question → Embeddings → Vector Search (openGauss) → Context Retrieval → DeepSeek LLM → Answer
```

### Why openGauss?

**Huawei's Enterprise Vector Database** - PostgreSQL-compatible with native vector support, ACID compliance, and high-performance indexing (IVFFlat, HNSW). Combines structured medical records with semantic embeddings in one database, enabling hybrid SQL + similarity queries. Enterprise-grade security, transactional integrity, and cost-effective (no separate vector DB service needed).

### Components

**Data Loader** - Loads medical Q&A datasets (CSV, Hugging Face: AI Medical Chatbot, BI55/MedText)  
**Embedder** - Sentence Transformers with configurable models (MiniLM, MPNet, BGE)  
**Vector Store** - openGauss interface with pgvector extension, cosine similarity search  
**LLM Client** - DeepSeek via OpenRouter API for context-aware answer generation

**Features:** Hybrid search (semantic + SQL), real-time updates, scalable to millions of documents, production-ready with error handling and logging.

### Quick Start

Install openGauss from Huawei

`cd rag && cp .env`

**3. Index & Query:**
```bash
python indexer.py  # Load, embed, and store medical documents
python query.py    # Interactive Q&A interface
```

**Data Sources:** AI Medical Chatbot CSV, BI55/MedText (Hugging Face)
**Performance:** <100ms retrieval, ~2-3s end-to-end, scales to 100K+ documents
**Full documentation:** `rag/README.md`

---

## Huawei Technology Integration

This project showcases **Huawei's enterprise AI/ML ecosystem**:

### MindSpore (Deep Learning)
- Trains all 5 classical ML models
- GPU-accelerated training with automatic differentiation
- Production deployment with `.ckpt` checkpoints
- Native Python API for rapid development

### openGauss (Vector Database)
- Stores medical embeddings and metadata
- PostgreSQL-compatible for easy migration
- Vector similarity search with multiple index types
- Supports hybrid SQL + semantic queries
- Enterprise features: ACID, HA, security

**Benefits:**
- **Unified Stack:** Single vendor for ML training + vector DB
- **Performance:** Optimized for Huawei hardware
- **Support:** Enterprise-grade documentation and community
- **Open Source:** Both MindSpore and openGauss are open source

---

## Quick Start Guide

**Classical ML Models:** `cd models && python main.py`  
**Fine-tune DeepSeek:** Open `notebooks/deepseek_lora_finetuning.ipynb`  
**RAG System:** `cd rag && pip install -r requirements.txt && python indexer.py && python query.py`

---

## Use Cases

**Healthcare:** Predict patient health risks, answer medical questions with evidence-based responses, analyze lifestyle factors  
**Fitness & Wellness:** Classify activity levels, predict workout effectiveness, personalized recommendations  
**Medical Research:** Query medical literature, extract data insights, validate hypotheses with ML predictions
---

## Technical Reference

### Feature Inputs (Z-score Normalized)

**Sleep Models (1A, 1B):**
Time metrics (bed hours, sleep duration, sleep onset), quality metrics (efficiency, regularity, movements, snoring), physiological (heart rate), temporal (day, month, hour), mood & lifestyle notes (coffee, tea, workout, stress), binary flags (weekend, snoring)

**Lifestyle Models (2A, 2B, 2C):**
Physical metrics (age, weight, height, BMI, body composition), heart rate (max, avg, resting), workout metrics (duration, frequency, intensity, calories), nutrition (carbs, protein, fats, water, caloric balance), categorical (experience level, meal frequency)

**Usage:** Set features to `None` to use training means. Scaled features typically range -3.0 to 3.0. Run `python models/feature_means.py` for reference values.

### RAG System Configuration

- **Embedding:** sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Vector Index:** IVFFlat or HNSW for cosine similarity search
- **Retrieval:** Top-K documents (default K=5, <100ms retrieval time)
- **LLM:** DeepSeek-R1-Distill-Qwen-1.5B via OpenRouter

---

## Contributing

This project showcases Huawei MindSpore and openGauss integration with modern LLM techniques (LoRA fine-tuning, RAG). Contributions welcome for additional datasets, model improvements, and new use cases.
