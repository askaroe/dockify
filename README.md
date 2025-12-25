# Dockify - AI-Powered Health & Wellness Platform

A comprehensive health tracking and AI-powered medical intelligence platform built with a microservices architecture. Dockify integrates mobile health data with machine learning models and LLM-powered recommendations to provide personalized health insights.

---

## Project Architecture

The project consists of three main components:

```
dockify/
├── dockify-frontend/    # Kotlin Multiplatform mobile app (Android & iOS)
├── dockify-backend/     # Go REST API server
└── dockify-ml/          # Python ML/AI services (MindSpore, RAG, DeepSeek)
```

---

## Frontend (dockify-frontend)

A **Kotlin Multiplatform** mobile application targeting Android and iOS platforms with Compose Multiplatform UI.

### Tech Stack
| Technology | Purpose |
|------------|---------|
| Jetpack Compose | Modern declarative UI framework |
| Material Design 3 | Design system with dynamic theming |
| Ktor | HTTP client for API communication |
| Koin | Dependency injection |
| Kotlin Coroutines & Flow | Asynchronous processing |
| DataStore | Encrypted local storage |

### Features
- **Health Connect / HealthKit Integration** - Reads data from device health APIs
- **Authentication** - Email/password login with JWT tokens
- **Health Metrics Sync** - Upload health data to backend
- **AI Recommendations** - Display personalized health insights
- **Location Services** - Find nearby hospitals and users

### Build & Run

**Android:**
```shell
# macOS/Linux
./gradlew :composeApp:assembleDebug

# Windows
.\gradlew.bat :composeApp:assembleDebug
```

**iOS:**

Open `iosApp/` directory in Xcode and run the project.

---

## Backend (dockify-backend)

A **Go** REST API server following clean architecture principles with Gin framework.

### Tech Stack
| Technology | Purpose |
|------------|---------|
| Go 1.24+ | Programming language |
| Gin | HTTP web framework |
| OpenGauss (pgx) | Database |
| Swagger | API documentation |
| Logrus | Structured logging |

### Project Structure
```
internal/
├── entity/      # Domain entities
├── handlers/    # HTTP handlers
├── repository/  # Data access layer
├── services/    # Business logic
├── router/      # Route definitions
├── server/      # Server configuration
└── gateway/     # External service integrations
```

### API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/register` | User registration |
| POST | `/api/v1/login` | User authentication |
| POST | `/api/v1/metrics` | Submit health metrics |
| GET | `/api/v1/metrics` | Get health metrics |
| GET | `/api/v1/recommendation` | Get AI recommendations |
| POST | `/api/v1/hospitals/nearest` | Find nearby hospitals |
| POST | `/api/v1/location/nearest` | Find nearby users |
| GET | `/health` | Health check |

### Run
```shell
cd dockify-backend

# Set up configuration in config/
# Run migrations from db/

go run main.go
```

---

## ML Services (dockify-ml)

An AI system combining classical ML, LLM fine-tuning, and RAG for health analysis.

### Tech Stack
| Technology | Purpose |
|------------|---------|
| MindSpore | Deep learning framework (Huawei) |
| openGauss | Vector database for RAG (Huawei) |
| DeepSeek | LLM for medical Q&A |
| Sentence Transformers | Text embeddings |

### ML Models
| Model | Type | Purpose |
|-------|------|---------|
| **Sleep Quality Scorer** | Regression | Score 0-100 |
| **Sleep Stage Classifier** | Classification | deep/light/rem/restless |
| **Lifestyle Classifier** | Classification | sedentary/active/athletic |
| **Activity Predictor** | Regression | Calories prediction |
| **Health Risk Scorer** | Regression | Risk score 0-100 |

### Components

1. **Classical ML Models** - MindSpore-trained models for health predictions
2. **DeepSeek LoRA Fine-tuning** - Parameter-efficient LLM adaptation for medical domain
3. **RAG System** - Medical Q&A with openGauss vector database

### Run

**ML Predictions:**
```shell
cd dockify-ml/models
python main.py
```

**RAG System:**
```shell
cd dockify-ml/rag
pip install -r requirements.txt
python indexer.py    # Index medical documents
python query.py      # Interactive Q&A
```

**LLM Fine-tuning:**

Open `notebooks/deepseek_lora_finetuning.ipynb` in Jupyter.

---

## Quick Start

### Prerequisites
- **Frontend**: Android Studio / Xcode, JDK 17+
- **Backend**: Go 1.24+, PostgreSQL
- **ML**: Python 3.11+, MindSpore, openGauss

### 1. Start Backend
```shell
cd dockify-backend
# Configure database in config/
go run main.go
```

### 2. Start ML Services (optional)
```shell
cd dockify-ml/rag
python indexer.py && python query.py
```

### 3. Build & Run Mobile App
```shell
cd dockify-frontend
./gradlew :composeApp:assembleDebug
# Install APK on Android device
```

---

## License

This project showcases Huawei MindSpore and openGauss integration with modern mobile and AI technologies.
