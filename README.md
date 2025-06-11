# watsonx-enablement

A comprehensive toolkit for IBM watsonx.ai development, featuring modular RAG pipelines, AI service deployments, and SDK implementations.

## 📁 Project Structure

```
watsonx-enablement/
├── api/           # API runners and testing utilities
├── data/          # Sample datasets and test files
├── deploy/        # Production-ready AI service deployments
├── doc/           # Documentation and guides
├── pipeline/      # Modular RAG pipeline implementation
├── sdk/           # Basic SDK functions (WIP)
└── utils/         # Utility scripts and helpers
```

## 🚀 Quick Start

### 1. Environment Setup

**Create Python Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Install Dependencies**
```bash
pip install -r requirements.txt
```

**Configure Environment Variables**
Copy `.env.copy` to `.env` and update with your IBM Cloud credentials:
```env
# IBM Cloud Configuration
API_KEY=your_ibm_cloud_api_key          # IBM Cloud API Key
PROJECT_ID=your_watsonx_project_id      # watsonx.ai Project ID
WATSONX_URL=https://us-south.ml.cloud.ibm.com
SPACE_ID=your_deployment_space_id       # IBM Cloud Deployment Space ID

# Milvus Vector Database Configuration
MILVUS_HOST=localhost                   # Milvus server host address
MILVUS_PORT=19530                       # Milvus server port
MILVUS_USERNAME=root                    # Milvus username (if authentication enabled)
MILVUS_PASSWORD=milvus                  # Milvus password (if authentication enabled)
```

### 2. Data Preparation

**Download RAG Dataset**
```bash
# Option 1: From Hugging Face
git clone https://huggingface.co/datasets/neural-bridge/rag-dataset-1200 data/rag-dataset-1200

# Option 2: Convert from Parquet (if needed)
python utils/convert_parquet_to_csv.py
```

## 🔧 Core Components

### 🚀 Deploy - Production AI Services

Ready-to-deploy AI service implementations:

- **RAG Services**: `aiservice_rag_milvus.py`, `aiservice_rag_inmemory.py`
- **Agent Services**: `aiservice_agent_basic.py`
- **Model Functions**: Embedding, reranking, and statistical models
- **Monitoring**: `endpoint_monitor.py`

**Usage:**
```bash
python deploy/aiservice_rag_milvus.py      # Milvus-based RAG
python deploy/aiservice_rag_inmemory.py    # In-memory RAG
python deploy/aiservice_agent_basic.py     # Basic AI agent
```

### 🔄 Pipeline - Modular RAG Implementation

Step-by-step RAG pipeline with clear separation of concerns:

1. **01_initialize.py** - Pipeline initialization
2. **02_validate_parameters.py** - Input validation
3. **03_process_query_embedding.py** - Query embedding
4. **04_search_vector_database.py** - Vector search
5. **05_rerank_results.py** - Result reranking
6. **06_generate_response.py** - Response generation
7. **07_save_final_results.py** - Result persistence

**Usage:**
```bash
# Run complete pipeline
python pipeline/full-pipeline.py

# Run individual steps
python pipeline/01_initialize.py
# ... continue with other steps
```

### 🌐 API - Testing and Integration

API runners and testing utilities:

- **Project Templates**: `run_project_prompt_template.py`, `run_project_chat_template.py`
- **Deployment Testing**: `run_deploy_prompt_template.py`, `test_deployed_function.py`

**Usage:**
```bash
# Test project-based prompt templates
python api/run_project_prompt_template.py

# Test chat templates
python api/run_project_chat_template.py

# Test deployed functions
python api/test_deployed_function.py
```

### 🛠️ SDK - Development Tools

Basic SDK implementations for common operations:

- **Text Generation**: `generate_basic.py`
- **Embeddings**: `generate_embedding.py`
- **Chat**: `chat_basic_with_langchain.py`
- **Vector DB**: `milvus_basic.py`, `milvus_ingestion.py`

**Usage:**
```bash
python sdk/generate_basic.py        # Basic text generation
python sdk/generate_embedding.py    # Generate embeddings
python sdk/milvus_basic.py          # Milvus operations
```

### 🔧 Utils - Helper Scripts

Utility scripts for data processing and project management:

- **Data Processing**: `cleanup_software_specs.py`, `create_sample_dataset.py`
- **Metrics**: `custom_metric.py`
- **Conversion**: `convert_parquet_to_csv.py`

**Usage:**
```bash
python utils/create_sample_dataset.py    # Create sample data
python utils/cleanup_software_specs.py   # Clean specifications
python utils/convert_parquet_to_csv.py   # Convert data formats
```

## 💾 Data

Sample datasets and test files for development:

- **Dialogue Data**: `sample_dialogue_data.csv`
- **RAG Datasets**: Training and test sets for RAG evaluation
- **Multimodal**: Sample images (`multimodal01-03.jpeg`)
- **Documents**: PDF files for document QA testing

> **Note**: Large dataset files are excluded from version control. Download separately using the data preparation steps above.

## 🧪 Sample Questions

Test the system with these sample queries:

**Document QA (Academy Awards PDF):**
- "Who won Best Actor?"
- "How many awards did Oppenheimer win?"
- "List all the winners in major categories"

**RAG Dataset Queries:**
- Technical questions from the RAG dataset
- Multi-turn conversations
- Domain-specific queries

## 🔍 Development Status

- ✅ **Deploy**: Production-ready AI services
- ✅ **Pipeline**: Complete modular RAG implementation
- ✅ **API**: Testing and integration utilities
- ✅ **Utils**: Data processing helpers
- 🚧 **SDK**: Basic implementations (Work in Progress)
- 📚 **Documentation**: Comprehensive guides available in `/doc`

## 🤝 Contributing

1. Follow the modular structure when adding new components
2. Place production code in `/deploy`
3. Use `/pipeline` for step-by-step processing logic
4. Add utilities to `/utils`
5. Update this README when adding new features

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.