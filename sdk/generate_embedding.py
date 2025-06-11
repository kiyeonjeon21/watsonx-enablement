"""
Embeddings Example with watsonx.ai

This script demonstrates text embedding capabilities of watsonx.ai.
It covers:
1. Basic setup and authentication
2. Text embedding generation
3. Batch processing
4. Query and document embeddings
"""

from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from dotenv import load_dotenv
import os

# 1. Load environment variables
print("1. Loading environment variables...")
load_dotenv()

# 2. Get API key and project ID
print("2. Getting API key and project ID...")
API_KEY = os.getenv("API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")

# 3. Set up model credentials
print("3. Setting up model credentials...")
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key=API_KEY
)

# 4. Show available embedding models
print("\n=== Available Embedding Models ===")
client = APIClient(credentials=credentials)
client.foundation_models.EmbeddingModels.show()

# 5. Configure embedding parameters
print("\n4. Configuring embedding parameters...")
embedding_params = {
    EmbedParams.TRUNCATE_INPUT_TOKENS: 1024,
    EmbedParams.RETURN_OPTIONS: {
        "input_text": True
    }
}

# 6. Initialize the embeddings service
print("5. Initializing the embeddings service...")
embeddings = Embeddings(
    model_id="ibm/granite-embedding-278m-multilingual",  # Example model ID
    params=embedding_params,
    credentials=credentials,
    project_id=PROJECT_ID,
    batch_size=1000,
    concurrency_limit=5
)

try:
    # 7. Single query embedding
    print("\n=== Single Query Embedding Example ===")
    query = "What is a Generative AI?"
    print(f"Query: {query}")
    embedding_vector = embeddings.embed_query(text=query)
    print(f"Embedding vector length: {len(embedding_vector)}")
    print(f"First few dimensions: {embedding_vector[:5]}")

    # 8. Batch document embeddings
    print("\n=== Batch Document Embeddings Example ===")
    documents = [
        "Generative AI is a type of artificial intelligence that can create new content.",
        "Large language models are a type of generative AI that can understand and generate human-like text.",
        "Watsonx.ai provides enterprise-ready AI models and tools."
    ]
    print("Generating embeddings for documents...")
    document_vectors = embeddings.embed_documents(texts=documents)
    print(f"Number of documents processed: {len(document_vectors)}")
    print(f"First document vector length: {len(document_vectors[0])}")

    # 9. Generate embeddings with custom parameters
    print("\n=== Custom Parameters Example ===")
    custom_params = {
        EmbedParams.TRUNCATE_INPUT_TOKENS: 1024
    }
    response = embeddings.generate(
        inputs=["This is a test sentence."],
        params=custom_params
    )
    print("Response with custom parameters:")
    print(response)

except Exception as e:
    print(f"An error occurred: {str(e)}")
finally:
    # Close persistent connection if it was used
    embeddings.close_persistent_connection() 