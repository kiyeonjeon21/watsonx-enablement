"""
Complete Milvus Operations Example

This script demonstrates basic vector operations using Milvus and watsonx.ai embeddings,
including ingestion, search, and collection management.
"""

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.foundation_models import Embeddings
from dotenv import load_dotenv
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
from pathlib import Path

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
print("Loading environment variables...")
load_dotenv()

# Environment variables
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_USER = os.getenv("MILVUS_USERNAME")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")

WATSONX_URL = os.getenv("WATSONX_URL")
API_KEY = os.getenv("API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
EMBEDDING_MODEL = os.getenv("WATSONX_EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

# Check required environment variables
required_vars = [
    "MILVUS_HOST", "MILVUS_PORT", "MILVUS_COLLECTION",
    "API_KEY", "PROJECT_ID"
]

for var in required_vars:
    if not locals()[var]:
        print(f"Error: {var} must be set in the .env file.")
        exit(1)

# Collection schema constants
MAX_LENGTHS = {
    "episode_id": 100,
    "program": 100,
    "title": 500,
    "episode_date": 50,
    "dialogue": 10000
}

# Index parameters
INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 200}
}

# def connect_to_milvus():
#     """Connect to Milvus server"""
#     try:
#         connections.connect(
#             host=MILVUS_HOST,
#             port=MILVUS_PORT,
#             user=MILVUS_USER,
#             password=MILVUS_PASSWORD,
#             secure=True
#         )
#         logger.info(f"Connected to Milvus server at {MILVUS_HOST}:{MILVUS_PORT}")
#     except Exception as e:
#         logger.error(f"Failed to connect to Milvus: {e}")
#         raise

def connect_to_milvus():
    """Connect to Milvus server"""
    try:
        # 연결 전에 현재 상태 출력
        print(f"\nTrying to connect to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
        print(f"Username: {MILVUS_USER}")
        print(f"Secure connection: True")
        
        connections.connect(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            user=MILVUS_USER,
            password=MILVUS_PASSWORD,
            secure=True
        )
        
        # 연결 확인
        if connections.has_connection("default"):
            print("Successfully connected to Milvus")
        else:
            print("Connection failed")
            
        logger.info(f"Connected to Milvus server at {MILVUS_HOST}:{MILVUS_PORT}")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        # 상세 에러 정보 출력
        print(f"\nDetailed error information:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        raise

def create_collection():
    """Create a new collection with the defined schema"""
    try:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="episode_id", dtype=DataType.VARCHAR, max_length=MAX_LENGTHS["episode_id"]),
            FieldSchema(name="program", dtype=DataType.VARCHAR, max_length=MAX_LENGTHS["program"]),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=MAX_LENGTHS["title"]),
            FieldSchema(name="episode_date", dtype=DataType.VARCHAR, max_length=MAX_LENGTHS["episode_date"]),
            FieldSchema(name="dialogue", dtype=DataType.VARCHAR, max_length=MAX_LENGTHS["dialogue"]),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
        ]
        
        schema = CollectionSchema(fields=fields, description="News Dialogues")
        collection = Collection(name=MILVUS_COLLECTION, schema=schema)
        
        logger.info(f"Created new collection: {MILVUS_COLLECTION}")
        return collection
        
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise

def generate_embedding(texts: list) -> list:
    """Generate embedding for texts using watsonx.ai"""
    try:
        credentials = Credentials(
            url=WATSONX_URL,
            api_key=API_KEY
        )

        embed_params = {
            EmbedParams.TRUNCATE_INPUT_TOKENS: 512,
            EmbedParams.RETURN_OPTIONS: {'input_text': True}
        }

        embedding_model = Embeddings(
            model_id=EMBEDDING_MODEL,
            params=embed_params,
            credentials=credentials,
            project_id=PROJECT_ID
        )

        return embedding_model.embed_documents(texts=texts)
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise

def drop_collection():
    """Drop the existing collection"""
    try:
        # Connect to Milvus first
        connect_to_milvus()
        
        if utility.has_collection(MILVUS_COLLECTION):
            utility.drop_collection(MILVUS_COLLECTION)
            logger.info(f"Dropped collection: {MILVUS_COLLECTION}")
        else:
            logger.info(f"Collection {MILVUS_COLLECTION} does not exist")
    except Exception as e:
        logger.error(f"Failed to drop collection: {e}")
        raise

def ingest_data(data_path: str, batch_size: int = 1000):
    """Ingest data from CSV file into Milvus"""
    try:
        # Connect to Milvus
        connect_to_milvus()
        
        # Read data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} records from {data_path}")
        
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Create or get collection
        if utility.has_collection(MILVUS_COLLECTION):
            collection = Collection(MILVUS_COLLECTION)
        else:
            collection = create_collection()
            
        # Process in batches
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size]
            
            # Split dialogues and prepare data
            all_chunks = []
            chunk_indices = []
            
            for idx, dialogue in enumerate(batch_df['dialogue']):
                chunks = text_splitter.split_text(str(dialogue))
                all_chunks.extend(chunks)
                chunk_indices.extend([idx] * len(chunks))
                
            # Generate embeddings
            embeddings = generate_embedding(all_chunks)
            
            # Prepare data for insertion
            entities = [
                batch_df.iloc[chunk_indices]['episode_id'].astype(str).tolist(),
                batch_df.iloc[chunk_indices]['program'].tolist(),
                batch_df.iloc[chunk_indices]['title'].tolist(),
                batch_df.iloc[chunk_indices]['episode_date'].astype(str).tolist(),
                all_chunks,
                embeddings
            ]
            
            # Insert data
            collection.insert(entities)
            logger.info(f"Inserted batch {i//batch_size + 1}, processed {len(all_chunks)} chunks")
            
        # Create index if it doesn't exist
        if not collection.has_index():
            collection.create_index(field_name="embedding", index_params=INDEX_PARAMS)
            logger.info("Created index on embedding field")
            
        logger.info("Data ingestion completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to ingest data: {e}")
        raise
    finally:
        # Cleanup
        try:
            collection.release()
            connections.disconnect("default")
        except:
            pass

def search_similar_texts(query: str, limit: int = 3):
    """Search for similar texts in Milvus"""
    try:
        # Connect to Milvus
        connect_to_milvus()

        # Check if collection exists
        if not utility.has_collection(MILVUS_COLLECTION):
            logger.error(f"Collection does not exist: {MILVUS_COLLECTION}")
            return []

        # Load collection
        collection = Collection(MILVUS_COLLECTION)
        collection.load()

        # Generate query embedding
        query_embedding = generate_embedding([query])[0]

        # Perform search
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 64}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["episode_id", "program", "title", "episode_date", "dialogue"]
        )

        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "similarity": 1 - hit.distance,  # Convert distance to similarity
                    "content": hit.entity.get("dialogue"),
                    "metadata": {
                        "episode_id": hit.entity.get("episode_id"),
                        "program": hit.entity.get("program"),
                        "title": hit.entity.get("title"),
                        "date": hit.entity.get("episode_date")
                    }
                }
                formatted_results.append(result)

        return formatted_results

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise
    finally:
        # Cleanup
        try:
            collection.release()
            connections.disconnect("default")
        except:
            pass

if __name__ == "__main__":
    # Set up data paths
    data_dir = Path("data")
    sample_data_path = data_dir / "sample_dialogue_data.csv"
    
    # Create sample data if it doesn't exist
    if not sample_data_path.exists():
        print("\nCreating sample dataset...")
        from create_sample_dataset import create_sample_data
        create_sample_data(str(sample_data_path))
    
    # Example usage
    print("\nDropping existing collection...")
    drop_collection()
    
    print("\nIngesting data...")
    ingest_data(str(sample_data_path))
    
    # Search example
    query = "Tell me about climate change discussions"
    print(f"\nSearching for: '{query}'")
    
    try:
        results = search_similar_texts(query, limit=3)
        
        print("\nSearch Results:")
        for i, result in enumerate(results, 1):
            print(f"\nResult #{i}")
            print(f"Similarity: {result['similarity']:.4f}")
            print(f"Program: {result['metadata']['program']}")
            print(f"Title: {result['metadata']['title']}")
            print(f"Date: {result['metadata']['date']}")
            print(f"Content: {result['content'][:200]}...")
            
    except Exception as e:
        print(f"An error occurred: {e}")