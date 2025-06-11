import pandas as pd
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from pathlib import Path
import logging
from tqdm import tqdm
import time
import sys
import os
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.foundation_models import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Project paths
ROOT_DIR = Path.cwd()  # Use current working directory
DATA_DIR = ROOT_DIR / "src/data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Default values
DEFAULT_BATCH_SIZE = 1000
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

# Text splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set default level to INFO
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s' # Added logger name
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Set our logger to DEBUG

# Set higher level for verbose libraries
logging.getLogger("ibm_watsonx_ai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

class MilvusIngestor:
    def __init__(self, 
                 host: str = None,
                 port: str = None,
                 collection_name: str = None,
                 data_path: str = None,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        """Initialize the Milvus ingestor.
        
        Args:
            host (str): Milvus server host. If None, uses environment variable.
            port (str): Milvus server port. If None, uses environment variable.
            collection_name (str): Name of the collection to create/use. If None, uses environment variable.
            data_path (str): Path to processed data file. If None, uses default path.
            batch_size (int): Number of records to process in each batch
        """
        # Load environment variables from .env
        env_path = ROOT_DIR / ".env"
        if not env_path.exists():
            raise FileNotFoundError(f"No .env file found at {env_path}")
            
        # Clear any existing environment variables to ensure we're using only .env
        os.environ.clear()
        load_dotenv(env_path, override=True)
        logger.info(f"Loaded environment variables from {env_path}")
        
        # Log all environment variables for debugging
        logger.debug("Current environment variables:")
        for key, value in os.environ.items():
            if "MILVUS" in key or "WATSONX" in key:
                logger.debug(f"{key}: {value}")
        
        # Initialize Milvus connection parameters
        self.host = host or os.getenv("MILVUS_HOST")
        self.port = port or os.getenv("MILVUS_PORT")
        self.collection_name = collection_name or os.getenv("MILVUS_COLLECTION")
        
        if not all([self.host, self.port, self.collection_name]):
            raise ValueError("Milvus configuration not found. Please set MILVUS_HOST, MILVUS_PORT, and MILVUS_COLLECTION in .env file.")
            
        self.batch_size = batch_size
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Set data path
        self.data_path = Path(data_path) if data_path else PROCESSED_DATA_DIR / "dialogue_data_sample.csv"
        # self.data_path = Path(data_path) if data_path else PROCESSED_DATA_DIR / "dialogue_data.csv"
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        logger.info(f"Using data file: {self.data_path}")
        
        # Initialize Watsonx.ai credentials
        self._init_watsonx_credentials()
        
    def _init_watsonx_credentials(self):
        """Initialize Watsonx.ai credentials and parameters."""
        try:
            self.watsonx_credentials = Credentials(
                url=os.getenv("WATSONX_URL"),
                api_key=os.getenv("API_KEY")
            )
            self.watsonx_project_id = os.getenv("PROJECT_ID")
            self.embedding_model_id = os.getenv("WATSONX_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
            logger.info("Watsonx.ai credentials initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Watsonx.ai credentials: {e}")
            raise
        
    def connect(self):
        """Connect to Milvus server."""
        try:
            connections.connect(
                host=self.host,
                port=self.port,
                user=os.getenv("MILVUS_USERNAME"),
                password=os.getenv("MILVUS_PASSWORD"),
                secure=True
            )
            logger.info(f"Connected to Milvus server at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus server: {e}")
            raise
            
    def create_collection(self):
        """Create a new collection or get existing one."""
        try:
            if utility.has_collection(self.collection_name):
                logger.info(f"Using existing collection: {self.collection_name}")
                return Collection(self.collection_name)
                
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="episode_id", dtype=DataType.VARCHAR, max_length=MAX_LENGTHS["episode_id"]),
                FieldSchema(name="program", dtype=DataType.VARCHAR, max_length=MAX_LENGTHS["program"]),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=MAX_LENGTHS["title"]),
                FieldSchema(name="episode_date", dtype=DataType.VARCHAR, max_length=MAX_LENGTHS["episode_date"]),
                FieldSchema(name="dialogue", dtype=DataType.VARCHAR, max_length=MAX_LENGTHS["dialogue"]),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
            ]
            
            schema = CollectionSchema(fields=fields, description="NPR News Dialogues")
            collection = Collection(name=self.collection_name, schema=schema)
            
            # Create index (Moved to after ingestion)
            # collection.create_index(field_name="embedding", index_params=INDEX_PARAMS)
            logger.info(f"Created new collection schema: {self.collection_name}")
            return collection
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
            
    def generate_embeddings(self, texts: list) -> list:
        """Generate embeddings using IBM Watsonx.ai."""
        try:
            embed_params = {
                EmbedParams.TRUNCATE_INPUT_TOKENS: 1024,
                EmbedParams.RETURN_OPTIONS: {'input_text': True}
            }
            
            embedding_api = Embeddings(
                model_id=self.embedding_model_id,
                params=embed_params,
                credentials=self.watsonx_credentials,
                project_id=self.watsonx_project_id
            )
            
            logger.info(f"Generating embeddings using model: {self.embedding_model_id}")
            return embedding_api.embed_documents(texts=texts)
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
        
    def prepare_data_batch(self, df_batch: pd.DataFrame) -> list:
        """Prepare a batch of data for insertion."""
        # Split dialogues into chunks
        all_chunks = []
        chunk_indices = []
        
        for idx, dialogue in enumerate(df_batch['dialogue']):
            chunks = self.text_splitter.split_text(dialogue)
            all_chunks.extend(chunks)
            chunk_indices.extend([idx] * len(chunks))
            
        # Generate embeddings for chunks
        start_embed_time = time.time()
        embeddings = self.generate_embeddings(all_chunks)
        end_embed_time = time.time()
        logger.debug(f"Batch embedding generation time: {end_embed_time - start_embed_time:.2f}s for {len(all_chunks)} chunks")
        
        # Prepare data
        data = [
            df_batch.iloc[chunk_indices]['episode'].astype(str).tolist(),  # episode_id
            df_batch.iloc[chunk_indices]['program'].tolist(),              # program
            df_batch.iloc[chunk_indices]['title'].tolist(),                # title
            df_batch.iloc[chunk_indices]['episode_date'].astype(str).tolist(),  # episode_date
            all_chunks,                                                    # dialogue chunks
            embeddings                                                     # embeddings
        ]
        
        return data
        
    def drop_collection(self):
        """Drop the existing collection."""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} does not exist")
        except Exception as e:
            logger.error(f"Failed to drop collection: {e}")
            raise

    def ingest_data(self):
        """Ingest data into Milvus."""
        total_start_time = time.time()
        try:
            # Connect to Milvus
            self.connect()
            
            # Drop existing collection
            self.drop_collection()
            
            # Get or create collection
            collection = self.create_collection()
            
            # Load data
            load_start_time = time.time()
            logger.info(f"Loading data from {self.data_path}")
            df = pd.read_csv(self.data_path)
            total_records = len(df)
            load_end_time = time.time()
            logger.info(f"Loaded {total_records} records in {load_end_time - load_start_time:.2f}s")
            
            # Process in batches
            total_batches = (total_records + self.batch_size - 1) // self.batch_size
            logger.info(f"Processing {total_batches} batches of size {self.batch_size}")
            
            ingestion_start_time = time.time()
            total_inserted_count = 0
            for i in tqdm(range(0, total_records, self.batch_size), total=total_batches):
                batch_start_time = time.time()
                df_batch = df.iloc[i:i+self.batch_size]
                
                prep_start_time = time.time()
                data = self.prepare_data_batch(df_batch)
                prep_end_time = time.time()
                logger.debug(f"Batch {i//self.batch_size + 1} data preparation time: {prep_end_time - prep_start_time:.2f}s")
                
                insert_start_time = time.time()
                result = collection.insert(data)
                inserted_count = len(result.primary_keys)
                total_inserted_count += inserted_count
                insert_end_time = time.time()
                logger.debug(f"Batch {i//self.batch_size + 1} insertion time: {insert_end_time - insert_start_time:.2f}s, Inserted: {inserted_count}")
                
                batch_end_time = time.time()
                logger.debug(f"Batch {i//self.batch_size + 1} total processing time: {batch_end_time - batch_start_time:.2f}s")
                
            ingestion_end_time = time.time()
            logger.info(f"Total data preparation and insertion time: {ingestion_end_time - ingestion_start_time:.2f}s")
                
            # Flush collection
            flush_start_time = time.time()
            collection.flush()
            flush_end_time = time.time()
            logger.info(f"Flushed collection in {flush_end_time - flush_start_time:.2f}s. Total entities: {collection.num_entities}")
            logger.info(f"Successfully processed {total_records} records, inserted {total_inserted_count} entities.")

            # Create index after ingestion
            index_start_time = time.time()
            logger.info(f"Creating index for embedding field...")
            collection.create_index(field_name="embedding", index_params=INDEX_PARAMS)
            utility.wait_for_index_building_complete(self.collection_name) # Wait for index building
            index_end_time = time.time()
            logger.info(f"Index created successfully in {index_end_time - index_start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to ingest data: {e}")
            raise
        finally:
            try:
                connections.disconnect("default")
                logger.info("Disconnected from Milvus server")
            except Exception as e:
                logger.error(f"Error while disconnecting from Milvus: {e}")
            
    def search_test(self, query_text: str, limit: int = 3):
        """
        테스트용 검색 함수: 쿼리 텍스트에 대한 벡터 검색 수행
        
        Args:
            query_text (str): 검색할 텍스트 쿼리
            limit (int): 반환할 결과 수 (기본값: 3)
            
        Returns:
            list: 검색 결과 목록
        """
        try:
            # Milvus 연결
            self.connect()
            
            # 컬렉션 로드
            if not utility.has_collection(self.collection_name):
                logger.error(f"컬렉션이 존재하지 않음: {self.collection_name}")
                return []
                
            collection = Collection(self.collection_name)
            collection.load()
            logger.info(f"컬렉션 로드됨: {self.collection_name}")
            
            # 쿼리 텍스트 임베딩 생성
            query_embedding = self.generate_embeddings([query_text])[0]
            
            # 검색 수행
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 64}}
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["episode_id", "program", "title", "episode_date", "dialogue"]
            )
            
            # 결과 포매팅
            formatted_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "distance": hit.distance,
                        "episode_id": hit.entity.get("episode_id"),
                        "program": hit.entity.get("program"),
                        "title": hit.entity.get("title"),
                        "episode_date": hit.entity.get("episode_date"),
                        "dialogue": hit.entity.get("dialogue")
                    }
                    formatted_results.append(result)
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            raise
        finally:
            try:
                # 검색 완료 후 리소스 해제
                collection.release()
                connections.disconnect("default")
                logger.info("Milvus 서버 연결 해제됨")
            except Exception as e:
                logger.error(f"Milvus 연결 해제 중 오류: {e}")
            
if __name__ == "__main__":
    # Change to project root directory
    os.chdir(ROOT_DIR)
    
    ingestor = MilvusIngestor()
    ingestor.ingest_data()
    
    # 검색 테스트 예시 (주석 해제하여 사용)
    query = "According to host Rebecca Roberts, how has the rise in women's income affected household dynamics, particularly in terms of financial tension or role adjustments?"
    # query = "Rebecca Roberts"
    results = ingestor.search_test(query)
    print(f"검색 결과 (쿼리: '{query}'):")
    for i, r in enumerate(results):
        print(f"결과 #{i+1} (유사도: {r['distance']:.4f})")
        print(f"내용: {r}")
        # print(f"내용: {r['dialogue'][:]}...")