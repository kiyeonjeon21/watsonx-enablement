# ############## #
# 01. 초기화 #
# ############## #
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from dotenv import load_dotenv
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import os
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models import Rerank
import json
from datetime import datetime

# Global variables to store pipeline state
pipeline_state = {
    "checkpoints": {},
    "results": {},
    "config": {}
}

def save_checkpoint(stage_name, status, data=None, error=None):
    """Save checkpoint for each pipeline stage"""
    pipeline_state["checkpoints"][stage_name] = {
        "status": status,  # "success", "failed", "in_progress"
        "timestamp": datetime.now().isoformat(),
        "data": data,
        "error": str(error) if error else None
    }
    print(f"[CHECKPOINT] {stage_name}: {status}")

def check_stage_status(stage_name):
    """Check if a stage completed successfully"""
    checkpoint = pipeline_state["checkpoints"].get(stage_name, {})
    return checkpoint.get("status") == "success"

def initialize_pipeline():
    """Initialize all connections and configurations"""
    try:
        save_checkpoint("01_initialization", "in_progress")
        
        # Load environment variables
        load_dotenv()
        
        # Get required environment variables
        api_key = os.getenv("API_KEY")
        project_id = os.getenv("PROJECT_ID")
        watsonx_url = os.getenv("WATSONX_URL")
        
        # Store configuration
        pipeline_state["config"] = {
            "api_key": api_key,
            "project_id": project_id,
            "watsonx_url": watsonx_url,
            "milvus_host": os.getenv("MILVUS_HOST"),
            "milvus_port": os.getenv("MILVUS_PORT"),
            "milvus_user": os.getenv("MILVUS_USERNAME"),
            "milvus_password": os.getenv("MILVUS_PASSWORD"),
            "milvus_collection": os.getenv("MILVUS_COLLECTION"),
            "embedding_model": os.getenv("WATSONX_EMBEDDING_MODEL", "ibm/granite-embedding-278m-multilingual"),
            "generation_model": "meta-llama/llama-3-3-70b-instruct"
        }
        
        # Initialize credentials
        credentials = Credentials(
            url=watsonx_url,
            api_key=api_key
        )
        client = APIClient(credentials=credentials)
        
        # Store initialized objects
        pipeline_state["results"]["credentials"] = credentials
        pipeline_state["results"]["client"] = client
        
        save_checkpoint("01_initialization", "success", {"message": "All connections initialized"})
        print("✅ Initialization completed successfully")
        
    except Exception as e:
        save_checkpoint("01_initialization", "failed", error=e)
        raise

# ############## #
# 02. 파라미터 체크 #
# ############## #
def validate_parameters():
    """Validate all required parameters and configurations"""
    try:
        save_checkpoint("02_parameter_validation", "in_progress")
        
        # Check if initialization was successful
        if not check_stage_status("01_initialization"):
            raise Exception("Initialization stage must be completed first")
        
        config = pipeline_state["config"]
        required_params = [
            "api_key", "project_id", "watsonx_url", 
            "milvus_host", "milvus_port", "milvus_user", 
            "milvus_password", "milvus_collection"
        ]
        
        missing_params = []
        for param in required_params:
            if not config.get(param):
                missing_params.append(param)
        
        if missing_params:
            raise Exception(f"Missing required parameters: {missing_params}")
        
        # Test Milvus connection
        connections.connect(
            host=config["milvus_host"],
            port=config["milvus_port"],
            user=config["milvus_user"],
            password=config["milvus_password"],
            secure=True
        )
        
        if not connections.has_connection("default"):
            raise Exception("Failed to connect to Milvus")
        
        connections.disconnect("default")
        
        save_checkpoint("02_parameter_validation", "success", {"message": "All parameters validated"})
        print("✅ Parameter validation completed successfully")
        
    except Exception as e:
        save_checkpoint("02_parameter_validation", "failed", error=e)
        raise

# ############## #
# 03. 임베딩 #
# ############## #
def process_query_embedding(user_query):
    """Process user query and generate embedding"""
    try:
        save_checkpoint("03_embedding", "in_progress")
        
        if not check_stage_status("02_parameter_validation"):
            raise Exception("Parameter validation stage must be completed first")
        
        config = pipeline_state["config"]
        credentials = pipeline_state["results"]["credentials"]
        
        # Set up embedding parameters
        embedding_params = {
            EmbedParams.TRUNCATE_INPUT_TOKENS: 1024,
            EmbedParams.RETURN_OPTIONS: {
                "input_text": True
            }
        }
        
        # Initialize embeddings model
        embeddings = Embeddings(
            model_id=config["embedding_model"],
            params=embedding_params,
            credentials=credentials,
            project_id=config["project_id"],
            batch_size=1000,
            concurrency_limit=5
        )
        
        # Generate query embedding
        print(f"Generating embedding for query: {user_query}")
        query_embedding = embeddings.embed_query(text=user_query)
        
        # Store results
        pipeline_state["results"]["user_query"] = user_query
        pipeline_state["results"]["query_embedding"] = query_embedding
        pipeline_state["results"]["embeddings_model"] = embeddings
        
        save_checkpoint("03_embedding", "success", {
            "query": user_query,
            "embedding_length": len(query_embedding)
        })
        print(f"✅ Embedding completed - Vector length: {len(query_embedding)}")
        
    except Exception as e:
        save_checkpoint("03_embedding", "failed", error=e)
        raise

# ############## #
# 04. 벡터디비 검색 #
# ############## #
def create_sample_collection():
    """Create sample collection with some data for testing"""
    config = pipeline_state["config"]
    embedding_dim = len(pipeline_state["results"]["query_embedding"])
    print(f"Using embedding dimension: {embedding_dim}")
    
    # 컬렉션이 이미 존재하면 삭제
    if utility.has_collection(config["milvus_collection"]):
        utility.drop_collection(config["milvus_collection"])
        print(f"Dropped existing collection: {config['milvus_collection']}")
    
    # Define schema using proper PyMilvus method
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    ]
    schema = CollectionSchema(fields=fields, description="Sample collection for RAG pipeline")
    
    # Create collection
    collection = Collection(name=config["milvus_collection"], schema=schema)
    print(f"Created collection: {config['milvus_collection']}")
    
    # Sample documents
    sample_docs = [
        "Artificial intelligence is a branch of computer science that aims to create intelligent machines that can think and learn like humans.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        "Deep learning is a type of machine learning that uses neural networks with multiple layers to analyze and learn from data.",
        "Natural language processing is a field of artificial intelligence that focuses on the interaction between computers and human language.",
        "Computer vision is an area of artificial intelligence that enables machines to interpret and understand visual information from the world.",
        "Robotics combines artificial intelligence with mechanical engineering to create autonomous machines that can perform tasks.",
        "Data science involves extracting insights and knowledge from structured and unstructured data using scientific methods and algorithms.",
        "Big data refers to extremely large datasets that require special tools and techniques to store, process, and analyze effectively."
    ]
    
    # Generate embeddings for sample documents
    embeddings_model = pipeline_state["results"]["embeddings_model"]
    print("Generating embeddings for sample documents...")
    sample_embeddings = embeddings_model.embed_documents(texts=sample_docs)
    print(f"Generated embeddings for {len(sample_docs)} documents")
    
    # Prepare data for insertion
    data = [sample_docs, sample_embeddings]
    
    # Insert data
    insert_result = collection.insert(data)
    print(f"Inserted {insert_result.insert_count} entities.")
    
    # Flush to make sure data is written to disk
    collection.flush()
    print("Data flushed to disk")
    
    # Create index for the embedding field
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200}
    }
    
    print("Creating index...")
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Index created successfully")
    
    # Load collection into memory
    collection.load()
    print("Collection loaded into memory")
    
    # Verify collection has data
    entity_count = collection.num_entities
    print(f"Collection now contains {entity_count} entities")
    
    return collection

def search_vector_database(limit=5):
    """Search similar documents in vector database"""
    try:
        save_checkpoint("04_vector_search", "in_progress")
        
        if not check_stage_status("03_embedding"):
            raise Exception("Embedding stage must be completed first")
        
        config = pipeline_state["config"]
        query_embedding = pipeline_state["results"]["query_embedding"]
        
        # Connect to Milvus
        print(f"Connecting to Milvus at {config['milvus_host']}:{config['milvus_port']}")
        connections.connect(
            host=config["milvus_host"],
            port=config["milvus_port"],
            user=config["milvus_user"],
            password=config["milvus_password"],
            secure=True
        )
        
        # Check if collection exists and has data. If not, (re)create it.
        recreate_collection = False
        if not utility.has_collection(config["milvus_collection"]):
            print("Collection not found.")
            recreate_collection = True
        else:
            collection = Collection(config["milvus_collection"])
            if collection.num_entities == 0:
                print("Collection found, but it is empty. It will be recreated.")
                recreate_collection = True

        if recreate_collection:
            collection = create_sample_collection()
        else:
            print("Collection found with data, loading...")
            collection = Collection(config["milvus_collection"])
            if not collection.has_index():
                print("No index found. Creating index...")
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "HNSW", 
                    "params": {"M": 16, "efConstruction": 200}
                }
                collection.create_index(field_name="embedding", index_params=index_params)
                collection.flush()
            collection.load()
        
        # Check collection status
        entity_count = collection.num_entities
        print(f"Collection has {entity_count} entities")
        
        if entity_count == 0:
            print("Collection is empty, cannot perform search")
            search_results = []
        else:
            # Perform search
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}  # HNSW search parameter
            }
            
            print(f"Searching with query embedding of length {len(query_embedding)}")
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["text"]
            )
            
            # Process results
            search_results = []
            for hits in results:
                print(f"Found {len(hits)} hits")
                for hit in hits:
                    search_results.append({
                        "similarity": float(1 - hit.distance),
                        "content": hit.entity.get("text", ""),
                        "id": hit.id,
                        "distance": float(hit.distance)
                    })
        
        # Store results
        pipeline_state["results"]["search_results"] = search_results
        
        # Print search results for debugging
        print(f"\nSearch Results ({len(search_results)} found):")
        for i, result in enumerate(search_results[:3]):
            print(f"  {i+1}. Similarity: {result['similarity']:.4f}")
            print(f"     Content: {result['content'][:100]}...")
        
        # Cleanup
        try:
            collection.release()
        except Exception as e:
            print(f"Warning during collection release: {e}")
        
        connections.disconnect("default")
        
        save_checkpoint("04_vector_search", "success", {
            "results_count": len(search_results),
            "top_similarity": search_results[0]["similarity"] if search_results else 0
        })
        print(f"✅ Vector search completed - Found {len(search_results)} results")
        
    except Exception as e:
        save_checkpoint("04_vector_search", "failed", error=e)
        print(f"Error in vector search: {str(e)}")
        if 'collection' in locals():
            try:
                collection.release()
                connections.disconnect("default")
            except:
                pass
        raise

# ############## #
# 05. 리랭킹 #
# ############## #
def rerank_results():
    """Rerank search results using reranking model"""
    try:
        save_checkpoint("05_reranking", "in_progress")
        
        if not check_stage_status("04_vector_search"):
            raise Exception("Vector search stage must be completed first")
        
        config = pipeline_state["config"]
        credentials = pipeline_state["results"]["credentials"]
        user_query = pipeline_state["results"]["user_query"]
        search_results = pipeline_state["results"]["search_results"]
        
        if not search_results:
            print("No search results to rerank")
            pipeline_state["results"]["reranked_results"] = []
            save_checkpoint("05_reranking", "success", {"message": "No results to rerank"})
            return
        
        # Initialize reranker
        rerank = Rerank(
            model_id="cross-encoder/ms-marco-minilm-l-12-v2",
            credentials=credentials,
            project_id=config["project_id"]
        )
        
        # Prepare documents for reranking
        documents = [result["content"] for result in search_results]
        
        # Perform reranking
        print(f"Reranking {len(documents)} documents...")
        rerank_response = rerank.generate(
            query=user_query,
            inputs=documents
        )
        
        # Combine original results with rerank scores
        reranked_results = []
        if "results" in rerank_response:
            for i, rerank_item in enumerate(rerank_response["results"]):
                if i < len(search_results):
                    result = search_results[i].copy()
                    result["rerank_score"] = rerank_item.get("score", 0)
                    reranked_results.append(result)
            
            # Sort by rerank score
            reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        else:
            # Fallback to original results if reranking fails
            reranked_results = search_results
        
        # Store results
        pipeline_state["results"]["reranked_results"] = reranked_results
        
        save_checkpoint("05_reranking", "success", {
            "reranked_count": len(reranked_results),
            "top_score": reranked_results[0]["rerank_score"] if reranked_results else 0
        })
        print(f"✅ Reranking completed - {len(reranked_results)} results reranked")
        
    except Exception as e:
        # Fallback to original search results
        pipeline_state["results"]["reranked_results"] = pipeline_state["results"]["search_results"]
        save_checkpoint("05_reranking", "failed", error=e)
        print(f"⚠️  Reranking failed, using original search results: {str(e)}")

# ############## #
# 06. 생성 #
# ############## #
def generate_response():
    """Generate final response using retrieved context"""
    try:
        save_checkpoint("06_generation", "in_progress")
        
        if not check_stage_status("05_reranking"):
            raise Exception("Reranking stage must be completed first")
        
        config = pipeline_state["config"]
        credentials = pipeline_state["results"]["credentials"]
        user_query = pipeline_state["results"]["user_query"]
        reranked_results = pipeline_state["results"]["reranked_results"]
        
        # Prepare context from top results
        context_docs = reranked_results[:3]  # Use top 3 results
        context = "\n\n".join([f"Document {i+1}: {doc['content']}" for i, doc in enumerate(context_docs)])
        
        # Set up generation parameters
        generate_params = {
            GenParams.MAX_NEW_TOKENS: 500,
            GenParams.TEMPERATURE: 0.7,
            GenParams.TOP_P: 0.9
        }
        
        # Initialize model
        chat_model = ModelInference(
            model_id=config["generation_model"],
            params=generate_params,
            credentials=credentials,
            project_id=config["project_id"]
        )
        
        # Prepare chat messages with context
        system_prompt = """You are a helpful AI assistant. Use the provided context documents to answer the user's question. 
If the context doesn't contain enough information to answer the question, say so clearly.
Always cite which document(s) you used for your answer."""
        
        user_prompt = f"""Context Documents:
{context}

Question: {user_query}

Please provide a comprehensive answer based on the context provided above."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print("Generating response...")
        response = chat_model.chat(messages=messages)
        
        # Extract response content
        generated_text = ""
        if response and "choices" in response and response["choices"]:
            generated_text = response['choices'][0].get('message', {}).get('content', '')
        
        # Store results
        final_response = {
            "query": user_query,
            "context_docs": context_docs,
            "generated_response": generated_text,
            "metadata": {
                "model": config["generation_model"],
                "context_count": len(context_docs),
                "generation_params": generate_params
            }
        }
        
        pipeline_state["results"]["final_response"] = final_response
        
        save_checkpoint("06_generation", "success", {
            "response_length": len(generated_text),
            "context_docs_used": len(context_docs)
        })
        print(f"✅ Response generation completed - {len(generated_text)} characters generated")
        
    except Exception as e:
        save_checkpoint("06_generation", "failed", error=e)
        raise

# ############## #
# 07. 결과 저장 #
# ############## #
def save_results(output_file="pipeline_results.json"):
    """Save final results and pipeline state"""
    try:
        save_checkpoint("07_save_results", "in_progress")
        
        if not check_stage_status("06_generation"):
            raise Exception("Generation stage must be completed first")
        
        # Prepare output data
        output_data = {
            "pipeline_execution": {
                "timestamp": datetime.now().isoformat(),
                "checkpoints": pipeline_state["checkpoints"],
                "success": all(cp["status"] == "success" for cp in pipeline_state["checkpoints"].values())
            },
            "final_response": pipeline_state["results"]["final_response"],
            "pipeline_state": {
                "config": {k: v for k, v in pipeline_state["config"].items() if k not in ["api_key", "milvus_password"]},
                "query_embedding_length": len(pipeline_state["results"].get("query_embedding", [])),
                "search_results_count": len(pipeline_state["results"].get("search_results", [])),
                "reranked_results_count": len(pipeline_state["results"].get("reranked_results", []))
            }
        }
        
        # Save to file
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        save_checkpoint("07_save_results", "success", {
            "output_file": str(output_path),
            "file_size": output_path.stat().st_size
        })
        print(f"✅ Results saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        save_checkpoint("07_save_results", "failed", error=e)
        raise

# ############## #
# 메인 파이프라인 실행 #
# ############## #
def run_full_pipeline(user_query, output_file="pipeline_results.json"):
    """Run the complete RAG pipeline"""
    print("🚀 Starting RAG Pipeline Execution")
    print("=" * 50)
    
    try:
        # Stage 1: Initialize
        print("\n📋 Stage 1: Initialization")
        initialize_pipeline()
        
        # Stage 2: Validate parameters
        print("\n🔍 Stage 2: Parameter Validation")
        validate_parameters()
        
        # Stage 3: Process query embedding
        print("\n🔤 Stage 3: Query Embedding")
        process_query_embedding(user_query)
        
        # Stage 4: Search vector database
        print("\n🔎 Stage 4: Vector Database Search")
        search_vector_database()
        
        # Stage 5: Rerank results
        print("\n📊 Stage 5: Result Reranking")
        rerank_results()
        
        # Stage 6: Generate response
        print("\n💭 Stage 6: Response Generation")
        generate_response()
        
        # Stage 7: Save results
        print("\n💾 Stage 7: Save Results")
        output_path = save_results(output_file)
        
        print("\n" + "=" * 50)
        print("🎉 Pipeline execution completed successfully!")
        
        # Display final response
        final_response = pipeline_state["results"]["final_response"]
        print(f"\n📝 Query: {final_response['query']}")
        print(f"\n🤖 Response:\n{final_response['generated_response']}")
        
        return final_response
        
    except Exception as e:
        print(f"\n❌ Pipeline failed at stage: {str(e)}")
        print("\nCheckpoint status:")
        for stage, checkpoint in pipeline_state["checkpoints"].items():
            status_emoji = "✅" if checkpoint["status"] == "success" else "❌" if checkpoint["status"] == "failed" else "⏳"
            print(f"  {status_emoji} {stage}: {checkpoint['status']}")
        raise

# Example usage
if __name__ == "__main__":
    # Example query
    test_query = "What is artificial intelligence and how does it work?"
    
    try:
        result = run_full_pipeline(test_query)
    except Exception as e:
        print(f"Pipeline execution failed: {e}")