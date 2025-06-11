# ##################################################################################### #
# 이 스크립트는 IBM Watson Studio Pipelines와 호환되도록 리팩토링되었습니다.             #
# 각 함수는 파이프라인의 한 단계를 나타내며, 독립적인 노트북 셀에 복사하여 사용할 수 있습니다. #
# 전역 상태(global state)를 제거하고 함수 간 데이터 전달을 명확히 하여 모듈성을 높였습니다.    #
# ##################################################################################### #

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
from ibm_watson_studio_pipelines import WSPipelines

def initialize_pipeline():
    """모든 연결 및 구성을 초기화하고 Watson Pipelines 클라이언트를 설정합니다."""
    try:
        load_dotenv()
        
        api_key = os.getenv("API_KEY")
        project_id = os.getenv("PROJECT_ID")
        watsonx_url = os.getenv("WATSONX_URL")
        
        config = {
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
        
        credentials = Credentials(url=config["watsonx_url"], api_key=config["api_key"])
        client = APIClient(credentials=credentials)
        pipelines_client = WSPipelines.from_apikey(apikey=config["api_key"])
        
        print("✅ Initialization completed successfully")
        
        return {
            "config": config,
            "credentials": credentials,
            "client": client,
            "pipelines_client": pipelines_client
        }
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        raise

# ############## #
# 02. 파라미터 체크 #
# ############## #
def validate_parameters(config):
    """필요한 모든 매개변수와 구성을 검증합니다."""
    try:
        required_params = [
            "api_key", "project_id", "watsonx_url", 
            "milvus_host", "milvus_port", "milvus_user", 
            "milvus_password", "milvus_collection"
        ]
        
        missing_params = [p for p in required_params if not config.get(p)]
        if missing_params:
            raise Exception(f"Missing required parameters: {missing_params}")
        
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
        
        print("✅ Parameter validation completed successfully")
        return {"validation_status": "success"}
        
    except Exception as e:
        print(f"❌ Parameter validation failed: {e}")
        raise

# ############## #
# 03. 임베딩 #
# ############## #
def process_query_embedding(user_query, config, credentials, project_id):
    """사용자 쿼리를 처리하고 임베딩을 생성합니다."""
    try:
        embedding_params = {
            EmbedParams.TRUNCATE_INPUT_TOKENS: 1024,
            EmbedParams.RETURN_OPTIONS: {"input_text": True}
        }
        
        embeddings = Embeddings(
            model_id=config["embedding_model"],
            params=embedding_params,
            credentials=credentials,
            project_id=project_id,
            batch_size=1000,
            concurrency_limit=5
        )
        
        print(f"Generating embedding for query: {user_query}")
        query_embedding = embeddings.embed_query(text=user_query)
        embedding_dim = len(query_embedding)
        
        print(f"✅ Embedding completed - Vector length: {embedding_dim}")
        
        return {
            "user_query": user_query,
            "query_embedding": query_embedding,
            "embedding_dim": embedding_dim
        }
        
    except Exception as e:
        print(f"❌ Query embedding failed: {e}")
        raise

# ############## #
# 04. 벡터디비 검색 #
# ############## #
def create_sample_collection(config, credentials, project_id, embedding_dim):
    """테스트용 샘플 컬렉션을 생성합니다."""
    if utility.has_collection(config["milvus_collection"]):
        utility.drop_collection(config["milvus_collection"])
        print(f"Dropped existing collection: {config['milvus_collection']}")
    
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
    ]
    schema = CollectionSchema(fields=fields, description="Sample collection for RAG pipeline")
    collection = Collection(name=config["milvus_collection"], schema=schema)
    print(f"Created collection: {config['milvus_collection']}")
    
    sample_docs = [
        "Artificial intelligence is a branch of computer science that aims to create intelligent machines.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn from experience.",
        "Deep learning is a type of machine learning that uses neural networks with multiple layers.",
        "Natural language processing focuses on the interaction between computers and human language.",
        "Computer vision enables machines to interpret and understand visual information.",
    ]
    
    embeddings_model = Embeddings(
        model_id=config["embedding_model"],
        credentials=credentials,
        project_id=project_id
    )
    sample_embeddings = embeddings_model.embed_documents(texts=sample_docs)
    
    collection.insert([sample_docs, sample_embeddings])
    collection.flush()
    print(f"Inserted {len(sample_docs)} sample documents.")
    
    index_params = {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
    print(f"Collection '{config['milvus_collection']}' created and loaded with {collection.num_entities} entities.")
    return collection

def search_vector_database(config, credentials, project_id, query_embedding, embedding_dim, limit=5):
    """벡터 데이터베이스에서 유사한 문서를 검색합니다."""
    try:
        print(f"Connecting to Milvus at {config['milvus_host']}:{config['milvus_port']}")
        connections.connect(
            host=config["milvus_host"],
            port=config["milvus_port"],
            user=config["milvus_user"],
            password=config["milvus_password"],
            secure=True
        )
        
        recreate_collection = not utility.has_collection(config["milvus_collection"])
        if not recreate_collection:
            collection = Collection(config["milvus_collection"])
            if collection.num_entities == 0:
                print("Collection is empty, recreating.")
                recreate_collection = True
        
        if recreate_collection:
            print("Collection not found or empty. Creating a sample collection.")
            collection = create_sample_collection(config, credentials, project_id, embedding_dim)
        else:
            collection = Collection(config["milvus_collection"])
            if not collection.has_index():
                index_params = {"metric_type": "COSINE", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}}
                collection.create_index(field_name="embedding", index_params=index_params)
                collection.flush()
            collection.load()

        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            output_fields=["text"]
        )
        
        search_results = [{
            "similarity": float(1 - hit.distance),
            "content": hit.entity.get("text", ""),
            "id": hit.id,
            "distance": float(hit.distance)
        } for hits in results for hit in hits]
        
        print(f"\nSearch Results ({len(search_results)} found):")
        for i, result in enumerate(search_results[:3]):
            print(f"  {i+1}. Similarity: {result['similarity']:.4f}, Content: {result['content'][:100]}...")
            
        collection.release()
        connections.disconnect("default")
        
        print(f"✅ Vector search completed - Found {len(search_results)} results")
        return {"search_results": search_results}
        
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        if 'collection' in locals():
            try:
                collection.release()
                connections.disconnect("default")
            except: pass
        raise

# ############## #
# 05. 리랭킹 #
# ############## #
def rerank_results(user_query, search_results, config, credentials, project_id):
    """재순위 지정 모델을 사용하여 검색 결과를 재정렬합니다."""
    try:
        if not search_results:
            print("No search results to rerank")
            return {"reranked_results": []}
        
        rerank = Rerank(
            model_id="cross-encoder/ms-marco-minilm-l-12-v2",
            credentials=credentials,
            project_id=project_id
        )
        documents = [result["content"] for result in search_results]
        
        print(f"Reranking {len(documents)} documents...")
        rerank_response = rerank.generate(query=user_query, inputs=documents)
        
        reranked_results = []
        if "results" in rerank_response:
            for i, rerank_item in enumerate(rerank_response["results"]):
                if i < len(search_results):
                    result = search_results[i].copy()
                    result["rerank_score"] = rerank_item.get("score", 0)
                    reranked_results.append(result)
            reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        else:
            reranked_results = search_results
        
        print(f"✅ Reranking completed - {len(reranked_results)} results reranked")
        return {"reranked_results": reranked_results}
        
    except Exception as e:
        print(f"⚠️ Reranking failed, using original search results: {str(e)}")
        return {"reranked_results": search_results}

# ############## #
# 06. 생성 #
# ############## #
def generate_response(user_query, reranked_results, config, credentials, project_id):
    """검색된 컨텍스트를 사용하여 최종 응답을 생성합니다."""
    try:
        context_docs = reranked_results[:3]
        context = "\n\n".join([f"Document {i+1}: {doc['content']}" for i, doc in enumerate(context_docs)])
        
        generate_params = {GenParams.MAX_NEW_TOKENS: 500, GenParams.TEMPERATURE: 0.7, GenParams.TOP_P: 0.9}
        
        chat_model = ModelInference(
            model_id=config["generation_model"],
            params=generate_params,
            credentials=credentials,
            project_id=project_id
        )
        
        system_prompt = "You are a helpful AI assistant. Use the provided context documents to answer the user's question. If the context doesn't contain enough information, say so clearly. Always cite which document(s) you used for your answer."
        user_prompt = f"Context Documents:\n{context}\n\nQuestion: {user_query}\n\nPlease provide a comprehensive answer based on the context provided above."
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        print("Generating response...")
        response = chat_model.chat(messages=messages)
        generated_text = response['choices'][0].get('message', {}).get('content', '') if response and "choices" in response and response["choices"] else ""
        
        final_response = {
            "query": user_query,
            "context_docs": context_docs,
            "generated_response": generated_text,
            "metadata": {"model": config["generation_model"], "context_count": len(context_docs)}
        }
        
        print(f"✅ Response generation completed - {len(generated_text)} characters generated")
        return {"final_response": final_response}
        
    except Exception as e:
        print(f"❌ Response generation failed: {e}")
        raise

# ############## #
# 07. 결과 저장 #
# ############## #
def save_final_results(output_file, final_response, config, intermediate_results):
    """최종 결과와 파이프라인 상태를 저장합니다."""
    try:
        output_data = {
            "pipeline_execution_summary": {"timestamp": datetime.now().isoformat()},
            "final_response": final_response,
            "pipeline_inputs": {
                "user_query": final_response.get("query"),
                "config": {k: v for k, v in config.items() if k not in ["api_key", "milvus_password"]},
            },
            "intermediate_results_summary": {
                "query_embedding_length": intermediate_results.get("embedding_dim"),
                "search_results_count": len(intermediate_results.get("search_results", [])),
                "reranked_results_count": len(intermediate_results.get("reranked_results", []))
            }
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Final results saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Saving results failed: {e}")
        raise

# ############## #
# 메인 파이프라인 실행 #
# ############## #
def run_full_pipeline(user_query, output_file="pipeline_results.json"):
    """전체 RAG 파이프라인을 실행하고 Watson Studio Pipelines 통합을 시뮬레이션합니다."""
    print("🚀 Starting RAG Pipeline Execution (Watson Pipelines Simulation)")
    print("=" * 60)
    
    try:
        # 📋 Stage 1: Initialization
        # 이 노트북은 초기 구성을 로드하고 클라이언트를 설정합니다.
        print("\n📋 Stage 1: Initialization")
        init_results = initialize_pipeline()
        config = init_results["config"]
        credentials = init_results["credentials"]
        pipelines_client = init_results["pipelines_client"]
        project_id = config["project_id"]
        # >> Watson Pipeline: 이 단계의 결과(예: 상태)를 저장합니다.
        # pipelines_client.store_results({"initialization_status": "success"})
        
        # 🔍 Stage 2: Parameter Validation
        # 이 노트북은 구성 값의 유효성을 검사합니다.
        print("\n🔍 Stage 2: Parameter Validation")
        validate_parameters(config)
        # >> Watson Pipeline: 검증 상태를 저장합니다.
        # pipelines_client.store_results({"validation_status": "success"})
        
        # 🔤 Stage 3: Query Embedding
        # 이 노트북은 사용자 쿼리를 입력받아 임베딩 벡터를 생성합니다.
        print("\n🔤 Stage 3: Query Embedding")
        embedding_output = process_query_embedding(user_query, config, credentials, project_id)
        # >> Watson Pipeline: 생성된 임베딩과 차원을 다음 단계를 위해 저장합니다.
        # pipelines_client.store_results(embedding_output)
        
        # 🔎 Stage 4: Vector Database Search
        # 이 노트북은 임베딩을 사용하여 벡터 DB에서 관련 문서를 검색합니다.
        print("\n🔎 Stage 4: Vector Database Search")
        search_output = search_vector_database(config, credentials, project_id, embedding_output["query_embedding"], embedding_output["embedding_dim"])
        # >> Watson Pipeline: 검색 결과를 저장합니다.
        # pipelines_client.store_results(search_output)
        
        # 📊 Stage 5: Result Reranking
        # 이 노트북은 검색 결과를 재순위화하여 관련성을 높입니다.
        print("\n📊 Stage 5: Result Reranking")
        rerank_output = rerank_results(user_query, search_output["search_results"], config, credentials, project_id)
        # >> Watson Pipeline: 재순위화된 결과를 저장합니다.
        # pipelines_client.store_results(rerank_output)
        
        # 💭 Stage 6: Response Generation
        # 이 노트북은 재순위화된 결과를 컨텍스트로 사용하여 최종 답변을 생성합니다.
        print("\n💭 Stage 6: Response Generation")
        generation_output = generate_response(user_query, rerank_output["reranked_results"], config, credentials, project_id)
        # >> Watson Pipeline: 최종 생성된 응답을 저장합니다.
        # pipelines_client.store_results(generation_output)

        # 💾 Stage 7: Save Final Results (Local)
        # 이 단계는 파이프라인의 최종 결과를 로컬 파일에 요약하여 저장합니다.
        print("\n💾 Stage 7: Save Final Local Results")
        final_response = generation_output["final_response"]
        save_final_results(
            output_file, 
            final_response, 
            config, 
            {
                "embedding_dim": embedding_output["embedding_dim"],
                "search_results": search_output["search_results"],
                "reranked_results": rerank_output["reranked_results"]
            }
        )
        
        print("\n" + "=" * 60)
        print("🎉 Pipeline execution completed successfully!")
        
        print(f"\n📝 Query: {final_response['query']}")
        print(f"\n🤖 Response:\n{final_response['generated_response']}")
        
        return final_response
        
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    test_query = "What is artificial intelligence and how does it relate to machine learning?"
    
    try:
        result = run_full_pipeline(test_query)
    except Exception as e:
        print(f"\nTop-level error caught: Pipeline failed.")