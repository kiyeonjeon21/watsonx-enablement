import os
import json
import traceback
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
from pymilvus import connections, Collection, utility
from ibm_watson_studio_pipelines import WSPipelines


"""쿼리 임베딩을 사용하여 벡터 DB에서 문서를 검색합니다. 실패 시 오류를 기록합니다."""
api_key = os.getenv("API_KEY")
pipelines_client = WSPipelines.from_apikey(apikey=api_key)

try:
    config_json = os.getenv("config_json")
    if not config_json:
        raise ValueError("Input 'config_json' not found in environment variables.")
    config = json.loads(config_json)
    user_query = os.getenv("user_query", "What is Artificial Intelligence?")
    
    # 쿼리 임베딩 생성
    credentials = Credentials(url=config["watsonx_url"], api_key=config["api_key"])
    embeddings = Embeddings(model_id=config["embedding_model"], credentials=credentials, project_id=config["project_id"])
    
    print(f"Generating embedding for query: '{user_query}'")
    query_embedding = embeddings.embed_query(text=user_query)
    
    connections.connect(host=config["milvus_host"], port=config["milvus_port"], user=config["milvus_user"], password=config["milvus_password"], secure=True)
    if not utility.has_collection(config["milvus_collection"]):
        raise ValueError(f"Collection '{config['milvus_collection']}' not found in Milvus.")

    collection = Collection(config["milvus_collection"])
    collection.load()
    
    search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
    results = collection.search(data=[query_embedding], anns_field="embedding", param=search_params, limit=5, output_fields=["text"])
    search_results = [{"content": hit.entity.get("text", ""), "similarity": 1 - hit.distance} for hits in results for hit in hits]
    
    collection.release()
    connections.disconnect("default")

    pipelines_client.store_results({
        "stage_status": "success",
        "search_results_json": json.dumps(search_results)
    })
    print(f"✅ Stage 4: Vector search completed. Found {len(search_results)} results.")

except Exception as e:
    error_info = {"stage_status": "failed", "error_message": str(e), "traceback": traceback.format_exc()}
    pipelines_client.store_results(error_info)
    print(f"❌ Stage 4 failed: {e}")
    raise