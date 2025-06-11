import os
import json
import traceback
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watson_studio_pipelines import WSPipelines

def main():
    """사용자 쿼리의 임베딩을 생성합니다. 실패 시 오류를 기록합니다."""
    api_key = os.getenv("API_KEY")
    pipelines_client = WSPipelines.from_apikey(apikey=api_key)
    
    try:
        config_json = os.getenv("config_json")
        if not config_json:
            raise ValueError("Input 'config_json' not found in environment variables.")
        config = json.loads(config_json)
        user_query = os.getenv("user_query", "What is Artificial Intelligence?") 

        credentials = Credentials(url=config["watsonx_url"], api_key=config["api_key"])
        embeddings = Embeddings(model_id=config["embedding_model"], credentials=credentials, project_id=config["project_id"])
        
        print(f"Generating embedding for query: '{user_query}'")
        query_embedding = embeddings.embed_query(text=user_query)
        
        pipelines_client.store_results({
            "stage_status": "success",
            "user_query": user_query,
            "query_embedding_json": json.dumps(query_embedding),
            "embedding_dim": len(query_embedding)
        })
        print(f"✅ Stage 3: Query embedding completed. Vector length: {len(query_embedding)}")

    except Exception as e:
        error_info = {"stage_status": "failed", "error_message": str(e), "traceback": traceback.format_exc()}
        pipelines_client.store_results(error_info)
        print(f"❌ Stage 3 failed: {e}")
        raise

if __name__ == "__main__":
    main()