import os
import json
import traceback
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Rerank
from ibm_watson_studio_pipelines import WSPipelines


"""검색 결과를 재정렬합니다. 실패 시 오류를 기록합니다."""
api_key = os.getenv("API_KEY")
pipelines_client = WSPipelines.from_apikey(apikey=api_key)

try:
    config_json = os.getenv("config_json")
    user_query = os.getenv("user_query")
    search_results_json = os.getenv("search_results_json")
    
    if not all([config_json, user_query, search_results_json]):
        raise ValueError("One or more required inputs are missing from environment variables.")
    
    config = json.loads(config_json)
    search_results = json.loads(search_results_json)
    reranked_results = []

    if not search_results:
        print("No results to rerank, skipping.")
    else:
        try:
            credentials = Credentials(url=config["watsonx_url"], api_key=config["api_key"])
            rerank = Rerank(model_id="cross-encoder/ms-marco-minilm-l-12-v2", credentials=credentials, project_id=config["project_id"])
            documents = [result["content"] for result in search_results]
            
            print(f"Reranking {len(documents)} documents...")
            rerank_response = rerank.generate(query=user_query, inputs=documents)
            
            for i, rerank_item in enumerate(rerank_response.get("results", [])):
                if i < len(search_results):
                    search_results[i]["rerank_score"] = rerank_item.get("score", 0)
            reranked_results = sorted(search_results, key=lambda x: x.get("rerank_score", 0), reverse=True)
        except Exception as e:
            print(f"⚠️ Reranking process failed, using original search results: {str(e)}")
            reranked_results = search_results

    pipelines_client.store_results({
        "status": "success",
        "reranked_results_json": json.dumps(reranked_results)
    })
    print(f"✅ Stage 5: Reranking completed. {len(reranked_results)} results processed.")

except Exception as e:
    error_info = {"status": "failed", "error_message": str(e), "traceback": traceback.format_exc()}
    pipelines_client.store_results(error_info)
    print(f"❌ Stage 5 failed: {e}")
    raise