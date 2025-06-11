import os
import json
import traceback
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_studio_pipelines import WSPipelines

def main():
    """컨텍스트를 사용하여 최종 답변을 생성합니다. 실패 시 오류를 기록합니다."""
    api_key = os.getenv("API_KEY")
    pipelines_client = WSPipelines.from_apikey(apikey=api_key)
    
    try:
        config_json = os.getenv("config_json")
        user_query = os.getenv("user_query")
        reranked_results_json = os.getenv("reranked_results_json")
        
        if not all([config_json, user_query, reranked_results_json]):
            raise ValueError("One or more required inputs are missing from environment variables.")

        config = json.loads(config_json)
        reranked_results = json.loads(reranked_results_json)

        context_docs = reranked_results[:3]
        context = "\n\n".join([f"Document: {doc['content']}" for doc in context_docs])
        
        system_prompt = "Use the provided context to answer the user's question. If the answer is not in the context, say so."
        user_prompt = f"Context:\n{context}\n\nQuestion: {user_query}"
        
        credentials = Credentials(url=config["watsonx_url"], api_key=config["api_key"])
        model = ModelInference(model_id=config["generation_model"], credentials=credentials, project_id=config["project_id"], params={GenParams.MAX_NEW_TOKENS: 500})
        
        print("Generating final response...")
        response = model.chat([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
        generated_text = response['choices'][0]['message']['content']

        final_response = {"query": user_query, "generated_response": generated_text, "context_docs": context_docs}
        
        pipelines_client.store_results({
            "status": "success",
            "final_response_json": json.dumps(final_response)
        })
        print("✅ Stage 6: Response generation completed.")

    except Exception as e:
        error_info = {"status": "failed", "error_message": str(e), "traceback": traceback.format_exc()}
        pipelines_client.store_results(error_info)
        print(f"❌ Stage 6 failed: {e}")
        raise

if __name__ == "__main__":
    main()