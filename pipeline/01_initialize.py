import os
import json
import traceback
from dotenv import load_dotenv
from ibm_watson_studio_pipelines import WSPipelines

def main():
    """환경 변수를 로드하고 파이프라인 설정을 구성합니다. 실패 시 오류를 기록합니다."""
    # 클라이언트를 먼저 초기화하여 오류 발생 시에도 결과를 저장할 수 있도록 합니다.
    api_key = os.getenv("API_KEY")
    if not api_key:
        # 이 단계에서는 클라이언트 초기화도 불가능하므로, 로그만 남기고 예외를 발생시킵니다.
        print("❌ CRITICAL: API_KEY 환경 변수가 설정되지 않았습니다.")
        raise ValueError("API_KEY environment variable is not set.")
    
    pipelines_client = WSPipelines.from_apikey(apikey=api_key)
    
    try:
        load_dotenv()
        
        config = {
            "api_key": api_key,
            "project_id": os.getenv("PROJECT_ID"),
            "watsonx_url": os.getenv("WATSONX_URL"),
            "milvus_host": os.getenv("MILVUS_HOST"),
            "milvus_port": os.getenv("MILVUS_PORT"),
            "milvus_user": os.getenv("MILVUS_USERNAME"),
            "milvus_password": os.getenv("MILVUS_PASSWORD"),
            "milvus_collection": os.getenv("MILVUS_COLLECTION"),
            "embedding_model": os.getenv("WATSONX_EMBEDDING_MODEL", "ibm/granite-embedding-278m-multilingual"),
            "generation_model": "meta-llama/llama-3-3-70b-instruct"
        }

        pipelines_client.store_results({
            "status": "success",
            "config_json": json.dumps(config)
        })
        
        print("✅ Stage 1: Initialization completed successfully.")

    except Exception as e:
        error_info = {
            "status": "failed",
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        pipelines_client.store_results(error_info)
        print(f"❌ Stage 1 failed: {e}")
        raise

if __name__ == "__main__":
    main()