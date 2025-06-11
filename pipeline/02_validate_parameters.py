import os
import json
import traceback
from pymilvus import connections
from ibm_watson_studio_pipelines import WSPipelines

def main():
    """설정 파라미터를 검증하고 Milvus 연결을 테스트합니다. 실패 시 오류를 기록합니다."""
    api_key = os.getenv("API_KEY")
    pipelines_client = WSPipelines.from_apikey(apikey=api_key)

    try:
        config_json = os.getenv("config_json")
        if not config_json:
            raise ValueError("Input 'config_json' not found in environment variables.")
        config = json.loads(config_json)

        required = ["project_id", "watsonx_url", "milvus_host", "milvus_port", "milvus_user", "milvus_password"]
        missing = [r for r in required if not config.get(r)]
        if missing:
            raise ValueError(f"Missing required config parameters: {missing}")

        print("Connecting to Milvus for validation...")
        connections.connect(host=config["milvus_host"], port=config["milvus_port"], user=config["milvus_user"], password=config["milvus_password"], secure=True)
        connections.disconnect("default")
        print("Milvus connection validated.")

        pipelines_client.store_results({"stage_status": "success"})
        print("✅ Stage 2: Parameter validation completed successfully.")

    except Exception as e:
        error_info = {"stage_status": "failed", "error_message": str(e), "traceback": traceback.format_exc()}
        pipelines_client.store_results(error_info)
        print(f"❌ Stage 2 failed: {e}")
        raise

if __name__ == "__main__":
    main()