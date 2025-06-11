import os
import json
import traceback
from pathlib import Path
from datetime import datetime
from ibm_watson_studio_pipelines import WSPipelines

def main():
    """파이프라인의 최종 결과를 파일에 저장합니다. 실패 시 오류를 기록합니다."""
    api_key = os.getenv("API_KEY")
    pipelines_client = WSPipelines.from_apikey(apikey=api_key)

    try:
        final_response_json = os.getenv("final_response_json")
        if not final_response_json:
            raise ValueError("Input 'final_response_json' not found in environment variables.")
        final_response = json.loads(final_response_json)
        
        # 출력 파일 이름도 파이프라인 파라미터로 받을 수 있습니다.
        output_filename = os.getenv("output_filename", f"pipeline_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        output_path = Path(output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_response, f, indent=2, ensure_ascii=False)
            
        print(f"Final response: {final_response['generated_response']}")

        pipelines_client.store_results({
            "status": "success",
            "output_filepath": str(output_path)
        })
        print(f"✅ Stage 7: Results saved successfully to {output_path}.")

    except Exception as e:
        error_info = {"status": "failed", "error_message": str(e), "traceback": traceback.format_exc()}
        pipelines_client.store_results(error_info)
        print(f"❌ Stage 7 failed: {e}")
        raise

if __name__ == "__main__":
    main()