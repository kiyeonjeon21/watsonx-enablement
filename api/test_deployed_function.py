import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# IBM Cloud API 설정
API_KEY = os.getenv("API_KEY")
DEPLOYMENT_ID = "049a72e5-9c15-45f5-91be-3f0558717d5a"  # 실제 deployment ID로 변경 필요
ML_ENDPOINT = "https://us-south.ml.cloud.ibm.com"

def get_iam_token(api_key):
    """IBM Cloud IAM 토큰 획득"""
    url = "https://iam.cloud.ibm.com/identity/token"
    
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }
    
    response = requests.post(url, headers=headers, data=data, verify=False)
    
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"토큰 획득 실패: {response.status_code} - {response.text}")

def score_model(iam_token, deployment_id, payload):
    """모델 스코어링"""
    url = f"{ML_ENDPOINT}/ml/v4/deployments/{deployment_id}/predictions?version=2021-05-01"
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {iam_token}"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"스코어링 실패: {response.status_code} - {response.text}")

# 테스트 실행
if __name__ == "__main__":
    try:
        # 1. IAM 토큰 획득
        print("IAM 토큰 획득 중...")
        iam_token = get_iam_token(API_KEY)
        print("✓ 토큰 획득 성공")
        
        # 2. 스코어링 페이로드 준비 (statsmodels 함수용)
        scoring_payload = {
            "input_data": [{
                "values": [
                    [1.5, 2.3, 0.8, 3.1, 1.2],
                    [2.1, 1.9, 2.5, 0.7, 3.4],
                    [1.8, 2.8, 1.1, 2.9, 0.9],
                    [3.2, 1.4, 2.7, 1.6, 2.2],
                    [0.9, 3.1, 1.4, 2.3, 1.7]
                ]
            }]
        }
        
        # 3. 모델 스코어링
        print("모델 스코어링 중...")
        result = score_model(iam_token, DEPLOYMENT_ID, scoring_payload)
        
        print("✓ 스코어링 성공!")
        print("결과:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")