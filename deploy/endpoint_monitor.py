import os
import time
from datetime import datetime
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials, APIClient

# Load environment variables
load_dotenv()

# Initialize credentials and client
credentials = Credentials(
    api_key=os.getenv("API_KEY"),
    url=os.getenv("WATSONX_URL")
)
client = APIClient(credentials)

# Set default space
space_id = os.getenv("SPACE_ID")
client.set.default_space(space_id)

def find_deployment_by_serving_name(serving_name):
    """
    serving_name으로 배포 ID 찾기
    """
    deployments_df = client.deployments.list()
    
    for deployment_id in deployments_df['ID']:
        deployment_details = client.deployments.get_details(deployment_id)
        entity = deployment_details.get('entity', {})
        online_params = entity.get('online', {}).get('parameters', {})
        current_serving_name = online_params.get('serving_name')

        if current_serving_name and current_serving_name.strip() == serving_name:
            return deployment_id
    
    raise ValueError(f"Could not find deployment with serving_name: '{serving_name}'")

def monitor_endpoint(serving_name, interval_seconds=2):
    """
    엔드포인트를 지속적으로 모니터링
    """
    try:
        # 배포 ID 찾기
        deployment_id = find_deployment_by_serving_name(serving_name)
        print(f"🎯 Found deployment ID: {deployment_id}")
        print(f"🚀 Starting continuous monitoring (interval: {interval_seconds}s)")
        print("Press Ctrl+C to stop\n")
        
        test_payload = {
            "input_data": [{
                "values": ["Test sentence for continuous monitoring"]
            }]
        }
        
        test_count = 0
        error_count = 0
        start_time = time.time()
        
        while True:
            try:
                test_count += 1
                current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # 밀리초까지
                
                # 요청 시작 시간
                request_start = time.time()
                
                # 엔드포인트 호출
                result = client.deployments.score(deployment_id, test_payload)
                
                # 응답 시간 계산
                response_time = (time.time() - request_start) * 1000  # ms
                
                # 결과 확인
                if 'predictions' in result and result['predictions']:
                    embedding = result['predictions'][0]['values'][0][1]
                    embedding_length = len(embedding)
                    first_5_values = embedding[:5] if len(embedding) >= 5 else embedding
                    
                    print(f"[{current_time}] #{test_count:3d}: ✅ OK ({response_time:.0f}ms) | dims: {embedding_length} | first_5: {[f'{x:.3f}' for x in first_5_values]}")
                else:
                    print(f"[{current_time}] #{test_count:3d}: ⚠️  Unexpected response format")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                error_count += 1
                current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{current_time}] #{test_count:3d}: ❌ ERROR ({str(e)[:50]}...)")
            
            time.sleep(interval_seconds)
        
        # 통계 출력
        total_time = time.time() - start_time
        print(f"\n📊 Monitoring Summary:")
        print(f"   Duration: {total_time:.1f} seconds")
        print(f"   Total requests: {test_count}")
        print(f"   Errors: {error_count}")
        print(f"   Success rate: {((test_count - error_count) / test_count * 100):.1f}%")
        print(f"   Average interval: {total_time / test_count:.2f}s")
        
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")

if __name__ == "__main__":
    serving_name = "bgem3_embedding_model"  # 여기서 serving_name 설정
    interval = 1  # 1초마다 테스트
    
    print(f"🎯 Target serving name: {serving_name}")
    print(f"⏱️  Test interval: {interval} seconds")
    
    monitor_endpoint(serving_name, interval)