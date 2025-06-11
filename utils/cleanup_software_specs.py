import os
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials, APIClient
import pandas as pd

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

print("=== Software Specifications 조회 및 삭제 도구 ===\n")

# 1. 모든 software specifications 조회
print("1. 현재 생성된 Software Specifications 조회 중...")
try:
    sw_specs_df = client.software_specifications.list()
    print(f"총 {len(sw_specs_df)} 개의 software specifications이 있습니다.\n")
    
    # 이름별로 정렬하여 표시
    if len(sw_specs_df) > 0:
        print("현재 Software Specifications:")
        print("-" * 80)
        for idx, row in sw_specs_df.iterrows():
            print(f"ID: {row['ID']}")
            print(f"Name: {row['NAME']}")
            print(f"Created: {row['CREATED']}")
            print(f"Description: {row.get('DESCRIPTION', 'N/A')}")
            print("-" * 80)
    
except Exception as e:
    print(f"Software specifications 조회 중 오류: {e}")

# 2. 특정 패턴으로 필터링 (예: statsmodels 관련)
print("\n2. 특정 패턴으로 필터링...")
filter_patterns = [
    "statsmodels",
    "bge-m3", 
    "bge-reranker",
    "software_spec"  # 일반적인 패턴
]

for pattern in filter_patterns:
    filtered_specs = sw_specs_df[sw_specs_df['NAME'].str.contains(pattern, case=False, na=False)]
    if len(filtered_specs) > 0:
        print(f"\n'{pattern}' 패턴과 일치하는 Software Specifications ({len(filtered_specs)}개):")
        for idx, row in filtered_specs.iterrows():
            print(f"  - {row['NAME']} (ID: {row['ID']})")

# 3. 삭제할 Software Specifications 선택
print("\n3. 삭제 옵션:")
print("a) 모든 custom software specifications 삭제")
print("b) 특정 패턴으로 삭제")
print("c) 개별 ID로 삭제")
print("d) 삭제하지 않음")

choice = input("\n선택하세요 (a/b/c/d): ").lower().strip()

if choice == 'a':
    # 모든 custom software specifications 삭제 (runtime 기본 스펙 제외)
    print("\n모든 custom software specifications 삭제 중...")
    custom_specs = sw_specs_df[~sw_specs_df['NAME'].str.startswith('runtime-', na=False)]
    
    if len(custom_specs) > 0:
        confirm = input(f"{len(custom_specs)}개의 custom software specifications을 삭제하시겠습니까? (y/N): ")
        if confirm.lower() == 'y':
            for idx, row in custom_specs.iterrows():
                try:
                    client.software_specifications.delete(row['ID'])
                    print(f"✅ 삭제됨: {row['NAME']} (ID: {row['ID']})")
                except Exception as e:
                    print(f"❌ 삭제 실패: {row['NAME']} - {e}")
        else:
            print("삭제가 취소되었습니다.")
    else:
        print("삭제할 custom software specifications이 없습니다.")

elif choice == 'b':
    # 특정 패턴으로 삭제
    pattern = input("삭제할 패턴을 입력하세요 (예: statsmodels): ")
    if pattern:
        pattern_specs = sw_specs_df[sw_specs_df['NAME'].str.contains(pattern, case=False, na=False)]
        
        if len(pattern_specs) > 0:
            print(f"\n'{pattern}' 패턴과 일치하는 Software Specifications:")
            for idx, row in pattern_specs.iterrows():
                print(f"  - {row['NAME']} (ID: {row['ID']})")
            
            confirm = input(f"\n{len(pattern_specs)}개를 삭제하시겠습니까? (y/N): ")
            if confirm.lower() == 'y':
                for idx, row in pattern_specs.iterrows():
                    try:
                        client.software_specifications.delete(row['ID'])
                        print(f"✅ 삭제됨: {row['NAME']} (ID: {row['ID']})")
                    except Exception as e:
                        print(f"❌ 삭제 실패: {row['NAME']} - {e}")
            else:
                print("삭제가 취소되었습니다.")
        else:
            print(f"'{pattern}' 패턴과 일치하는 software specifications이 없습니다.")

elif choice == 'c':
    # 개별 ID로 삭제
    spec_id = input("삭제할 Software Specification ID를 입력하세요: ")
    if spec_id:
        try:
            # ID로 상세 정보 조회
            spec_details = client.software_specifications.get_details(spec_id)
            spec_name = spec_details['entity']['name']
            
            confirm = input(f"'{spec_name}' (ID: {spec_id})를 삭제하시겠습니까? (y/N): ")
            if confirm.lower() == 'y':
                client.software_specifications.delete(spec_id)
                print(f"✅ 삭제됨: {spec_name} (ID: {spec_id})")
            else:
                print("삭제가 취소되었습니다.")
        except Exception as e:
            print(f"❌ 삭제 실패: {e}")

else:
    print("삭제하지 않습니다.")

# 4. 삭제 후 현재 상태 확인
print("\n4. 삭제 후 현재 상태:")
try:
    updated_sw_specs_df = client.software_specifications.list()
    print(f"현재 총 {len(updated_sw_specs_df)}개의 software specifications이 있습니다.")
    
    # Custom specifications만 표시
    custom_specs = updated_sw_specs_df[~updated_sw_specs_df['NAME'].str.startswith('runtime-', na=False)]
    if len(custom_specs) > 0:
        print(f"Custom software specifications: {len(custom_specs)}개")
        for idx, row in custom_specs.iterrows():
            print(f"  - {row['NAME']} (ID: {row['ID']})")
    else:
        print("Custom software specifications이 없습니다.")
        
except Exception as e:
    print(f"업데이트된 목록 조회 중 오류: {e}")

print("\n=== 작업 완료 ===")