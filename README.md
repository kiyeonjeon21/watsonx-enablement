# watsonx-enablement

## 환경 설정

1. Python 가상환경 생성
```bash
python -m venv .venv
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

3. 환경변수 설정
`.env.copy` 파일을 `.env`로 복사한 후, 아래 값들을 본인의 환경에 맞게 수정하세요:
```env
API_KEY=your_ibm_cloud_api_key
PROJECT_ID=your_watsonx_project_id
WATSONX_URL=your_watsonx_url
```

## RAG 데이터셋 준비

1. Hugging Face 데이터셋 다운로드
```bash
git clone https://huggingface.co/datasets/neural-bridge/rag-dataset-1200
```

2. 데이터 형식 변환
```bash
python convert_parquet_to_csv.py
```

## watsonx.ai 기능

### 프롬프트 템플릿 관리
```bash
python prompt_template_manager.py 
```
- 프로젝트에 프롬프트 템플릿 등록 및 배포
- 대출 상담 시나리오 템플릿 예제 포함

### 프로젝트 기반 LLM API 호출
```bash
python run_project_prompt_template.py
```
- 프로젝트에 등록된 프롬프트 템플릿으로 LLM 호출
- 일반 호출 및 스트리밍 방식 지원

### 프로젝트 기반 Chat API 호출
```bash
python run_project_chat_template.py
```
- Chat API를 활용한 대화형 LLM 인터페이스
- 일반 호출 및 스트리밍 방식 지원

### 배포 공간 기반 LLM API 호출
```bash
python run_deploy_prompt_template.py
```
- 배포된 프롬프트 템플릿 기반 LLM 호출
- RAG 데이터셋 기반 질의응답 기능 구현

### 문서 기반 질의응답 예제
- PDF 문서 기반 질의응답 기능 구현
- 예제 문서: data/2024 Academy Awards Summary.PDF

샘플 질문:
- "Who won Best Actor?" - 문서 내 실제 수상자 정보 확인
- "How many awards did Oppenheimer win?" - 특정 영화의 수상 내역 집계
- "Who won the award for brand new superstar?" - 존재하지 않는 상에 대한 질문 예시 (잘못된 질문)