import os
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

# Create deployable function for BGE Reranker
def deployable_callable():
    """
    Deployable python function that performs reranking using BGE-Reranker-v2-m3
    """
    def score(payload):
        """
        Score method that computes relevance scores between queries and passages
        """
        try:
            from FlagEmbedding import FlagReranker
            import numpy as np
            
            # Initialize reranker model
            reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
            
            # Get input data from payload
            # Expected format: [["query1", "passage1"], ["query2", "passage2"], ...]
            query_passage_pairs = payload['input_data'][0]['values']
            
            # Compute relevance scores
            scores = reranker.compute_score(query_passage_pairs, normalize=True)  # normalize to 0-1 range
            
            # Handle both single pair and multiple pairs
            if isinstance(scores, (int, float)):
                scores = [scores]  # Convert single score to list
            elif isinstance(scores, np.ndarray):
                scores = scores.tolist()  # Convert numpy array to list
            
            # Create results with query-passage pairs and their scores
            results = []
            for i, (query_passage_pair, score) in enumerate(zip(query_passage_pairs, scores)):
                results.append({
                    'query': query_passage_pair[0],
                    'passage': query_passage_pair[1],
                    'relevance_score': float(score)
                })
            
            return {
                'predictions': [
                    {
                        'fields': ['query', 'passage', 'relevance_score'],
                        'values': [[result['query'], result['passage'], result['relevance_score']] for result in results]
                    }
                ]
            }
            
        except Exception as e:
            return {"error": str(e)}
        
    return score

# Test the function locally with reranker examples
test_query_passage_pairs = [
    ["what is panda?", "hi"],
    ["what is panda?", "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."],
    ["machine learning", "Machine learning is a subset of artificial intelligence that focuses on algorithms."],
    ["machine learning", "I like to eat pizza on weekends."]
]

test_payload = {
    "input_data": [{
        "values": test_query_passage_pairs
    }]
}

# Test the function locally
# print("Testing reranker function locally...")
# try:
#     # deployable_callable()은 score 함수를 반환하므로, 이를 호출하고 테스트 페이로드를 전달
#     local_score_func = deployable_callable()
#     local_result = local_score_func(test_payload)
#     print("Local test result:", local_result)
    
#     # 결과 검증
#     if 'predictions' in local_result and len(local_result['predictions']) > 0:
#         print("Local test successful!")
#         # 점수 결과 출력
#         print("\nLocal Reranking Results:")
#         print("-" * 60)
#         for values in local_result['predictions'][0]['values']:
#             query, passage, score = values
#             print(f"Query: {query}")
#             print(f"Passage: {passage[:50]}...")
#             print(f"Relevance Score: {score:.4f}")
#             print("-" * 60)
#     else:
#         print("Warning: Local test returned unexpected format")
# except Exception as e:
#     print(f"Local test failed with error: {e}")


# Create conda environment configuration (same as before)
config_yml = """
name: python311
channels:
  - pip
dependencies:
  - pip
  - pip:
    - FlagEmbedding
    - torch
    - transformers
    - sentence-transformers
"""

with open("config_reranker.yaml", "w", encoding="utf-8") as f:
    f.write(config_yml)

# Get base software spec
base_sw_spec_id = client.software_specifications.get_id_by_name("runtime-24.1-py3.11")

# Create package extension
meta_prop_pkg_extn = {
    client.package_extensions.ConfigurationMetaNames.NAME: "bge-reranker env",
    client.package_extensions.ConfigurationMetaNames.DESCRIPTION: "Environment for BGE-Reranker model",
    client.package_extensions.ConfigurationMetaNames.TYPE: "conda_yml"
}

pkg_extn_details = client.package_extensions.store(meta_props=meta_prop_pkg_extn, file_path="config_reranker.yaml")
pkg_extn_id = client.package_extensions.get_id(pkg_extn_details)

# Create software specification
custom_sw_spec_name = "bge-reranker software spec"
try:
    # 이름으로 ID 조회 시도
    sw_spec_id = client.software_specifications.get_id_by_name(custom_sw_spec_name)
except:
    # 없는 경우에만 새로 생성
    meta_prop_sw_spec = {
        client.software_specifications.ConfigurationMetaNames.NAME: custom_sw_spec_name,
        client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "Software specification for BGE-Reranker",
        client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION: {"guid": base_sw_spec_id}
    }

    sw_spec_details = client.software_specifications.store(meta_props=meta_prop_sw_spec)
    sw_spec_id = client.software_specifications.get_id(sw_spec_details)

    # Add package extension to software specification
    client.software_specifications.add_package_extension(sw_spec_id, pkg_extn_id)

# Store the function
meta_props = {
    client.repository.FunctionMetaNames.NAME: "bge-reranker function",
    client.repository.FunctionMetaNames.SOFTWARE_SPEC_ID: sw_spec_id
}

function_details = client.repository.store_function(meta_props=meta_props, function=deployable_callable)
function_id = client.repository.get_function_id(function_details)

# Deploy the function
hardware_spec_id = client.hardware_specifications.get_id_by_name('M')  # 4vCPU, 16GB RAM
deployment_metadata = {
    client.deployments.ConfigurationMetaNames.NAME: "BGE-Reranker Deployment",
    client.deployments.ConfigurationMetaNames.ONLINE: {},
    client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: { 
        "id": hardware_spec_id, 
        'num_nodes': 1
    },
    client.deployments.ConfigurationMetaNames.SERVING_NAME : 'bge_reranker_model'
}

function_deployment = client.deployments.create(function_id, meta_props=deployment_metadata)

deployment_id = client.deployments.get_id(function_deployment)
print(f"Deployment ID: {deployment_id}")

# Test the deployed function
test_result = client.deployments.score(deployment_id, test_payload)
print("Test result:", test_result)

# Print scores in a readable format
if 'predictions' in test_result:
    print("\nReranking Results:")
    print("-" * 80)
    for values in test_result['predictions'][0]['values']:
        query, passage, score = values
        print(f"Query: {query}")
        print(f"Passage: {passage[:60]}...")
        print(f"Relevance Score: {score:.4f}")
        print("-" * 80)