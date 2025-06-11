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

# Create deployable function for BGE-M3 embedding
def deployable_callable():
    """
    Deployable python function that generates embeddings using BGE-M3
    """
    def score(payload):
        """
        Score method that generates embeddings
        """
        try:
            from FlagEmbedding import BGEM3FlagModel
            import numpy as np
            
            # Initialize model
            model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
            
            # Get input text from payload - following Medium article pattern
            sentences = payload['input_data'][0]['values']
            
            # Generate embeddings (only dense for simplicity and JSON compatibility)
            output = model.encode(
                sentences,
                return_dense=True,
                return_sparse=False,  # Disable for now to avoid serialization issues
                return_colbert_vecs=False,  # Disable for now to avoid serialization issues
                max_length=8192
            )
            
            # Convert to simple list format following Medium article pattern
            embeddings = output['dense_vecs']
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            return {
                'predictions': [
                    {
                        'fields': ['sentence', 'embedding'],
                        'values': [[sentence, embedding] for sentence, embedding in zip(sentences, embeddings)]
                    }
                ]
            }
            
        except Exception as e:
            return {"error": str(e)}
        
    return score

# Test the function locally
test_texts = ["What is BGE M3?", "Definition of embedding model"]
test_payload = {
    "input_data": [{
        "values": test_texts
    }]
}
# print("Testing function locally...")
# try:
#     # deployable_callable()은 score 함수를 반환하므로, 이를 호출하고 테스트 페이로드를 전달
#     local_score_func = deployable_callable()
#     local_result = local_score_func(test_payload)
#     print("Local test result:", local_result)
    
#     # 결과 검증
#     if 'predictions' in local_result and len(local_result['predictions']) > 0:
#         print("Local test successful!")
#     else:
#         print("Warning: Local test returned unexpected format")
# except Exception as e:
#     print(f"Local test failed with error: {e}")


# Create conda environment configuration
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

with open("config.yaml", "w", encoding="utf-8") as f:
    f.write(config_yml)

# Get base software spec
base_sw_spec_id = client.software_specifications.get_id_by_name("runtime-24.1-py3.11")

# Create package extension
meta_prop_pkg_extn = {
    client.package_extensions.ConfigurationMetaNames.NAME: "bge-m3 env",
    client.package_extensions.ConfigurationMetaNames.DESCRIPTION: "Environment for BGE-M3 embedding model",
    client.package_extensions.ConfigurationMetaNames.TYPE: "conda_yml"
}

pkg_extn_details = client.package_extensions.store(meta_props=meta_prop_pkg_extn, file_path="config.yaml")
pkg_extn_id = client.package_extensions.get_id(pkg_extn_details)

# Create software specification
custom_sw_spec_name = "bge-m3 software spec"
try:
    # 이름으로 ID 조회 시도
    sw_spec_id = client.software_specifications.get_id_by_name(custom_sw_spec_name)
except:
    # 없는 경우에만 새로 생성
    meta_prop_sw_spec = {
        client.software_specifications.ConfigurationMetaNames.NAME: custom_sw_spec_name,
        client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "Software specification for BGE-M3",
        client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION: {"guid": base_sw_spec_id}
    }
    sw_spec_details = client.software_specifications.store(meta_props=meta_prop_sw_spec)
    sw_spec_id = client.software_specifications.get_id(sw_spec_details)

    # Add package extension to software specification
    client.software_specifications.add_package_extension(sw_spec_id, pkg_extn_id)

# Store the function
meta_props = {
    client.repository.FunctionMetaNames.NAME: "bge-m3 embedding function",
    client.repository.FunctionMetaNames.SOFTWARE_SPEC_ID: sw_spec_id
}

function_details = client.repository.store_function(meta_props=meta_props, function=deployable_callable)
function_id = client.repository.get_function_id(function_details)

# Deploy the function
hardware_spec_id = client.hardware_specifications.get_id_by_name('M')  # 4vCPU, 16GB RAM
deployment_metadata = {
    client.deployments.ConfigurationMetaNames.NAME: "BGE-M3 Embedding Deployment",
    client.deployments.ConfigurationMetaNames.ONLINE: {},
    client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: { 
        "id": hardware_spec_id, 
        'num_nodes': 1
    },
    client.deployments.ConfigurationMetaNames.SERVING_NAME : 'bgem3_embedding_model'
}


function_deployment = client.deployments.create(function_id, meta_props=deployment_metadata)

deployment_id = client.deployments.get_id(function_deployment)
print(f"Deployment ID: {deployment_id}")

# Test the deployed function
test_result = client.deployments.score(deployment_id, test_payload)
print("Test result:", test_result)