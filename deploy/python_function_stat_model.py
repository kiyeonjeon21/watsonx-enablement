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

# Create deployable function for statsmodels
def deployable_callable():
    """
    Deployable python function that provides statistical descriptions using statsmodels
    """
    def score(payload):
        """
        Score method that computes statistical descriptions
        """
        try:
            from statsmodels.stats.descriptivestats import describe
            
            # Get input data from payload
            data = payload['input_data'][0]['values']
            
            # Compute statistical description
            description = describe(data)
            
            return {
                'predictions': [
                    {
                        'fields': ['statistical_description'],
                        'values': [[str(description)]]
                    }
                ]
            }
            
        except Exception as e:
            return {"error": str(e)}
        
    return score

# Test the function locally
import numpy as np

test_data = np.random.randn(10, 10).tolist()  # Convert to list for JSON compatibility
test_payload = {
    "input_data": [{
        "values": test_data
    }]
}

print("Testing statsmodels function locally...")
try:
    local_score_func = deployable_callable()
    local_result = local_score_func(test_payload)
    print("Local test result:", local_result)
    
    # 결과 검증
    if 'predictions' in local_result and len(local_result['predictions']) > 0:
        print("Local test successful!")
    else:
        print("Warning: Local test returned unexpected format")
except Exception as e:
    print(f"Local test failed with error: {e}")

# Create conda environment configuration
config_yml = """
name: python311
dependencies:
  - pip
  - pip:
    - statsmodels
    - numpy
"""

with open("config_stats.yaml", "w", encoding="utf-8") as f:
    f.write(config_yml)

# Get base software spec
base_sw_spec_id = client.software_specifications.get_id_by_name("runtime-24.1-py3.11")

# Create package extension
meta_prop_pkg_extn = {
    client.package_extensions.ConfigurationMetaNames.NAME: "statsmodels env",
    client.package_extensions.ConfigurationMetaNames.DESCRIPTION: "Environment with statsmodels",
    client.package_extensions.ConfigurationMetaNames.TYPE: "conda_yml"
}

pkg_extn_details = client.package_extensions.store(meta_props=meta_prop_pkg_extn, file_path="config_stats.yaml")
pkg_extn_id = client.package_extensions.get_id(pkg_extn_details)

# Create software specification
custom_sw_spec_name = "statsmodels software spec"
try:
    # 이름으로 ID 조회 시도
    sw_spec_id = client.software_specifications.get_id_by_name(custom_sw_spec_name)
except:
    # 없는 경우에만 새로 생성
    meta_prop_sw_spec = {
        client.software_specifications.ConfigurationMetaNames.NAME: custom_sw_spec_name,
        client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "Software specification for statsmodels",
        client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION: {"guid": base_sw_spec_id}
    }
    
    sw_spec_details = client.software_specifications.store(meta_props=meta_prop_sw_spec)
    sw_spec_id = client.software_specifications.get_id(sw_spec_details)
    
    # Add package extension to software specification
    client.software_specifications.add_package_extension(sw_spec_id, pkg_extn_id)

# Store the function
meta_props = {
    client.repository.FunctionMetaNames.NAME: "statsmodels function",
    client.repository.FunctionMetaNames.SOFTWARE_SPEC_ID: sw_spec_id
}

function_details = client.repository.store_function(meta_props=meta_props, function=deployable_callable)
function_id = client.repository.get_function_id(function_details)

# Deploy the function
hardware_spec_id = client.hardware_specifications.get_id_by_name('S')  # Small instance for stats
deployment_metadata = {
    client.deployments.ConfigurationMetaNames.NAME: "Statsmodels Deployment",
    client.deployments.ConfigurationMetaNames.ONLINE: {},
    client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: { 
        "id": hardware_spec_id, 
        'num_nodes': 1
    },
    client.deployments.ConfigurationMetaNames.SERVING_NAME : 'statsmodels_function'
}

function_deployment = client.deployments.create(function_id, meta_props=deployment_metadata)

deployment_id = client.deployments.get_id(function_deployment)
print(f"Deployment ID: {deployment_id}")

# Test the deployed function
test_result = client.deployments.score(deployment_id, test_payload)
print("Test result:", test_result)