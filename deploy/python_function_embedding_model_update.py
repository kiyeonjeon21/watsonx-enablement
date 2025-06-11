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
    Updated version with optimized configuration for production deployment
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
            
            # Get input text from payload
            sentences = payload['input_data'][0]['values']
            
            # Generate embeddings (only dense for simplicity and JSON compatibility)
            output = model.encode(
                sentences,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
                max_length=8192
            )
            
            # Convert to simple list format
            embeddings = output['dense_vecs']
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            # Slice embeddings to only first 5 dimensions for testing
            embeddings = [embedding[-10:] for embedding in embeddings]

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

def update_deployment(serving_name):
    """
    Update the function of a deployment with specific serving name
    """
    try:
        # 1. List all deployments to get their IDs
        deployments_df = client.deployments.list()

        # 2. Iterate through deployments to find the one with the matching serving_name
        existing_deployment = None
        for deployment_id in deployments_df['ID']:
            deployment_details = client.deployments.get_details(deployment_id)
            
            # Correct path to serving_name is entity.online.parameters.serving_name
            entity = deployment_details.get('entity', {})
            online_params = entity.get('online', {}).get('parameters', {})
            current_serving_name = online_params.get('serving_name')

            if current_serving_name and current_serving_name.strip() == serving_name:
                print(f"✅ Match found for deployment ID: {deployment_id}")
                existing_deployment = deployment_details
                break

        if not existing_deployment:
            raise ValueError(f"Could not find a deployment with serving_name: '{serving_name}'")
            
        existing_deployment_id = client.deployments.get_id(existing_deployment)
        print(f"Proceeding with update for deployment ID: {existing_deployment_id}")
        
        # Get existing hardware spec from the deployment
        existing_hw_spec = existing_deployment.get('entity', {}).get('hardware_spec', {})
        
        # Get software spec ID (assuming it's already created)
        custom_sw_spec_name = "bge-m3 software spec"
        sw_spec_id = client.software_specifications.get_id_by_name(custom_sw_spec_name)
        
        # Create new function
        function_name = f"bge-m3 embedding function update"
        meta_props = {
            client.repository.FunctionMetaNames.NAME: function_name,
            client.repository.FunctionMetaNames.SOFTWARE_SPEC_ID: sw_spec_id
        }
        
        # Store new function
        function_details = client.repository.store_function(meta_props=meta_props, function=deployable_callable)
        new_function_id = client.repository.get_function_id(function_details)
        
        # Update deployment with new function only (no other fields allowed when updating asset)
        update_changes = {
            'asset': {'id': new_function_id}
        }
        
        # Update deployment with new function
        updated_deployment = client.deployments.update(existing_deployment_id, update_changes)
        print("Deployment updated successfully")
        
        # Test the updated deployment
        test_texts = ["What is BGE M3?", "Definition of embedding model"]
        test_payload = {
            "input_data": [{
                "values": test_texts
            }]
        }
        
        test_result = client.deployments.score(existing_deployment_id, test_payload)
        print("Test result:", test_result)
        
        return existing_deployment_id
        
    except Exception as e:
        print(f"Error during deployment update: {e}")
        raise

if __name__ == "__main__":
    # Update deployment with specific serving name
    serving_name = "bgem3_embedding_model"
    deployment_id = update_deployment(serving_name)
    print(f"Updated deployment ID: {deployment_id}")