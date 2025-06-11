import os
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials, APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.prompts import PromptTemplateManager, PromptTemplate
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods, PromptTemplateFormats
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Load environment variables
load_dotenv()

def initialize_credentials():
    """Initialize WatsonX credentials from environment variables"""
    return Credentials(
        url=os.getenv("WATSONX_URL"),
        api_key=os.getenv("API_KEY")
    )

def create_loan_prompt_template(project_id):
    """Create a prompt template for loan-related queries"""
    credentials = initialize_credentials()
    prompt_mgr = PromptTemplateManager(credentials=credentials, project_id=project_id)
    
    loan_prompt = PromptTemplate(
        name="Loan Assistant",
        model_id="ibm/granite-3-3-8b-instruct",
        model_params={
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 250,
            GenParams.MIN_NEW_TOKENS: 50,
            GenParams.TEMPERATURE: 0.7
        },
        description="AI assistant for loan-related queries",
        task_ids=["generation"],
        input_variables=["question"],
        instruction="You are a helpful financial advisor. Provide clear and concise answers to loan-related questions.",
        input_prefix="Human",
        output_prefix="Assistant",
        input_text="{question}",
        examples=[
            ["What are the main types of loans available?", 
             "The main types of loans include: 1) Personal loans 2) Mortgage loans 3) Auto loans 4) Student loans 5) Business loans. Each type serves different purposes and comes with specific terms and requirements."],
            ["How does loan interest work?",
             "Loan interest is the cost of borrowing money, expressed as a percentage rate. It can be fixed or variable, and is calculated based on the principal amount, loan term, and interest rate type. The total repayment includes both principal and interest."]
        ]
    )
    
    return prompt_mgr.store_prompt(prompt_template=loan_prompt)

def create_deployment(project_id, prompt_template):
    """Create a deployment for the prompt template"""
    credentials = initialize_credentials()
    client = APIClient(wml_credentials=credentials)

    space_id = os.getenv("SPACE_ID")

    # Promote prompt template to space
    promoted_prompt_id = client.spaces.promote(
        asset_id=prompt_template.prompt_id,
        source_project_id=project_id,
        target_space_id=space_id
    )

    print(f"Promoted prompt template to space with ID: {promoted_prompt_id}")

    client.set.default_space(space_id)

    # The SDK may not correctly form the deployment payload for a prompt_template asset type.
    # We explicitly construct the 'asset' part of the payload with the required href.
    deployment_config = {
        client.deployments.ConfigurationMetaNames.NAME: "Loan Assistant Deployment",
        "asset": {
            "href": f"/ml/v4/prompt_templates/{promoted_prompt_id}?space_id={space_id}"
        },
        client.deployments.ConfigurationMetaNames.ONLINE: {},
        client.deployments.ConfigurationMetaNames.BASE_MODEL_ID: "ibm/granite-3-3-8b-instruct"
    }
    
    # We still pass artifact_id to satisfy the method's signature,
    # but the 'asset' key in meta_props should override the default behavior.
    return client.deployments.create(
        artifact_id=promoted_prompt_id,
        meta_props=deployment_config,
        space_id=space_id
    )

def generate_loan_response(project_id, deployment_id, question):
    """Generate response using the deployed model"""
    credentials = initialize_credentials()

    # Space ID를 사용하도록 수정
    space_id = os.getenv("SPACE_ID")

    model_inference = ModelInference(
        deployment_id=deployment_id,
        credentials=credentials,
        space_id=space_id
    )
    
    return model_inference.generate_text(
        params={
            "prompt_variables": {"question": question},
            GenParams.DECODING_METHOD: "greedy",
            GenParams.STOP_SEQUENCES: ['\n\n'],
            GenParams.MAX_NEW_TOKENS: 250
        }
    )

def main():
    # Your project ID from WatsonX
    project_id = os.getenv("PROJECT_ID")
    
    # Create prompt template
    prompt_template = create_loan_prompt_template(project_id)
    print(f"Created prompt template with ID: {prompt_template.prompt_id}")
    
    # Create deployment
    deployment_details = create_deployment(project_id, prompt_template)
    deployment_id = deployment_details.get("metadata", {}).get("id")
    print(f"Created deployment with ID: {deployment_id}")
    
    # Example usage
    sample_question = "What factors should I consider before taking a personal loan?"
    response = generate_loan_response(project_id, deployment_id, sample_question)
    print("\nSample Question:", sample_question)
    print("Response:", response)

if __name__ == "__main__":
    main()