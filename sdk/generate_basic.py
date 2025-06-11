"""
Basic Text Generation Example with watsonx.ai

This script demonstrates basic text generation using the Watsonx.ai Python SDK,
extracted from the original generate_text.py example.
"""

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from dotenv import load_dotenv
import os

# --- Initialization (Adapted from generate_text.py) ---
print("Loading environment variables...")
load_dotenv()

API_KEY = os.getenv("API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
SERVICE_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

if not API_KEY or not PROJECT_ID:
    print("Error: API_KEY and PROJECT_ID must be set in the .env file.")
    exit(1)

print("Setting up credentials...")
credentials = Credentials(
    url=SERVICE_URL,
    api_key=API_KEY
)

# Define generation parameters
generate_params = {
    GenParams.MAX_NEW_TOKENS: 250
}

# Initialize the model inference object for text generation
MODEL_ID = "meta-llama/llama-3-3-70b-instruct" # Or another suitable model
print(f"Initializing model: {MODEL_ID}...")
text_gen_model = ModelInference(
    model_id=MODEL_ID,
    params=generate_params,
    credentials=credentials,
    project_id=PROJECT_ID
)
# --- End of Initialization ---


# --- Text Generation Example (Section 8 from generate_text.py) ---
print("\n=== Basic Text Generation Example ===")

prompt = "What is IBM watsonx.ai?"
print(f"\nInput Prompt: \"{prompt}\"")

try:
    print("\nSending generation request...")
    # Use the generate method
    response = text_gen_model.generate(prompt=prompt)

    print("\nGenerated Text:")
    if response and "results" in response and response["results"]:
        generated_text = response['results'][0].get('generated_text', None)
        if generated_text:
            print(generated_text)
        else:
            print("Could not extract generated text from response.")
            print("Full response:", response)
    else:
        print("Received an unexpected response format:")
        print(response)

except Exception as e:
    print(f"\nAn error occurred during text generation: {str(e)}")

print("\n--- END OF BASIC TEXT GENERATION EXAMPLE ---")