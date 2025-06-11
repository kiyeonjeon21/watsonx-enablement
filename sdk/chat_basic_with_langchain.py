"""
Basic Chat Example using LangChain's ChatWatsonx integration.

Based on: https://python.langchain.com/docs/integrations/chat/ibm_watsonx/
Requires: pip install langchain-ibm langchain-core python-dotenv
"""

from langchain_ibm import ChatWatsonx
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

# --- Initialization ---
print("Loading environment variables...")
load_dotenv()

# Note: LangChain's ChatWatsonx uses API_KEY by default if credentials aren't passed.
# Ensure API_KEY and PROJECT_ID are in your .env file.
API_KEY = os.getenv("API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")
URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

if not API_KEY or not PROJECT_ID:
    print("Error: API_KEY and PROJECT_ID must be set in the .env file.")
    exit(1)

print("Initializing ChatWatsonx...")
# Initialize ChatWatsonx - credentials are automatically picked up from
# environment variables (WATSONX_APIKEY) if not explicitly passed,
# but we pass them here for clarity based on the env var name we use.
# The constructor directly takes project_id and url.
chat = ChatWatsonx(
    model_id="meta-llama/llama-3-3-70b-instruct", # Example model
    url=URL,
    project_id=PROJECT_ID,
    apikey=API_KEY,
    params={ # Optional parameters
        "max_new_tokens": 250,
        "temperature": 0.7,
    }
)
# --- End of Initialization ---


# --- Basic Invocation Example ---
print("\n=== Basic Invocation Example ===")

system_message = SystemMessage(
    content="You are a helpful assistant which replies briefly."
)
human_message = HumanMessage(content="Hi, how are you?")

print("\nInput Messages:")
print(f"- System: {system_message.content}")
print(f"- Human: {human_message.content}")

try:
    print("\nInvoking chat model...")
    ai_response = chat.invoke([system_message, human_message])

    print("\nAI Response:")
    print(f"- Content: {ai_response.content}")
    # print(f"- Full Response Object: {ai_response}") # Uncomment for more details

except Exception as e:
    print(f"\nAn error occurred during invocation: {str(e)}")

print("\n--- END OF BASIC CHAT EXAMPLE ---")