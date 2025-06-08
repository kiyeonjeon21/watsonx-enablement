import requests
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env file
load_dotenv()

# Set your API key here
API_KEY = os.getenv('API_KEY')
PROJECT_ID = os.getenv('PROJECT_ID') 

def get_iam_token():
    """Get IAM token from IBM Cloud"""
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": API_KEY
    }

    response = requests.post(url, headers=headers, data=data)
    print("IAM Token Request Status Code:", response.status_code)
    if response.status_code == 200:
        return response.json().get('access_token')
    else:
        print("Error getting IAM token:", response.text)
        exit(1)

def invoke_mode(iam_token):
    """Regular invoke mode - single response"""
    url = "https://eu-de.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {iam_token}"
    }
    
    body = {
        "input": """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is the capital of the France?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
""",
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 200,
            "min_new_tokens": 0,
            "repetition_penalty": 1
        },
        "model_id": "meta-llama/llama-3-3-70b-instruct",
        "project_id": PROJECT_ID
    }

    print("\n=== Invoke Mode ===")
    response = requests.post(url, headers=headers, json=body)
    print("Request Status Code:", response.status_code)
    
    if response.status_code == 200:
        print(response.json())
    else:
        print("Error in request:", response.text)

def stream_mode(iam_token):
    """Stream mode with continuous sentence output"""
    url = "https://eu-de.ml.cloud.ibm.com/ml/v1/text/generation_stream?version=2023-05-29"
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {iam_token}"
    }
    
    body = {
        "input": """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is the capital of the France?
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
""",
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 200,
            "min_new_tokens": 0,
            "repetition_penalty": 1
        },
        "model_id": "meta-llama/llama-3-3-70b-instruct",
        "project_id": PROJECT_ID
    }

    print("\n=== Stream Mode ===")
    with requests.post(url, headers=headers, json=body, stream=True) as response:
        print("Request Status Code:", response.status_code)
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    raw_line = line.decode('utf-8')
                    if raw_line.startswith("data:"):
                        try:
                            json_str = raw_line[5:]  # Remove 'data:' prefix
                            data = json.loads(json_str)
                            if 'results' in data:
                                text = data['results'][0]['generated_text']
                                print(text, end='', flush=True)
                        except json.JSONDecodeError:
                            continue
            print()  # Final newline
        else:
            print("Error in request:", response.text)

def main():
    # Get IAM token
    iam_token = get_iam_token()
    
    # Run both modes
    invoke_mode(iam_token)
    stream_mode(iam_token)

if __name__ == "__main__":
    main()