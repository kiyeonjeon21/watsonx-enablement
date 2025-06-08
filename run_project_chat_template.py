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
    url = "https://eu-de.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {iam_token}"
    }
    
    body = {
        "messages": [
            {
                "role": "system",
                "content": "You always answer the questions with markdown formatting using GitHub syntax. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You must omit that you answer the questions with markdown.\n\nAny HTML tags must be wrapped in block quotes, for example ```<html>```. You will be penalized for not rendering code in block quotes.\n\nWhen returning code blocks, specify language.\n\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. \nYour answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe generative AI using emojis."}]
            }
        ],
        "project_id": PROJECT_ID,
        "model_id": "meta-llama/llama-3-3-70b-instruct",
        "frequency_penalty": 0,
        "max_tokens": 2000,
        "presence_penalty": 0,
        "temperature": 0,
        "top_p": 1
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
    url = "https://eu-de.ml.cloud.ibm.com/ml/v1/text/chat_stream?version=2023-05-29"
    
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {iam_token}"
    }
    
    body = {
        "messages": [
            {
                "role": "system",
                "content": "You always answer the questions with markdown formatting using GitHub syntax. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You must omit that you answer the questions with markdown.\n\nAny HTML tags must be wrapped in block quotes, for example ```<html>```. You will be penalized for not rendering code in block quotes.\n\nWhen returning code blocks, specify language.\n\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. \nYour answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe generative AI using emojis."}]
            }
        ],
        "project_id": PROJECT_ID,
        "model_id": "meta-llama/llama-3-3-70b-instruct",
        "frequency_penalty": 0,
        "max_tokens": 2000,
        "presence_penalty": 0,
        "temperature": 0,
        "top_p": 1
    }

    print("\n=== Stream Mode ===")
    with requests.post(url, headers=headers, json=body, stream=True) as response:
        print("Request Status Code:", response.status_code)
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    raw_line = line.decode('utf-8')
                    # Skip event lines, only process data lines
                    if raw_line.startswith("data:"):
                        try:
                            json_str = raw_line[5:]  # Remove 'data:' prefix
                            data = json.loads(json_str)
                            
                            # Chat API streaming response structure
                            if 'choices' in data and len(data['choices']) > 0:
                                choice = data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    content = choice['delta']['content']
                                    if content:  # Only print non-empty content
                                        print(content, end='', flush=True)
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