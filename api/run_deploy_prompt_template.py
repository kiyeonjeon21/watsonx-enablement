import requests
from dotenv import load_dotenv
import os
import json
import pandas as pd
import random

# Load environment variables from .env file
load_dotenv()

# Set your API key here
API_KEY = os.getenv('API_KEY')

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

def load_rag_dataset():
    """Load RAG dataset from CSV file"""
    df = pd.read_csv('rag_dataset_1200_train_sample.csv')
    return df

def get_random_qa_pair():
    """Get a random Q&A pair from the dataset"""
    df = load_rag_dataset()
    random_row = df.iloc[random.randint(0, len(df)-1)]
    print("\n=== Selected Q&A Pair ===")
    print("\nContext:", random_row['context'])
    print("\nQuestion:", random_row['question'])
    print("\nExpected Answer:", random_row['answer'])
    print("\n" + "="*50)
    return random_row['context'], random_row['question'], random_row['answer']

def invoke_mode(iam_token):
    """Regular invoke mode - single response"""
    # scoring_url = "https://us-south.ml.cloud.ibm.com/ml/v1/deployments/llama3_3_70b_basic_prompt/text/generation?version=2021-05-01"
    scoring_url = "https://eu-de.ml.cloud.ibm.com/ml/v1/deployments/rag_dev_llama_3_3_70b_instruct/text/generation?version=2021-05-01"
    scoring_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {iam_token}"
    }
    context, question, answer = get_random_qa_pair()
    scoring_data = {
        "parameters": {
            "prompt_variables": {
                "contexts": context,
                "question": question
            }
        }
    }

    print("\n=== Invoke Mode ===")
    scoring_response = requests.post(scoring_url, headers=scoring_headers, json=scoring_data)
    print("Scoring Request Status Code:", scoring_response.status_code)
    if scoring_response.status_code == 200:
        print(scoring_response.json())
    else:
        print("Error in scoring request:", scoring_response.text)

def stream_mode_events(iam_token):
    """Stream mode with event details"""
    # scoring_url = "https://us-south.ml.cloud.ibm.com/ml/v1/deployments/llama3_3_70b_basic_prompt/text/generation_stream?version=2021-05-01"
    scoring_url = "https://eu-de.ml.cloud.ibm.com/ml/v1/deployments/rag_dev_llama_3_3_70b_instruct/text/generation_stream?version=2021-05-01"
    scoring_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {iam_token}"
    }
    context, question, answer = get_random_qa_pair()
    scoring_data = {
        "parameters": {
            "prompt_variables": {
                "contexts": context,
                "question": question
            }
        }
    }

    print("\n=== Stream Mode (Events) ===")
    with requests.post(scoring_url, headers=scoring_headers, json=scoring_data, stream=True) as scoring_response:
        print("Scoring Request Status Code:", scoring_response.status_code)
        if scoring_response.status_code == 200:
            for line in scoring_response.iter_lines():
                if line:
                    print(line.decode('utf-8'))
        else:
            print("Error in scoring request:", scoring_response.text)

def stream_mode_sentence(iam_token):
    """Stream mode with continuous sentence output"""
    # scoring_url = "https://us-south.ml.cloud.ibm.com/ml/v1/deployments/llama3_3_70b_basic_prompt/text/generation_stream?version=2021-05-01"
    scoring_url = "https://eu-de.ml.cloud.ibm.com/ml/v1/deployments/rag_dev_llama_3_3_70b_instruct/text/generation_stream?version=2021-05-01"
    scoring_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {iam_token}"
    }
    context, question, answer = get_random_qa_pair()
    scoring_data = {
        "parameters": {
            "prompt_variables": {
                "contexts": context,
                "question": question
            }
        }
    }

    print("\n=== Stream Mode (Sentence) ===")
    with requests.post(scoring_url, headers=scoring_headers, json=scoring_data, stream=True) as scoring_response:
        if scoring_response.status_code == 200:
            for line in scoring_response.iter_lines():
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
            print("Error in scoring request:", scoring_response.text)

def main():
    # Get IAM token
    iam_token = get_iam_token()
    
    # Run all modes
    invoke_mode(iam_token)
    stream_mode_events(iam_token)
    stream_mode_sentence(iam_token)

if __name__ == "__main__":
    main()