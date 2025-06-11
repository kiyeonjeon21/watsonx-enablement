import os
from ibm_watsonx_ai import APIClient, Credentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

credentials = Credentials(
    url=os.getenv("WATSONX_URL"),
    api_key=os.getenv("API_KEY")
)
client = APIClient(credentials)

# Connecting to a space
space_id = os.getenv("SPACE_ID")
client.set.default_space(space_id)

# Promote asset(s) to space
source_project_id = os.getenv("PROJECT_ID")
vector_index_id = client.spaces.promote("816466bd-ed55-41b8-af0d-3fa5dddb6eaa", source_project_id, space_id)
print(vector_index_id)

# Define the function
params = {
    "space_id": space_id, 
    "vector_index_id": vector_index_id,
    "watsonx_url": os.getenv("WATSONX_URL")
}

def gen_ai_service(context, params = params, **custom):
    # import dependencies
    import json
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai.foundation_models.utils import Tool, Toolkit
    from ibm_watsonx_ai import APIClient, Credentials
    import os
    import requests
    import re

    vector_index_id = params.get("vector_index_id")
    space_id = params.get("space_id")
    service_url = params.get("watsonx_url")

    def proximity_search( query, api_client ):
        document_search_tool = Toolkit(
            api_client=api_client
        ).get_tool("RAGQuery")

        config = {
            "vectorIndexId": vector_index_id,
            "spaceId": space_id
        }

        results = document_search_tool.run(
            input=query,
            config=config
        )

        return results.get("output")


    def get_api_client(context):
        credentials = Credentials(
            url=service_url,
            token=context.get_token()
        )

        api_client = APIClient(
            credentials = credentials,
            space_id = space_id
        )

        return api_client

    def inference_model( messages, context, stream ):
        query = messages[-1].get("content")
        api_client = get_api_client(context)

        grounding_context = proximity_search(query, api_client)

        grounding = grounding_context
        messages.insert(0, {
            "role": f"system",
            "content": f"""{grounding}

You always answer the questions with markdown formatting. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You must omit that you answer the questions with markdown.

Any HTML tags must be wrapped in block quotes, for example ```<html>```. You will be penalized for not rendering code in block quotes.

When returning code blocks, specify language.

Given the document and the current conversation between a user and an assistant, your task is as follows: answer any user query by using information from the document. Always answer as helpfully as possible, while being safe. When the question cannot be answered using the context or document, output the following response: "I cannot answer that question based on the provided document.".

Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

"""
        })
        model_id = "meta-llama/llama-3-3-70b-instruct"
        parameters =  {
            "frequency_penalty": 0,
            "max_tokens": 2000,
            "presence_penalty": 0,
            "temperature": 0,
            "top_p": 1
        }
        model = ModelInference(
            model_id = model_id,
            api_client = api_client,
            params = parameters,
        )
        # Generate grounded response
        if (stream == True):
            generated_response = model.chat_stream(messages=messages)
        else:
            generated_response = model.chat(messages=messages)

        return generated_response


    def generate(context):
        payload = context.get_json()
        messages = payload.get("messages")
        
        # Grounded inferencing
        generated_response = inference_model(messages, context, False)

        execute_response = {
            "headers": {
                "Content-Type": "application/json"
            },
            "body": generated_response
        }

        return execute_response

    def generate_stream(context):
        payload = context.get_json()
        messages = payload.get("messages")

        # Grounded inferencing
        response_stream = inference_model(messages, context, True)

        for chunk in response_stream:
            yield chunk

    return generate, generate_stream

# Test locally
from ibm_watsonx_ai.deployments import RuntimeContext

context = RuntimeContext(api_client=client)

streaming = False
findex = 1 if streaming else 0
local_function = gen_ai_service(context, vector_index_id=vector_index_id, space_id=space_id)[findex]
messages = []
local_question = "Change this question to test your function"

messages.append({ "role" : "user", "content": local_question })

context = RuntimeContext(api_client=client, request_payload_json={"messages": messages})

response = local_function(context)

result = ''

if (streaming):
    for chunk in response:
        if (len(chunk["choices"])):
            print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
else:
    print(response)

# Store and deploy the AI Service
config_yml = """
name: python311
channels:
  - pip
dependencies:
  - pip
  - pip:
    - ibm-watsonx-ai>=1.3.0,<1.4.0
    - duckduckgo-search
    - wikipedia
    - langchain==0.3.20
    - langgraph==0.2.76
"""

with open("config.yaml", "w", encoding="utf-8") as f:
    f.write(config_yml)

# Get base software spec
base_sw_spec_id = client.software_specifications.get_id_by_name("runtime-24.1-py3.11")

# Create package extension
meta_prop_pkg_extn = {
    client.package_extensions.ConfigurationMetaNames.NAME: "rag-inmemory-env",
    client.package_extensions.ConfigurationMetaNames.DESCRIPTION: "Environment for RAG In-Memory Service",
    client.package_extensions.ConfigurationMetaNames.TYPE: "conda_yml"
}

pkg_extn_details = client.package_extensions.store(meta_props=meta_prop_pkg_extn, file_path="config.yaml")
pkg_extn_id = client.package_extensions.get_id(pkg_extn_details)

# Create software specification
custom_sw_spec_name = "rag-inmemory-software-spec"
try:
    software_spec_id = client.software_specifications.get_id_by_name(custom_sw_spec_name)
except:
    meta_prop_sw_spec = {
        client.software_specifications.ConfigurationMetaNames.NAME: custom_sw_spec_name,
        client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "Software specification for RAG In-Memory Service",
        client.software_specifications.ConfigurationMetaNames.BASE_SOFTWARE_SPECIFICATION: {"guid": base_sw_spec_id}
    }
    sw_spec_details = client.software_specifications.store(meta_props=meta_prop_sw_spec)
    software_spec_id = client.software_specifications.get_id(sw_spec_details)

    # Add package extension to software specification
    client.software_specifications.add_package_extension(software_spec_id, pkg_extn_id)

# Define the request and response schemas for the AI service
request_schema = {
    "application/json": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "messages": {
                "title": "The messages for this chat session.",
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "title": "The role of the message author.",
                            "type": "string",
                            "enum": ["user","assistant"]
                        },
                        "content": {
                            "title": "The contents of the message.",
                            "type": "string"
                        }
                    },
                    "required": ["role","content"]
                }
            }
        },
        "required": ["messages"]
    }
}

response_schema = {
    "application/json": {
        "oneOf": [{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","description":"AI Service response for /ai_service_stream","properties":{"choices":{"description":"A list of chat completion choices.","type":"array","items":{"type":"object","properties":{"index":{"type":"integer","title":"The index of this result."},"delta":{"description":"A message result.","type":"object","properties":{"content":{"description":"The contents of the message.","type":"string"},"role":{"description":"The role of the author of this message.","type":"string"}},"required":["role"]}}}}},"required":["choices"]},{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","description":"AI Service response for /ai_service","properties":{"choices":{"description":"A list of chat completion choices","type":"array","items":{"type":"object","properties":{"index":{"type":"integer","description":"The index of this result."},"message":{"description":"A message result.","type":"object","properties":{"role":{"description":"The role of the author of this message.","type":"string"},"content":{"title":"Message content.","type":"string"}},"required":["role"]}}}}},"required":["choices"]}]
    }
}
# Store the AI service in the repository
ai_service_metadata = {
    client.repository.AIServiceMetaNames.NAME: "rag service - inmemory",
    client.repository.AIServiceMetaNames.DESCRIPTION: "",
    client.repository.AIServiceMetaNames.SOFTWARE_SPEC_ID: software_spec_id,
    client.repository.AIServiceMetaNames.CUSTOM: {},
    client.repository.AIServiceMetaNames.REQUEST_DOCUMENTATION: request_schema,
    client.repository.AIServiceMetaNames.RESPONSE_DOCUMENTATION: response_schema,
    client.repository.AIServiceMetaNames.TAGS: ["wx-vector-index"]
}
ai_service_details = client.repository.store_ai_service(meta_props=ai_service_metadata, ai_service=gen_ai_service)
ai_service_id = client.repository.get_ai_service_id(ai_service_details)

# Deploy the stored AI Service
deployment_custom = {}
deployment_metadata = {
    client.deployments.ConfigurationMetaNames.NAME: "rag service - inmemory",
    client.deployments.ConfigurationMetaNames.ONLINE: {},
    client.deployments.ConfigurationMetaNames.CUSTOM: deployment_custom,
    client.deployments.ConfigurationMetaNames.DESCRIPTION: "",
    client.repository.AIServiceMetaNames.TAGS: ["wx-vector-index"]
}

function_deployment_details = client.deployments.create(ai_service_id, meta_props=deployment_metadata, space_id=space_id)

# Test AI Service
deployment_id = client.deployments.get_id(function_deployment_details)
print(deployment_id)
messages = []
remote_question = "Who won Best Actor?" # "Change this question to test your function"
messages.append({ "role" : "user", "content": remote_question })
payload = { "messages": messages }
result = client.deployments.run_ai_service(deployment_id, payload)
if "error" in result:
    print(result["error"])
else:
    print(result)