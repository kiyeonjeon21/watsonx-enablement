# Deploy AI Service: Chef Recipe Assistant
# This script creates and deploys a Watson ML AI service that recommends recipes

import os
import json
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.deployments import RuntimeContext

# ------------------ ENVIRONMENT SETUP ------------------

# Load environment variables
load_dotenv()

# Connect to Watson Machine Learning
credentials = Credentials(
    url=os.getenv("WATSONX_URL"),
    api_key=os.getenv("API_KEY")
)
client = APIClient(credentials)

# Set default space
space_id = os.getenv("SPACE_ID")
client.set.default_space(space_id)

# Project ID for asset promotion
source_project_id = os.getenv("PROJECT_ID")

# Configuration parameters
params = {
    "space_id": space_id,
    "watsonx_url": os.getenv("WATSONX_URL")
}

# ------------------ AI SERVICE DEFINITION ------------------

def gen_ai_service(context, params = params, **custom):
    # import dependencies
    import os
    from langchain_ibm import ChatWatsonx
    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai.foundation_models.utils import Tool, Toolkit
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent
    import json

    model = "meta-llama/llama-3-3-70b-instruct"
    
    service_url = params.get("watsonx_url")
    # Get credentials token
    credentials = {
        "url": service_url,
        "token": context.generate_token()
    }

    # Setup client
    client = APIClient(credentials)
    space_id = params.get("space_id")
    client.set.default_space(space_id)


    def create_chat_model(watsonx_client):
        parameters = {
            "frequency_penalty": 0,
            "max_tokens": 2000,
            "presence_penalty": 0,
            "temperature": 0,
            "top_p": 1
        }

        chat_model = ChatWatsonx(
            model_id=model,
            url=service_url,
            space_id=space_id,
            params=parameters,
            watsonx_client=watsonx_client,
        )
        return chat_model
    
    
    def create_custom_tool(tool_name, tool_description, tool_code, tool_schema):
        from langchain_core.tools import StructuredTool
        import ast
    
        def call_tool(**kwargs):
            tree = ast.parse(tool_code, mode="exec")
            custom_tool_functions = [ x for x in tree.body if isinstance(x, ast.FunctionDef) ]
            function_name = custom_tool_functions[0].name
            compiled_code = compile(tree, 'custom_tool', 'exec')
            namespace = {}
            exec(compiled_code, namespace)
            return namespace[function_name](**kwargs)
            
        tool = StructuredTool(
            name=tool_name,
            description = tool_description,
            func=call_tool,
            args_schema=tool_schema
        )
        return tool
    
    def create_custom_tools():
        custom_tools = []
    
    def create_tools(inner_client, context):
        tools = []
        top_k_results = 5
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=top_k_results))
        tools.append(wikipedia)
        max_results = 10
        search = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(max_results=max_results))
        tools.append(search)
        
        return tools
    
    def create_agent(model, tools, messages):
        memory = MemorySaver()
        instructions = """# Chef Recipe Assistant - Your Personal Sous Chef

## Your Role
You are an expert chef and culinary assistant specializing in recipe recommendations and cooking guidance. Your goal is to help users create delicious meals with the ingredients they have available, considering their preferences, dietary restrictions, and cooking skill level.

## Core Responsibilities
- Recommend recipes based on available ingredients
- Suggest ingredient substitutions when needed
- Provide cooking tips and techniques
- Consider dietary restrictions and preferences
- Offer variations for different skill levels
- Suggest complementary side dishes and beverages

## Guidelines
- Always ask about dietary restrictions or allergies if not mentioned
- Provide clear, step-by-step cooking instructions
- Include preparation and cooking times
- Suggest ingredient substitutions when possible
- Use markdown syntax for formatting recipes, ingredient lists, and cooking instructions
- When using search tools, focus on reputable cooking websites and culinary resources
- If you need to search for recipe information, try different search terms related to cooking and recipes
- Provide multiple recipe options when possible to give users variety

## Response Format
- Use markdown formatting for recipes and instructions
- Include ingredient lists in bullet points
- Number cooking steps clearly
- Add cooking tips in separate sections
- Include estimated serving sizes

When greeted, say "Hi! I'm your personal Sous Chef assistant. I'm here to help you create delicious meals with whatever ingredients you have on hand. What would you like to cook today?"

## Tools Usage Notes
- Use Wikipedia to find information about ingredients, cooking techniques, or regional cuisines
- Use web search to find current recipes, cooking tips, and ingredient information
- Always verify recipe information from multiple sources when possible
- Sometimes search results may not be perfect - try different cooking-related search terms"""
        for message in messages:
            if message["role"] == "system":
                instructions += message["content"]
        graph = create_react_agent(model, tools=tools, checkpointer=memory, state_modifier=instructions)
        return graph
    
    def convert_messages(messages):
        converted_messages = []
        for message in messages:
            if (message["role"] == "user"):
                converted_messages.append(HumanMessage(content=message["content"]))
            elif (message["role"] == "assistant"):
                converted_messages.append(AIMessage(content=message["content"]))
        return converted_messages

    def generate(context):
        payload = context.get_json()
        messages = payload.get("messages")
        inner_credentials = {
            "url": service_url,
            "token": context.get_token()
        }

        inner_client = APIClient(inner_credentials)
        model = create_chat_model(inner_client)
        tools = create_tools(inner_client, context)
        agent = create_agent(model, tools, messages)
        
        generated_response = agent.invoke(
            { "messages": convert_messages(messages) },
            { "configurable": { "thread_id": "42" } }
        )

        last_message = generated_response["messages"][-1]
        generated_response = last_message.content

        execute_response = {
            "headers": {
                "Content-Type": "application/json"
            },
            "body": {
                "choices": [{
                    "index": 0,
                    "message": {
                       "role": "assistant",
                       "content": generated_response
                    }
                }]
            }
        }

        return execute_response

    def generate_stream(context):
        print("Generate stream", flush=True)
        payload = context.get_json()
        headers = context.get_headers()
        is_assistant = headers.get("X-Ai-Interface") == "assistant"
        messages = payload.get("messages")
        inner_credentials = {
            "url": service_url,
            "token": context.get_token()
        }
        inner_client = APIClient(inner_credentials)
        model = create_chat_model(inner_client)
        tools = create_tools(inner_client, context)
        agent = create_agent(model, tools, messages)

        response_stream = agent.stream(
            { "messages": messages },
            { "configurable": { "thread_id": "42" } },
            stream_mode=["updates", "messages"]
        )

        for chunk in response_stream:
            chunk_type = chunk[0]
            finish_reason = ""
            usage = None
            if (chunk_type == "messages"):
                message_object = chunk[1][0]
                if (message_object.type == "AIMessageChunk" and message_object.content != ""):
                    message = {
                        "role": "assistant",
                        "content": message_object.content
                    }
                else:
                    continue
            elif (chunk_type == "updates"):
                update = chunk[1]
                if ("agent" in update):
                    agent = update["agent"]
                    agent_result = agent["messages"][0]
                    if (agent_result.additional_kwargs):
                        kwargs = agent["messages"][0].additional_kwargs
                        tool_call = kwargs["tool_calls"][0]
                        if (is_assistant):
                            message = {
                                "role": "assistant",
                                "step_details": {
                                    "type": "tool_calls",
                                    "tool_calls": [
                                        {
                                            "id": tool_call["id"],
                                            "name": tool_call["function"]["name"],
                                            "args": tool_call["function"]["arguments"]
                                        }
                                    ] 
                                }
                            }
                        else:
                            message = {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": tool_call["id"],
                                        "type": "function",
                                        "function": {
                                            "name": tool_call["function"]["name"],
                                            "arguments": tool_call["function"]["arguments"]
                                        }
                                    }
                                ]
                            }
                    elif (agent_result.response_metadata):
                        # Final update
                        message = {
                            "role": "assistant",
                            "content": agent_result.content
                        }
                        finish_reason = agent_result.response_metadata["finish_reason"]
                        if (finish_reason): 
                            message["content"] = ""

                        usage = {
                            "completion_tokens": agent_result.usage_metadata["output_tokens"],
                            "prompt_tokens": agent_result.usage_metadata["input_tokens"],
                            "total_tokens": agent_result.usage_metadata["total_tokens"]
                        }
                elif ("tools" in update):
                    tools = update["tools"]
                    tool_result = tools["messages"][0]
                    if (is_assistant):
                        message = {
                            "role": "assistant",
                            "step_details": {
                                "type": "tool_response",
                                "id": tool_result.id,
                                "tool_call_id": tool_result.tool_call_id,
                                "name": tool_result.name,
                                "content": tool_result.content
                            }
                        }
                    else:
                        message = {
                            "role": "tool",
                            "id": tool_result.id,
                            "tool_call_id": tool_result.tool_call_id,
                            "name": tool_result.name,
                            "content": tool_result.content
                        }
                else:
                    continue

            chunk_response = {
                "choices": [{
                    "index": 0,
                    "delta": message
                }]
            }
            if (finish_reason):
                chunk_response["choices"][0]["finish_reason"] = finish_reason
            if (usage):
                chunk_response["usage"] = usage
            yield chunk_response

    return generate, generate_stream

# ------------------ LOCAL TESTING ------------------

# Initialize AI Service function locally
context = RuntimeContext(api_client=client)

# Choose between streaming and non-streaming mode
streaming = False
findex = 1 if streaming else 0
local_function = gen_ai_service(context, space_id=space_id)[findex]
messages = []

# Test question
local_question = "I'm looking for a quick dinner idea using ground beef, tomatoes, and pasta. Can you suggest a few options?"
messages.append({"role": "user", "content": local_question})

context = RuntimeContext(api_client=client, request_payload_json={"messages": messages})
response = local_function(context)

if streaming:
    for chunk in response:
        print(chunk, end="\n\n", flush=True)
else:
    print(json.dumps(response, indent=2))

# ------------------ DEPLOY AI SERVICE ------------------

# Look up software specification for the AI service
# software_spec_id_in_project = "c31aedfd-d5dc-400f-ba3a-0390d67d7e04"

# try:
#     software_spec_id = client.software_specifications.get_id_by_name("ai-service-v6-h-software-specification")
# except Exception as e:
#     software_spec_id = client.spaces.promote(software_spec_id_in_project, source_project_id, space_id)

# Create conda environment configuration
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
    client.package_extensions.ConfigurationMetaNames.NAME: "chef-assistant-env",
    client.package_extensions.ConfigurationMetaNames.DESCRIPTION: "Environment for Chef Recipe Assistant",
    client.package_extensions.ConfigurationMetaNames.TYPE: "conda_yml"
}

pkg_extn_details = client.package_extensions.store(meta_props=meta_prop_pkg_extn, file_path="config.yaml")
pkg_extn_id = client.package_extensions.get_id(pkg_extn_details)

# Create software specification
custom_sw_spec_name = "chef-assistant-software-spec"
try:
    # 이름으로 ID 조회 시도
    software_spec_id = client.software_specifications.get_id_by_name(custom_sw_spec_name)
except:
    # 없는 경우에만 새로 생성
    meta_prop_sw_spec = {
        client.software_specifications.ConfigurationMetaNames.NAME: custom_sw_spec_name,
        client.software_specifications.ConfigurationMetaNames.DESCRIPTION: "Software specification for Chef Recipe Assistant",
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
    client.repository.AIServiceMetaNames.NAME: "Sous Chef Assistant",
    client.repository.AIServiceMetaNames.DESCRIPTION: "A Sous Chef assistant that provides personalized recipe recommendations based on available ingredients and dietary preferences",
    client.repository.AIServiceMetaNames.SOFTWARE_SPEC_ID: software_spec_id,
    client.repository.AIServiceMetaNames.CUSTOM: {},
    client.repository.AIServiceMetaNames.REQUEST_DOCUMENTATION: request_schema,
    client.repository.AIServiceMetaNames.RESPONSE_DOCUMENTATION: response_schema,
    client.repository.AIServiceMetaNames.TAGS: ["wx-agent"]
}

ai_service_details = client.repository.store_ai_service(
    meta_props=ai_service_metadata, 
    ai_service=gen_ai_service
)

# Get the AI Service ID
ai_service_id = client.repository.get_ai_service_id(ai_service_details)

# Deploy the stored AI Service
deployment_custom = {
    "avatar_icon": "Chemistry",
    "avatar_color": "supportCautionMajor",
    "placeholder_image": "placeholder2.png",
    "sample_questions": [
        "What can I make with chicken breast, vegetables, and rice?",
        "I need a quick pasta recipe for dinner, what do you suggest?",
        "Can you recommend a healthy vegetarian meal?",
        "What's a good recipe for beginners using basic pantry ingredients?"
    ]
}

deployment_metadata = {
    client.deployments.ConfigurationMetaNames.NAME: "Sous Chef Assistant",
    client.deployments.ConfigurationMetaNames.ONLINE: {},
    client.deployments.ConfigurationMetaNames.CUSTOM: deployment_custom,
    client.deployments.ConfigurationMetaNames.DESCRIPTION: "Your personal sous chef for recipe recommendations and cooking guidance.",
    client.repository.AIServiceMetaNames.TAGS: ["wx-agent"]
}

function_deployment_details = client.deployments.create(
    ai_service_id, 
    meta_props=deployment_metadata, 
    space_id=space_id
)

# Get the deployment ID
deployment_id = client.deployments.get_id(function_deployment_details)
print(f"Deployment ID: {deployment_id}")

# ------------------ TEST DEPLOYED SERVICE ------------------

# Test the deployed service
messages = []
remote_question = "I'm in Boston, MA. I have chicken breast, bell peppers, onions, and rice in my fridge. What are some recipe ideas I can make with these ingredients?"
messages.append({"role": "user", "content": remote_question})
payload = {"messages": messages}

result = client.deployments.run_ai_service(deployment_id, payload)
if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(json.dumps(result, indent=2))