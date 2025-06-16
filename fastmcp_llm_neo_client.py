# client.py
import asyncio
from fastmcp import client
from fastmcp.client.transports import StreamableHttpTransport
import httpx
import pprint
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def initialize_llm_model():
    print("Initializing LLM model...")
    """Initialize the LLM model and tokenizer."""
    # Ensure the model is downloaded and loaded correctly
    # This function will load the model and tokenizer, and set the device to GPU if available.
    # If the model is not downloaded, it will be downloaded automatically by the transformers library.
    print("Loading model and tokenizer...")
    # -----------------------------
    # 1. Load Phi-2 Mini LLM
    # -----------------------------
    # model_id = "microsoft/phi-3-mini-4k-instruct"
    # model_id = "microsoft/phi-2"
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = initialize_llm_model()

def get_tool_call(natural_input):
    system_prompt = """You are a helpful assistant that converts user questions into tool calls. yur job to is call the right tools based on the user input
    in natural language. The tools are provided by a knowledge graph (KG) client that can query a Neo4j database.
    The KG client has the following tools available:
            - get_all_datasets_and_files: Returns all datasets and files in the Neo4j database.
            - get_features_for_file: Returns features for a specific file in the Neo4j database.
            - get_files_by_type: Returns files of a specific type (e.g., train, test) in the Neo4j database.
            - get_all_units: Returns all units in the Neo4j database.
    You will receive a natural language question from the user, and you need to convert it into a tool call in JSON format.
    The JSON format should look like this:
            {"method": "tool_name", "arguments": {"arg1": "value1", "arg2": "value2"}}
    where "tool_name" is the name of the tool to call, and "arguments" is a dictionary of arguments to pass to the tool.
    If the user question does not match any of the available tools, return None.
    If the user question is not clear or does not provide enough information to call a tool, return None.
    If the user question is about the Neo4j database for RUL prediction use case, you can use the tools listed above. An example
        Available tool: 
            - query(query: str)
            Return ONLY a JSON object like this:
            {"method": "query", "arguments": {"query": "MATCH ..."}}
            """
    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{natural_input}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        json_text = decoded.split("{", 1)[1].rsplit("}", 1)[0]
        return json.loads("{" + json_text + "}")
    except Exception as e:
        print("Failed to parse JSON from LLM:", e)
        print("Raw LLM output:", decoded)
        return None


async def getdatafrom_kg():
    '''Function to get data from the knowledge graph using the MCP client.
    This function initializes the MCP client, connects to the server, and retrieves data from the Neo4j database.'''
    # Example usage of the MCP client
    transport = StreamableHttpTransport("http://localhost:8000/mcp")

    async with client.Client(transport) as mcp_client:
        await mcp_client.ping()
        print("Ping successful")

        tools = await mcp_client.list_tools()
        print("Available tools:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")

        data_file_results = await mcp_client.call_tool("get_all_datasets_and_files", {})
        
        print("Neo4j results:")
        print(30*"-")
        print(pprint.pformat(data_file_results))
       
        feature_results = await mcp_client.call_tool("get_features_for_file", {"file_name": "train_FD001"})
        
        print("feature_results:")
        print(30*"-")
        print(pprint.pformat(feature_results))

        files_by_type_results = await mcp_client.call_tool("get_files_by_type", {"file_type": "test"})
        
        print("Files by type results:")
        print(30*"-")
        print(pprint.pformat(files_by_type_results))

        units_results = await mcp_client.call_tool("get_all_units", {})
        
        print("Units results:")
        print(30*"-")
        print(pprint.pformat(units_results))

async def main():
    # mcp = MCPClient("http://localhost:8000/mcp")

    # What are the databases and file nodes available in Neo4j  ? 
    user_input = input("Ask a question in English for the NCMAPS data on the Neo4j database ? ")


    transport = StreamableHttpTransport("http://localhost:8000/mcp")
    async with client.Client(transport) as mcp_client:
    
        tool_call = get_tool_call(user_input)
        if tool_call:
            print("Tool call generated:", tool_call)
            response = await mcp_client.call_tool(tool_call["method"], tool_call["arguments"])
            print("Response from tool call:", response)
        else:
            print("Failed to generate tool call from user input.")

if __name__ == "__main__":    
    asyncio.run(main())
    # Run the example
    # asyncio.run(getdatafrom_kg())

    
