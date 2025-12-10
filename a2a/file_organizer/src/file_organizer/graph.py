import os
from langgraph.graph import StateGraph, MessagesState, START
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import SystemMessage,  AIMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import ChatOpenAI

from file_organizer.configuration import Configuration

config = Configuration()

# Extend MessagesState to include a final answer
class ExtendedMessagesState(MessagesState):
     final_answer: str = ""

def get_mcpclient():
    
    return MultiServerMCPClient({
        "cloud_storage": {
            "url": os.getenv("MCP_URL", "http://cloud-storage-tool:8000/mcp"),
            "transport": os.getenv("MCP_TRANSPORT", "streamable_http"),
        }
    })

async def get_graph(client) -> StateGraph:
    llm = ChatOpenAI(
        model=config.llm_model,
        openai_api_key=config.llm_api_key,
        openai_api_base=config.llm_api_base,
        temperature=0,
    )

    # Get tools asynchronously
    tools = await client.get_tools()
    llm_with_tools = llm.bind_tools(tools)
    bucket_uri = os.getenv("BUCKET_URI")

    bucket_info = f"Target bucket: {bucket_uri}" if bucket_uri else "No bucket URI configured. Ask the user to specify which bucket to organize."

    sys_msg = SystemMessage(content=f"""You are a file organization assistant for cloud storage buckets.

{bucket_info}

CRITICAL INSTRUCTIONS:

1. Discovery Phase:
   - Call `get_objects` to list files.
   - **PAY ATTENTION:** The output of `get_objects` contains a field called `file_uri` for every file (e.g., "s3://my-bucket/folder/image.png").

2. Action Phase:
   - To move a file, call `perform_action`.
   - **MANDATORY:** For the `file_uri` argument, you must copy the **exact string** provided in the `file_uri` field from the `get_objects` output. Do not guess the path. Do not construct the URI yourself.
   - For the `target_uri` argument, ensure it is a valid folder path starting with the proper protocol (e.g. "s3://...") and **ending with a trailing slash '/'**.

3. Execution:
   - Call the tools directly. Do not output text descriptions or JSON strings.
""")

    # Node
    def assistant(state: ExtendedMessagesState) -> ExtendedMessagesState:
        messages = [sys_msg] + state["messages"]
    
        # Invoke the LLM
        result = llm_with_tools.invoke(messages)
        
        # Return the result to append to history
        state["messages"].append(result)
        return state

    # Build graph
    builder = StateGraph(ExtendedMessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    graph = builder.compile()
    return graph