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

    sys_msg = SystemMessage(content=f"""You are an autonomous file organization agent.

{bucket_info}

INSTRUCTIONS:
1. Call `get_objects` to see what is in the bucket.
2. Based on the file list, call `perform_action` repeatedly to move files to organized folders (e.g., /images/, /docs/).
3. **ACTUALLY CALL THE TOOLS.** Do not write a plan. Do not output raw JSON in the text.
>>>>>>> Stashed changes
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