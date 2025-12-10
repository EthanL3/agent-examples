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

    sys_msg = SystemMessage(content=f"""You are a tireless file organization engine for cloud storage.

{bucket_info}

**YOUR OPERATING MANUAL:**

1. **PHASE 1: DISCOVERY**
   - Call `get_objects` to retrieve the full list of files.

2. **PHASE 2: EXHAUSTIVE EXECUTION**
   - You must review **EVERY SINGLE FILE** in the list.
   - For **EACH** file that is not in the correct folder, generate a `perform_action` tool call.
   - **DO NOT STOP** after moving 1 or 2 files. If there are 20 unorganized files, you must generate 20 tool calls.
   - **MANDATORY:** Use the `file_uri` field from the discovery phase exactly as provided. Do not alter it.

3. **PHASE 3: REPORTING**
   - **ONLY** after you have generated all necessary tool calls and received their success results, output a final response.
   - The response should be a **concise** summary (e.g., "Moved 15 files: 10 images to /images and 5 PDFs to /docs.").
   - Do not list every file in the summary. Keep it high-level.
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