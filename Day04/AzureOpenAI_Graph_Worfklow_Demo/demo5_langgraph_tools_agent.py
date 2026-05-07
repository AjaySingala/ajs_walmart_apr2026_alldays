# Demo 5: LangGraph with Tools (@tool) + Agent Decision.
# Human-in-the-loop.

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path).
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
import os
from dotenv import load_dotenv

# TODO: Replace AzureChatOpenAI with ChatOpenAI.
from langchain_openai import AzureChatOpenAI

from langchain_core.tools import tool
from langgraph.graph import StateGraph, END

load_dotenv()

# ===== LLM =====
# TODO: Replace the following with connection to ChatOpenAI.
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version="2024-02-15-preview",
)

# ===== TOOLS =====

# @tool
# def summarize_tool(text: str) -> str:
#     """Summarizes business input"""
#     print(f"\n summarize_tool()...")
#     return llm.invoke(f"Summarize:\n{text}").content


# @tool
# def insights_tool(text: str) -> str:
#     """Extracts insights"""
#     print(f"\n insights_tool()...")
#     return llm.invoke(f"Extract insights:\n{text}").content


# @tool
# def recommend_tool(text: str) -> str:
#     """Recommends actions"""
#     print(f"\n recommend_tool()...")
#     return llm.invoke(f"Recommend actions:\n{text}").content

@tool
def summarize_tool(text: str) -> str:
    """Summarizes business input"""
    print(f"\n summarize_tool()...")

    prompt = f"""
    You are a senior business analyst.

    Context:
    The following is operational/customer feedback data.

    Task:
    - Summarize the key issues clearly
    - Highlight major trends or risks

    Output format:
    - 3-5 bullet points
    - Concise and executive-friendly
    - No fluff, no repetition

    Input:
    {text}
    """

    return llm.invoke(prompt).content

@tool
def insights_tool(text: str) -> str:
    """Extracts insights"""
    print(f"\n insights_tool()...")

    prompt = f"""
    You are a strategy consultant.

    Context:
    You are analyzing a business summary.

    Task:
    - Identify root causes
    - Highlight business impact
    - Call out risks and patterns

    Output format:
    - Key Insight 1: ...
    - Key Insight 2: ...
    - Key Insight 3: ...
    - Include impact (e.g., revenue, churn, CX)

    Input:
    {text}
    """

    return llm.invoke(prompt).content

@tool
def recommend_tool(text: str) -> str:
    """Recommends actions"""
    print(f"\n recommend_tool()...")

    prompt = f"""
    You are a senior strategy advisor.

    Context:
    Based on the insights provided, recommend business actions.

    Task:
    - Provide actionable recommendations
    - Prioritize high-impact actions
    - Keep it practical and implementable

    Output format:
    - Action 1: [What] | [Why] | [Expected Impact]
    - Action 2: ...
    - Action 3: ...

    Keep recommendations realistic for execution.

    Input:
    {text}
    """

    return llm.invoke(prompt).content


# ===== STATE =====
from typing import TypedDict

class AgentState(TypedDict):
    input: str
    summary: str
    insights: str
    actions: str
    approved: str 

# ===== NODES =====

def summarize_node(state: AgentState):
    print(f"\n summarize_node()...")
    # Step 1: summarize input
    # Logic
    result = summarize_tool.invoke(state["input"])
    # Logic
    print("\n🔹 Summary:\n", result)
    return {"summary": result}


def insights_node(state: AgentState):
    print(f"\n insights_node()...")
    # Step 2: extract insights
    result = insights_tool.invoke(state["summary"])
    print("\n🔹 Insights:\n", result)
    return {"insights": result}


def recommend_node(state: AgentState):
    print(f"\n recommend_node()...")
    # Step 3: generate recommendations
    result = recommend_tool.invoke(state["insights"])
    print("\n🔹 Recommendations:\n", result)
    return {"actions": result}


def human_approval_node(state: AgentState):
    print(f"\n human_approval_node()...")
    # Human-in-the-loop decision point
    print("\n🤖 Proposed Actions:\n", state["actions"])
    foo()
    decision = input("\n👉 Approve? (yes/no): ").strip().lower()

    return {"approved": decision}


# ===== ROUTER =====

def approval_router(state: AgentState):
    print(f"\n approval_router()...")
    # If approved → END
    if state["approved"] == "yes":
        print(f"\n THE END!...")
        return "end"
    # If not approved → restart from summarize
    return "retry"

# foo() is not a node because it is not identified/defined in the graph.
def foo():
    # does something
    pass

# ===== BUILD GRAPH =====
# Orchestrator.
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("summarize", summarize_node)
graph.add_node("insights", insights_node)
graph.add_node("recommend", recommend_node)
graph.add_node("human", human_approval_node)

# Define linear flow
graph.set_entry_point("summarize")
graph.add_edge("summarize", "insights")
graph.add_edge("insights", "recommend")
graph.add_edge("recommend", "human")

# Add human decision loop
graph.add_conditional_edges(
    "human",
    approval_router,
    {
        "retry": "summarize",  # loop again
        "end": END             # exit
    }
)

# Compile graph
app = graph.compile()


# ===== RUN =====

input_data = {
    "input": """
    Customer complaints increased by 25%.
    Delivery delays and poor support are major issues.
    Customer churn risk is rising significantly.
    """
}

app.invoke(input_data)
