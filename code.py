from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv

import os
import requests


 

load_dotenv()


 
search_tool = TavilySearch(
    max_results=2,
    tavily_api_key=os.environ.get("TAVILY_API_KEY")
)


 
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email using Brevo.

    Args:
        to: Email address of the recipient.
        subject: Subject of the email.
        body: Content of the email.
    """

    url = "https://api.brevo.com/v3/smtp/email"

    headers = {
        "accept": "application/json",
        "api-key": os.environ.get("BREVO_API_KEY"),
        "content-type": "application/json"
    }

    data = {
        "sender": {
            "name": "My AI Agent",
            "email": "vijaysuryahsn@gmail.com"
        },
        "to": [
            {
                "email": to
            }
        ],
        "subject": subject,
        "textContent": body
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=data
        )

        if response.status_code == 201:
            return f"Email successfully sent to {to}"

        return f"Failed to send email: {response.text}"

    except Exception as e:
        return f"Error while sending email: {str(e)}"



class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


 
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY")
)



tools = [
    search_tool,
    send_email
]

llm_with_tools = llm.bind_tools(tools)



def call_llm(state: GraphState) -> GraphState:

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }



tool_node = ToolNode(tools)



def route(state: GraphState) -> str:

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tool"

    return "end"



builder = StateGraph(GraphState)

# Add nodes
builder.add_node("llm", call_llm)
builder.add_node("tool", tool_node)

# Starting point
builder.set_entry_point("llm")

# Conditional routing
builder.add_conditional_edges(
    "llm",
    route,
    {
        "tool": "tool",
        "end": END
    }
)

# After tool execution → go back to LLM
builder.add_edge("tool", "llm")



graph = builder.compile()


result = graph.invoke(
    {
        "messages": [
            (
                "user",
                "Send an email to example@gmail.com "
                "with the subject 'Test Email' "
                "and say 'Hello, this is a test email from my LangGraph agent.'"
            )
        ]
    }
)

 

for m in result["messages"]:
    m.pretty_print()
 
