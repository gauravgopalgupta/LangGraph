from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv


load_dotenv()


class AgentState(TypedDict):
    message: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: float, b: float):
    """This is an addition function that adds two numbers together.
    Acting as a addition tool."""

    return a + b


def substract(a: float, b: float):
    """This is a subtraction function that subtracts two numbers.
    Acting as a subtraction tool."""

    return a - b


def multiply(a: float, b: float):
    """This is a multiplication function that multiplies two numbers.
    Acting as a multiplication tool."""

    return a * b


def divide(a: float, b: float):
    """This is a division function that divides two numbers.
    Acting as a division tool."""

    return a / b


tools = [add, substract, multiply, divide]

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=model_id,
    temperature=0.7
)
llm_model = ChatHuggingFace(llm=llm).bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    """definition of process node in the graph that invoke the llm agent."""
    system_prompt = SystemMessage(
        content=(
            "You are my AI assistant, please answer my query to the best of "
            "your ability.")
    )
    response = llm_model.invoke([system_prompt] + state["message"])
    print("AI Response :", response.content)
    return {"message": [response]}


def should_continue(state: AgentState):
    """definition of the decision edge in the graph that decides whether to
       continue the loop between agent node and tools or exit."""
    messages = state["message"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
graph.add_edge("tools", "agent")
app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["message"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


input = {
    "message":
    [(
        "user",
        "Can you please add 16 and 12 followed by multiplication by 7 followed"
        " by substraction by 6?"
    )]
}

print_stream(app.stream(input, stream_mode="values"))

input = {
    "message":
    [(
        "user",
        "Can you please add 16 and 12, followed by multiplication by 7, "
        "followed by substraction by 6, then finally divide by 7?, Also "
        "can you please tell me a joke?"
    )]
}

print_stream(app.stream(input, stream_mode="values"))
