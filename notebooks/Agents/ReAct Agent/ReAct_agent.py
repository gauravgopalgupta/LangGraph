from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
# from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

"""
Import Definitions:
- Annotated: Annotated: provides additional context (metadata) about the type
 it wraps without affecting the type itself. for example:
    email=Annotated[str,"email should be a valid email like abc@gmail.com"]
- Sequence: sequence: represents an ordered collection of items, similar to a
 list or tuple. It is automatically handle the state updates for sequences
 such as by adding new messages to a chat history.
- BaseMessage: Base class (The foundational class) for all message types in
  LangGraph.
- ToolMessage: Passes data back to LLM from tools after it has executed a tool
 such as the content and the tool_call_id.
- SystemMessage: Message for providing instructions to the LLM, used to set
 the behavior of the AI assistant.
- Tool: Represents a tool that can be used by the LLM to perform specific
  tasks.
- ChatHuggingFace: A chat-based LLM interface for HuggingFace models.

"""
load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int):
    """This is an addition function that adds two numbers together."""
    return a + b


tools = [add]

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=model_id,
    temperature=0.7
)
llm_agent = ChatHuggingFace(llm=llm).bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    """definition of process node in the graph that invoke the llm agent."""
    system_prompt = SystemMessage(
        content=(
            "You are my AI assistant, please answer my query to the best "
            "of your ability."
        )
    )
    response = llm_agent.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    """definition of the decision edge in the graph that decides whether to
       continue the loop between agent node and tools or exit."""
    messages = state["messages"]
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
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "Can you please add 12 and 16 for me?")]}
print_stream(app.stream(inputs, stream_mode="values"))


inputs = {
    "messages": 
    [(
        "user",
        "Can you please add 12 and 16 for me? Also, Please add 7 and 6 for me."
    )]
}
print_stream(app.stream(inputs, stream_mode="values"))
