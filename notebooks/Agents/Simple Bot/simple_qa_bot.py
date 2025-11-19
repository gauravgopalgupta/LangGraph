from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=model_id,
    temperature=0.7
)
llm_agent = ChatHuggingFace(llm=llm)


class AgentState(TypedDict):
    messages: List[HumanMessage]


def process(state: AgentState) -> AgentState:
    response = llm_agent.invoke(state["messages"])
    print("AI Response :", response.content)
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter your message: ")
agent.invoke({
    "messages": [HumanMessage(content=user_input)]
})
