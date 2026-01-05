from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv


load_dotenv()


model_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=model_id,
    temperature=0.7
)
llm_agent = ChatHuggingFace(llm=llm)


class AgentSate(TypedDict):
    message: List[Union[HumanMessage, AIMessage]]


def process(state: AgentSate) -> AgentSate:
    """definition of process node in the graph that invoke the llm agent."""
    response = llm_agent.invoke(state["message"])
    print("AI Response :", response.content)
    state["message"].append(AIMessage(content=response.content))
    return state


graph = StateGraph(AgentSate)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history = []
user_input = input("Your query: ")
while (user_input != "exit"):
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({
        "message": conversation_history
    })
    # print("Full Conversation History:", result["message"])
    conversation_history = result["message"]
    user_input = input("Your query: ")
