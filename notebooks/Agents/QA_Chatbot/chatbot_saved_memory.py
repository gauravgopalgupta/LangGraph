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


class AgentState(TypedDict):
    message: List[Union[HumanMessage, AIMessage]]


def process(state: AgentState) -> AgentState:
    """definition of process node in the graph that invoke the llm agent."""
    response = llm_agent.invoke(state["message"])
    print("AI Response :", response.content)
    state["message"].append(AIMessage(content=response.content))
    return state


graph = StateGraph(AgentState)
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
    print("Full Conversation History:", result["message"])
    conversation_history = result["message"]
    user_input = input("Your query: ")

# Saving your chat for future use in the chatbot with saved memory
with open("chat_logging_history.txt", "w") as file:
    file.write("Chat Conversation History:\n")

    for msg in conversation_history:
        if isinstance(msg, HumanMessage):
            file.write(f"you: {msg.content}\n")
        elif isinstance(msg, AIMessage):
            file.write(f"AI: {msg.content}\n")
    file.write("\nEnd of Conversation\n")
