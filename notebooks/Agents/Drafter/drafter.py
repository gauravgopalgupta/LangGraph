from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv


load_dotenv()


# This is the global variable to sotre the document content.
document_content = ""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """The tool function to update the document content."""
    global document_content
    document_content = content + "\n"
    return ("The document has been updated successfully! The current "
            f"content is:\n{document_content}")


@tool
def save(filename: str) -> str:
    """The tool function to save the current document content to a text file
      and finish the process.

    Args:
        filename (str): The name of the file where the document content will
          be saved.
    """

    global document_content

    if not filename.endswith(".txt"):
        filename += ".txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
            return ("The document has been successfully saved"
                    f" to the file: {filename}")

    except Exception as e:
        return f"An error occurred while saving the document: {str(e)}"


tools = [update, save]

model_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=model_id,
    temperature=0.7
)
llm_model = ChatHuggingFace(llm=llm).bind_tools(tools)


def model_call_agent(state: AgentState) -> AgentState:
    global document_content
    """This function is used to define the agent node task."""
    system_prompt = SystemMessage(
        content=(f"""
            You are a Drafter, a helpful writing assistant. You are going to
                help the user update and modifiy documents.

            - If the user wants to update or modify content, use the 'update'
                tool with the complete updated content.
            - If the user wants to save and finish, you need to use the 'save'
                tool.
            - Make sure to always show the current document state after
                modifications.

            The current document content is: {document_content}
            """)
    )

    if not state["messages"]:
        user_input = (
            "I'm ready to help you update a document. "
            "What would you like to create or modify?"
        )
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    complete_message = (
        [system_prompt]
        + list(state["messages"])
        + [user_message]
    )
    response = llm_model.invoke(complete_message)

    print("AI Response :", response.content)
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> str:
    """This function is used to define the decision edge that decides whether
      to continue the loop between agent node and tools or exit."""

    messages = state["messages"]
    if not messages:
        return "continue"

    # This looks for the most recent tool message.....
    for message in reversed(messages):
        # ... and check if this is a ToolMessage resulting from save tool call
        if (isinstance(message, ToolMessage) and
                "saved" in message.content.lower() and
                "document" in message.content.lower()):
            return "end"  # goes to the end edge which leads to the endpoint
        # you can use  message.tool_calls == "save" if needed.

    return "continue"  # goes back to the agent node default edge.


def print_messages(messages):
    """Function I made to print the message in a more readable format."""

    if not messages:
        print("No messages to display.")
        return

    for message in messages[-3:]:   # print only the last 3 messages.
        if isinstance(message, ToolMessage):
            print(f"\n TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)
graph.add_node("agent", model_call_agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",
        "end": END,
    },
)

app = graph.compile()


def run_drafter_agent():
    """Function to run the drafter agent interactively."""
    print("\n ===== DRAFTER SESSION STARTED ===== \n")

    state = {"messages": []}

    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])

    print("\n ===== DRAFTER SESSION ENDED ===== \n")


if __name__ == "__main__":
    run_drafter_agent()
