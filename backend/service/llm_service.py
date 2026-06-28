import asyncio
import json
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from typing import Any, List, TypedDict
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

BASE_DIR = Path(__file__).resolve().parent
CONVERSATION_FILE = BASE_DIR / "conversations.json"
SYSTEM_MESSAGE = SystemMessage(content="You are a helpful programming assistant.")

# load the api key from the .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set")

# initialize the model with the api key and the model name
model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.0,
    max_retries=2,
    groq_api_key=groq_api_key
)

class GraphState(TypedDict):
    messages: List[BaseMessage]
    thread_id: str
    conversation_title: str


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def _message_to_record(message: BaseMessage) -> dict[str, str]:
    return {
        "role": message.type,
        "content": message.content,
    }


def _record_to_message(record: dict[str, Any]) -> BaseMessage:
    role = record.get("role", "human")
    content = record.get("content", "")

    if role == "system":
        return SystemMessage(content=content)
    if role == "ai":
        return AIMessage(content=content)
    return HumanMessage(content=content)


def _load_conversations() -> list[dict[str, Any]]:
    if not CONVERSATION_FILE.exists():
        return []

    with CONVERSATION_FILE.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    if isinstance(data, list):
        return data

    return []


def _save_conversations(conversations: list[dict[str, Any]]) -> None:
    with CONVERSATION_FILE.open("w", encoding="utf-8") as file_handle:
        json.dump(conversations, file_handle, indent=2, ensure_ascii=False)


def _upsert_conversation(conversation: dict[str, Any]) -> None:
    conversations = _load_conversations()

    for index, existing in enumerate(conversations):
        if existing.get("thread_id") == conversation.get("thread_id"):
            conversations[index] = conversation
            break
    else:
        conversations.append(conversation)

    _save_conversations(conversations)


def _format_conversation_label(conversation: dict[str, Any]) -> str:
    title = conversation.get("title", "Untitled conversation")
    return title.strip() or "Untitled conversation"


def _build_state_from_conversation(conversation: dict[str, Any]) -> GraphState:
    messages = [
        _record_to_message(message_record)
        for message_record in conversation.get("messages", [])
    ]

    if not messages or messages[0].type != "system":
        messages.insert(0, SYSTEM_MESSAGE)

    return {
        "messages": messages,
        "thread_id": conversation["thread_id"],
        "conversation_title": conversation["title"],
    }


async def generate_conversation_title(first_user_input: str, thread_id: str) -> str:
    title_prompt = [
        SystemMessage(
            content=(
                "Generate a short, descriptive conversation title. "
                "Return only the title text, with no quotes, bullets, or extra explanation."
            )
        ),
        HumanMessage(
            content=(
                f"Thread ID: {thread_id}\n"
                f"User input: {first_user_input}\n"
                "Make the title concise and specific."
            )
        ),
    ]

    response = await model.ainvoke(title_prompt)
    title = str(response.content).strip()

    if not title:
        title = "Untitled conversation"

    title = title.splitlines()[0].strip().strip('"').strip("'")
    return title[:80] if len(title) > 80 else title


def _print_conversation_menu(conversations: list[dict[str, Any]]) -> None:
    print("\nPrevious conversations:")
    for index, conversation in enumerate(conversations, start=1):
        print(f"{index}. {_format_conversation_label(conversation)}")


async def select_conversation() -> GraphState:
    conversations = _load_conversations()

    if conversations:
        while True:
            print("\n1. Continue previous conversation")
            print("2. Start new conversation")
            choice = input("Select an option: ").strip().lower()

            if choice in {"1", "continue", "c"}:
                _print_conversation_menu(conversations)
                while True:
                    selected = input("Select a conversation by number: ").strip()
                    if selected.isdigit() and 1 <= int(selected) <= len(conversations):
                        return _build_state_from_conversation(conversations[int(selected) - 1])
                    print("Invalid selection. Try again.")

            if choice in {"2", "new", "n"}:
                break

            print("Invalid selection. Enter 1 or 2.")

    while True:
        first_user_input = input("Start your new conversation: ").strip()
        if first_user_input:
            break
        print("Conversation prompt cannot be empty.")

    thread_id = generate_thread_id()
    title = await generate_conversation_title(first_user_input, thread_id)
    messages = [
        SYSTEM_MESSAGE,
        HumanMessage(content=first_user_input),
    ]

    conversation = {
        "thread_id": thread_id,
        "title": title,
        "messages": [_message_to_record(message) for message in messages],
    }
    _upsert_conversation(conversation)

    print(f"\nCreated conversation: {title}")

    return {
        "messages": messages,
        "thread_id": thread_id,
        "conversation_title": title,
    }


async def run_chat(state: GraphState):
    while True:
        last_message = state["messages"][-1] if state["messages"] else None

        if not isinstance(last_message, HumanMessage):
            user_input = input("\nYou: ")

            if user_input.lower() in ["exit", "quit"]:
                break

            state["messages"].append(HumanMessage(content=user_input))
            _upsert_conversation(
                {
                    "thread_id": state["thread_id"],
                    "title": state["conversation_title"],
                    "messages": [_message_to_record(message) for message in state["messages"]],
                }
            )

        print("AI: ", end="", flush=True)

        full_response = ""

        async for chunk in model.astream(state["messages"]):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
        
        state["messages"].append(AIMessage(content=full_response))
        _upsert_conversation(
            {
                "thread_id": state["thread_id"],
                "title": state["conversation_title"],
                "messages": [_message_to_record(message) for message in state["messages"]],
            }
        )

    return state

workflow = StateGraph(GraphState)
workflow.add_node("run_chat", run_chat)
workflow.add_edge(START, "run_chat")
workflow.add_edge("run_chat", END)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


async def main():
    state = await select_conversation()
    result = await app.ainvoke(
        state,
        config={"configurable": {"thread_id": state["thread_id"]}},
    )
    print("Workflow result:", result)


if __name__ == "__main__":
    asyncio.run(main())


