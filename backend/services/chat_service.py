from __future__ import annotations

import asyncio
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from memory.conversation_store import (
    SYSTEM_MESSAGE,
    build_conversation_record,
    build_state_from_conversation,
    format_conversation_label,
    generate_thread_id,
    load_conversations,
    upsert_conversation,
)
from model.llm import get_chat_model


class GraphState(TypedDict):
    messages: list[BaseMessage]
    thread_id: str
    conversation_title: str


async def generate_conversation_title(first_user_input: str, thread_id: str) -> str:
    model = get_chat_model()
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
        print(f"{index}. {format_conversation_label(conversation)}")


async def select_conversation() -> GraphState:
    conversations = load_conversations()

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
                        return build_state_from_conversation(conversations[int(selected) - 1])
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

    conversation = build_conversation_record(
        thread_id=thread_id,
        title=title,
        messages=messages,
    )
    upsert_conversation(conversation)

    print(f"\nCreated conversation: {title}")

    return {
        "messages": messages,
        "thread_id": thread_id,
        "conversation_title": title,
    }


def _persist_state(state: GraphState) -> None:
    upsert_conversation(
        build_conversation_record(
            thread_id=state["thread_id"],
            title=state["conversation_title"],
            messages=state["messages"],
        )
    )


async def _stream_assistant_response(messages: list[BaseMessage]) -> str:
    model = get_chat_model()
    full_response = ""

    async for chunk in model.astream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_response += chunk.content

    return full_response


async def run_chat(state: GraphState):
    while True:
        last_message = state["messages"][-1] if state["messages"] else None

        if not isinstance(last_message, HumanMessage):
            user_input = input("\nYou: ").strip()

            if user_input.lower() in ["exit", "quit"]:
                break

            if not user_input:
                print("Message cannot be empty.")
                continue

            state["messages"].append(HumanMessage(content=user_input))
            _persist_state(state)

        print("AI: ", end="", flush=True)

        full_response = await _stream_assistant_response(state["messages"])

        state["messages"].append(AIMessage(content=full_response))
        _persist_state(state)

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
