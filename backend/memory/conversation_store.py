from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

BASE_DIR = Path(__file__).resolve().parent
CONVERSATION_FILE = BASE_DIR / "conversations.json"
SYSTEM_MESSAGE = SystemMessage(content="You are a helpful assistant.")


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def _message_to_record(message: BaseMessage) -> dict[str, str]:
    return {"role": message.type, "content": message.content}


def _record_to_message(record: dict[str, Any]) -> BaseMessage:
    role = record.get("role", "human")
    content = record.get("content", "")

    if role == "system":
        return SystemMessage(content=content)
    if role == "ai":
        return AIMessage(content=content)
    return HumanMessage(content=content)


def load_conversations() -> list[dict[str, Any]]:
    if not CONVERSATION_FILE.exists():
        return []

    with CONVERSATION_FILE.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    if isinstance(data, list):
        return data

    return []


def save_conversations(conversations: list[dict[str, Any]]) -> None:
    with CONVERSATION_FILE.open("w", encoding="utf-8") as file_handle:
        json.dump(conversations, file_handle, indent=2, ensure_ascii=False)


def upsert_conversation(conversation: dict[str, Any]) -> None:
    conversations = load_conversations()

    for index, existing in enumerate(conversations):
        if existing.get("thread_id") == conversation.get("thread_id"):
            conversations[index] = conversation
            break
    else:
        conversations.append(conversation)

    save_conversations(conversations)


def format_conversation_label(conversation: dict[str, Any]) -> str:
    title = conversation.get("title", "Untitled conversation")
    return title.strip() or "Untitled conversation"


def build_state_from_conversation(conversation: dict[str, Any]) -> dict[str, Any]:
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


def build_conversation_record(
    thread_id: str,
    title: str,
    messages: list[BaseMessage],
) -> dict[str, Any]:
    return {
        "thread_id": thread_id,
        "title": title,
        "messages": [_message_to_record(message) for message in messages],
    }
