import os
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from typing import TypedDict, List
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver



# load the api key from the .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# initialize the model with the api key and the model name
model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.0,
    max_retries=2,
    groq_api_key=groq_api_key
)

messages = [
    SystemMessage(content="You are a helpful programming assistant."),
]

class GraphState(TypedDict):
    messages: List[BaseMessage]



async def run_chat(state: GraphState):
    # use while loop to continue the conversation until the user types "exit"
    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            break
        
        state["messages"].append(HumanMessage(content=user_input))

        print("AI: ", end="", flush=True)

        full_response = ""
        
        # async invoke the model with the messages and print the result
        async for chunk in model.astream(state["messages"]):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
        
        state["messages"].append(AIMessage(content=full_response))

    return state

workflow = StateGraph(GraphState)
workflow.add_node("run_chat", run_chat)
workflow.add_edge(START, "run_chat")
workflow.add_edge("run_chat", END)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)


async def main():
    result = await app.ainvoke(
        {"messages": messages},
        config={"configurable": {"thread_id": "llm_service_chat"}},
    )
    print("Workflow result:", result)


if __name__ == "__main__":
    asyncio.run(main())


