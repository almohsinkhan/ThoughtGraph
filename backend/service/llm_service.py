import os
import asyncio
from unittest import result
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# load the api key from the .env file

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.0,
    max_retries=2,
    groq_api_key=groq_api_key
)

messages = [
    SystemMessage(content="You are a helpful programming assistant."),
]



async def run_chat():
    # use while loop to continue the conversation until the user types "exit"
    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            break
        
        messages.append(HumanMessage(content=user_input))

        print("AI: ", end="", flush=True)

        full_response = ""
        
        # async invoke the model with the messages and print the result
        async for chunk in model.astream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                full_response += chunk.content
        
        messages.append(AIMessage(content=full_response))

# Run the async loop
asyncio.run(run_chat())

