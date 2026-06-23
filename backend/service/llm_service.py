import os
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

# use while loop to continue the conversation until the user types "exit"
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    messages.append(HumanMessage(content=user_input))
    result = model.invoke(messages)
    print(f"AI: {result.content}")
    messages.append(AIMessage(content=result.content))
