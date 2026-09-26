from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from langchain.messages import HumanMessage, AIMessage, SystemMessage

@tool
def add(first: int, second: int) -> int:
    """Addition of two numbers"""
    return first + second
@tool
def sub(first: int, second: str) -> int:
    """Subtraction of two numbers"""
    return first - second
@tool
def mul(first: int, second: int) -> int:
    """Multiplication of two numbers"""
    return first * second

SYSTEM_PROMPT = """You are a Fang Yuan from novel Reverend Insanity.
You're helpful and all other traits of Fang Yuan's personality.
"""

model = init_chat_model(
    "gemini-3.5-flash-lite",
    model_provider="google-genai",
    max_tokens=2000,
    timeout=600,
    streaming=True,
)

agent = create_agent(
    model=model,
    tools=[add, sub, mul],
    system_prompt=SYSTEM_PROMPT,
)

conversation_history = []

def start():
    query = input("you: ")
    conversation_history.append(HumanMessage(query))
    
    result = agent.invoke({"messages": conversation_history})
    
    response = result["messages"][-1].content
    print(f'Fang Yuan: {response[0]['text']}\n\n')
    
    conversation_history.append(AIMessage(response))

    # adding complete state instead of just texts of both sides.
    
    #     states include: [   HumanMessage("What is 2 + 3?"),
    #                         AIMessage(tool_call = add(2, 3)),
    #                         ToolMessage("5"),
    #                         AIMessage("The answer is 5.")   ]

    # conversation_history.clear()
    # conversation_history.extend(result["messages"])

    

for _ in range(2):
    start()
    
print(conversation_history)