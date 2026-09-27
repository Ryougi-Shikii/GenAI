from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

@tool
def add(first: int, second: int) -> int:
    """Addition of two numbers"""
    return first+second

model = init_chat_model(
    "gemini-3.5-flash-lite",
    model_provider="google-genai",
    max_tokens=2000,
    streaming=True,
)

checkpointer = InMemorySaver()
config = {'configurable' : {'thread_id': 'FangYuanChat'}}

agent = create_agent(
    model=model,
    tools=[add],
    system_prompt = "you're fang yuan from reverend insanity, you're also very helpful but twist is that you're exactly like fang yuan.",
    checkpointer=checkpointer,
)

for _ in range(3):
    prompt = input("enter: ")
    response = agent.invoke({'messages': [{"role": "user", "content": prompt}]}, config=config)
    print(response['messages'][-1].content_blocks[0]['text'])

