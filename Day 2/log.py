from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from langchain.messages import HumanMessage, AIMessage, SystemMessage

model = init_chat_model(
    "gemini-3.5-flash-lite",
    model_provider="google-genai",
)

SYSTEM_PROMPT = """You are a Fang Yuan from novel Reverend Insanity.
You're helpful and all other traits of Fang Yuan's personality.
"""

human = HumanMessage(content=[
    {'type': 'text', 'text': 'whats this image?'},
    {'type': 'image', 'url': 'https://img.freepik.com/premium-photo/random-image_590832-6671.jpg'},
])
message = [SystemMessage(SYSTEM_PROMPT), human]


response = model.invoke(message)
print(response.content_blocks[0]['text'])