
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gemini-3.5-flash-lite",
    model_provider="google-genai",
    temperature=0.9,
    timeout=600,
    max_tokens=3000,
    streaming=True,
)

print(type(model))
print(model.profile)