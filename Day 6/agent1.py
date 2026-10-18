from dotenv import load_dotenv
load_dotenv()

import json, os, requests, base64
from pathlib import Path

from typing import Annotated, Sequence, TypedDict
from pydantic import BaseModel, TypeAdapter, Field
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware

class AgentState(TypedDict):
    """ State for the graph """
    messages : Annotated[Sequence[BaseMessage], add_messages]
    query: str
    script : dict
    prompts: dict
    images: list[dict]


class Line(BaseModel):
    """ line schema """
    id: str
    speaker: str
    text: str
    emotion: str
    intensity: float
    rate: float
    pauses: list[float] = []
    emphasis: list[str] = []
    visual_cue: str

class Script(BaseModel):
    lines: list[Line]


class ImagePrompt(BaseModel):
    line_id: str = Field(description="id of the corresponding Script line, e.g. 'L1'")
    prompt:  str = Field(description="detailed text-to-image prompt for that frame")

class Prompts(BaseModel):
    prompts: list[ImagePrompt]

def dump(obj, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

MODELS = {
    "creative": {
        "primary": ("openai/gpt-oss-120b", "groq"),
        "fallbacks": [
            ("gemini-3.5-flash-lite", "google-genai"),
        ],
        "temperature": 0.9,
        "max_tokens":  4000,
        "timeout":     90,
    },
    "ReAct": {
        "primary": ("gemini-3.5-flash-lite", "google-genai"),
        "fallbacks": [
            ("gemini-3.1-flash-lite", "google-genai"),
            ("openai/gpt-oss-120b", "groq"),
            ("qwen/qwen3.8-27b", "groq"),
        ],
        "temperature": 0.3,
        "max_tokens":  3000,
        "timeout":     90,
    },
}

def _build_model(role: str):
    """role -> creative or ReAct"""
    spec = MODELS[role]

    def _make(entry: tuple[str, str]):
        name, provider = entry
        return init_chat_model(
            name, model_provider=provider,
            temperature=spec["temperature"],
            timeout=spec.get("timeout", 180),
            max_tokens=spec["max_tokens"],
        )

    primary = _make(spec["primary"])
    fallbacks = [_make(e) for e in spec.get("fallbacks", [])]
    return primary.with_fallbacks(fallbacks, exceptions_to_handle=(Exception,)) if fallbacks else primary


def write_script(state: AgentState) -> dict:
    print("> inside write_script node")
    print("> creating agent")
    SCRIPT_PROMPT = """You are a scriptwriter for voiced content.

        Given a brief, produce a Script — a sequence of spoken lines with delivery metadata.

        OUTPUT FORMAT — STRICT:
        Return ONLY a JSON array. No prose before or after. No markdown fences. No bullet points.
        Each element is an object with exactly these keys:

        {"id": "L1", "speaker": "Name", "text": "...",
        "emotion": "lowercase_word", "intensity": 0.0-1.0, "rate": 0.5-2.0,
        "pauses": [0.0-2.0, ...], "emphasis": ["word1", "word2"],
        "visual_cue": "one short phrase"}

        RULES:
        - Every line MUST have a unique id ("L1", "L2", ...).
        - `speaker` is a character name, not a voice name.
        - `emotion` is a single lowercase word (deadpan, amused, cold, urgent, warm...).
        - `intensity` is 0..1. 0.4-0.6 normal, 0.9+ is a peak.
        - `rate` 1.0 normal, 0.85 for gravitas, 1.15 for excitement.
        - `emphasis` is a list of 1-3 words.
        - `text` contains no stage directions.
        - Write for the ear. Short sentences.

        EXAMPLE (one element — your array will have many):
        [{"id":"L1","speaker":"Alice","text":"Ready when you are.","emotion":"warm",
        "intensity":0.5,"rate":1.0,"pauses":[],"emphasis":["ready"],"visual_cue":"Alice smiles"}]
        """
    
    agent = create_agent(
        model=_build_model("creative"),
        system_prompt=SCRIPT_PROMPT,
        response_format=Script,
        middleware=[
            ModelRetryMiddleware(max_retries=3),
            ToolRetryMiddleware(max_retries=2),
        ],
    )
    
    query = state["query"]
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
    )
    
    sr = result.get("structured_response")
    if sr is None:
        raise RuntimeError(f"no structured_response from script agent: {result}")
    
    print("> write_script node is done!")
    
    return {"script": sr.model_dump()}
    


CF_ACCOUNT = os.environ["CF_ACCOUNT_ID"]
CF_TOKEN   = os.environ["CF_API_TOKEN"]
IMAGE_DIR  = Path("Day 6/images")

CF_URL = (
    f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT}/ai/run/"
    "@cf/black-forest-labs/flux-2-klein-4b"
)
    # "@cf/black-forest-labs/flux-1-schnell"

@tool
def generate_image(prompt: str, filename: str) -> str:
    """Generate one image from a detailed prompt. Returns the saved file path."""
    r = requests.post(
        CF_URL,
        headers={"Authorization": f"Bearer {CF_TOKEN}"},
        json={"prompt": prompt, "steps": 8},
        timeout=90,
    )

    if r.status_code != 200:
        raise RuntimeError(f"CF {r.status_code}: {r.text[:300]}")

    ct = r.headers.get("Content-Type", "").lower()
    if ct.startswith("image/"):
        data = r.content
    elif ct.startswith("application/json"):
        b64 = r.json().get("result", {}).get("image")
        if not b64:
            raise RuntimeError(f"CF returned json without image: {r.text[:300]}")
        if b64.startswith("data:image"):
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
    else:
        raise RuntimeError(f"unexpected content-type {ct}: {r.text[:300]}")

    IMAGE_DIR.mkdir(exist_ok=True)
    path = IMAGE_DIR / f"{filename}.png"
    path.write_bytes(data)
    return str(path)


def create_pictures(state: AgentState) -> dict:
    print("> inside create_prompt_for_pictures node")
    IMAGES_PROMPT = """You are a storyboard artist and prompt engineer.

        You will receive a brief and a Script — an ordered sequence of spoken lines.
        For EACH line you must craft ONE text-to-image prompt such that, taken together,
        the prompts form a visually consistent frame-by-frame sequence
        (like an animatic or a visual-novel cutscene) that matches the dialogue.

        You DO NOT generate images. You only return prompts.

        OUTPUT — STRICT:
        Return ONLY a JSON object:
        {"prompts":[{"line_id":"L1","prompt":"..."}, {"line_id":"L2","prompt":"..."}]}
        One entry per Script line, in the same order. No prose, no markdown fences.

        STEP 1 — STYLE LOCK (decide once, reuse verbatim in every prompt):
        USE SAME style block for the consistency over images unless needed.
        Add this in the prompt: "16:9 composition, no text, no watermark"
        Keep palette, lighting direction, and rendering style identical across frames
        unless the dialogue explicitly shifts mood.

        STEP 2 — CHARACTER SHEET (decide once per speaker, reuse verbatim):
        For each unique `speaker` invent a short, fixed appearance signature
        (hair color/style, eye color, outfit, 1 distinguishing feature).
        Repeat it word-for-word in every prompt where that speaker appears.
        Same character in different frames must read as the same person.

        STEP 3 — PER-FRAME COMPOSITION:
        - Subject, action and setting MUST follow the line's `text`.
        - Use `visual_cue` ONLY as a hint for framing / mood — do not copy it.
        - Translate `emotion` and `intensity` into expression, pose and shot size:
                intensity <= 0.3  -> wide / medium shot
                0.4 – 0.6        -> medium shot
                >= 0.8           -> close-up, stronger expression
        - Mention the speaker by name.
        - 3–4 sentences. Concrete visual nouns. No engine jargon.

        CONSTRAINTS:
        - Exactly one entry per line. No invented lines. Same order as the Script.
        - No on-screen text, subtitles, speech bubbles, or logos in the frame.
        - The whole set must look like it came from the same production.
        """
    
    script = Script.model_validate(state["script"])
    
    print("> creating agent")
    agent = create_agent(
        model=_build_model("creative"),
        system_prompt=IMAGES_PROMPT,
        response_format=Prompts,
        middleware=[
            ModelRetryMiddleware(max_retries=3),
            ToolRetryMiddleware(max_retries=2),
        ],
    )
    
    user_msg = (
        f"BRIEF:\n{state['query']}\n\n"
        f"SCRIPT:\n{state['script']}"
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
    )
    
    structured = result.get("structured_response")
    if structured is None:
        raise RuntimeError(f"no structured_response from prompts agent: {result}")
    
    print("> prompts are done!")
    
    by_id = {p.line_id: p.prompt for p in structured.prompts}
    images = []
    for line in script.lines:
        prompt = by_id.get(line.id) or line.visual_cue
        print(f"\n>    generating images:")
        try:
            path = generate_image.invoke({"prompt": prompt, "filename": line.id})
            images.append({"id": line.id, "path": path, "prompt": prompt, "ok": True})
            print(f"    {line.id} -> {path}")
        except Exception as e:
            images.append(
                {"id": line.id, "path": None, "prompt": prompt,
                 "ok": False, "error": str(e)}
            )
            print(f"    {line.id} FAILED: {e}")
        print(f"\n\n    for the prompt: {prompt}\n\n    for the line: {line.text}\n\n")
    print("> create_pictures node is Done!")

    return {
        "prompts": structured.model_dump(),
        "images": images,
    }



flow = StateGraph(AgentState)

flow.add_node("write_script", write_script)
flow.add_node("create_pictures", create_pictures)

flow.add_edge(START, "write_script")
flow.add_edge("write_script", "create_pictures")
flow.add_edge("create_pictures", END)

app = flow.compile()



content = "Hatsune Miku is flirting with Teto, six lines"
res = app.invoke({"messages": [HumanMessage(content=content)], "query":content})

dump(res["script"],  "Day 6/out/script.json")
dump(res["prompts"], "Day 6/out/prompts.json")
dump(res["images"],  "Day 6/out/images.json")
