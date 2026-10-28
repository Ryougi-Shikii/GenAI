from dotenv import load_dotenv
load_dotenv()

import json, os, sys
from pathlib import Path

# Ensure repository root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import Annotated, Sequence, TypedDict
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware

# Standalone tools from tools folder
from tools import create_image, generate_voice

OUT_DIR = ROOT_DIR / "Day 7" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class AgentState(TypedDict):
    """State for the multi-agent graph"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    script: dict
    images: list[dict]
    audio: list[dict]


class Line(BaseModel):
    """Line schema matching agent1.py"""
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


def dump_json(obj, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


MODELS = {
    "creative": {
        "primary": ("openai/gpt-oss-120b", "groq"),
        "fallbacks": [
            ("gemini-3.5-flash-lite", "google-genai"),
        ],
        "temperature": 0.9,
        "max_tokens": 4000,
        "timeout": 90,
    },
    "ReAct": {
        "primary": ("gemini-3.5-flash-lite", "google-genai"),
        "fallbacks": [
            ("gemini-3.1-flash-lite", "google-genai"),
            ("openai/gpt-oss-120b", "groq"),
            ("qwen/qwen3.8-27b", "groq"),
        ],
        "temperature": 0.2,
        "max_tokens": 4000,
        "timeout": 90,
    },
}


def _build_model(role: str):
    """role -> creative or ReAct"""
    spec = MODELS[role]

    def _make(entry: tuple[str, str]):
        name, provider = entry
        return init_chat_model(
            name,
            model_provider=provider,
            temperature=spec["temperature"],
            timeout=spec.get("timeout", 180),
            max_tokens=spec["max_tokens"],
        )

    primary = _make(spec["primary"])
    fallbacks = [_make(e) for e in spec.get("fallbacks", [])]
    return primary.with_fallbacks(fallbacks, exceptions_to_handle=(Exception,)) if fallbacks else primary


# ============================================================================
# AGENT 1: SCRIPT WRITER AGENT
# ============================================================================
def write_script_node(state: AgentState) -> dict:
    print("> inside write_script node")
    print("> creating script writer agent (Agent 1)")
    
    SCRIPT_PROMPT = """You are a master scriptwriter for voiced and visual multimedia.

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
        - `visual_cue` describes interactive body language and gaze toward the listener (e.g. "leaning closer to the other character with a tease", "turning with an amused smirk").
        - Write for the ear. Short sentences.
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
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    sr = result.get("structured_response")
    if sr is None:
        raise RuntimeError(f"no structured_response from script agent: {result}")

    print("> write_script node is done!")
    return {"script": sr.model_dump()}


# ============================================================================
# AGENT 2: IMAGE GENERATION AGENT (HAS TOOL: create_image)
# ============================================================================
def generate_images_node(state: AgentState) -> dict:
    print("> inside generate_images node")
    print("> creating image generation agent with create_image tool (Agent 2)")

    script_data = state["script"]
    script = Script.model_validate(script_data)

    IMAGE_AGENT_PROMPT = """You are an expert cinematic storyboard artist and art director for visual novel dialogue scenes.
You have access to the tool `create_image(prompt: str, filename: str)`.

YOUR OBJECTIVE:
Generate a visually cohesive, frame-by-frame conversation sequence where the characters in the Script feel like they are directly talking to each other in the exact same scene.

RULES FOR VISUAL CONSISTENCY & CONVERSATIONAL DIALOGUE:

1. SHARED LOCATION & STYLE LOCK (Decide once from the story brief/script, repeat in EVERY prompt):
- Establish ONE shared environment/room and lighting setup that fits the story, and keep it IDENTICAL across all frames.
- Do NOT jump between different environments across lines.

2. FIXED CHARACTER APPEARANCE (Decide once per character, repeat verbatim):
- For each unique speaker/character in the script, establish a fixed visual signature (hair style/color, eye color, outfit, distinguishing features).
- Repeat that exact character description word-for-word whenever that character appears in any frame.

3. CONVERSATIONAL COMPOSITION (Make dialogues feel like real dialogues):
- Frames must depict interpersonal interaction:
  • Use two-shots and over-the-shoulder (OTS) camera angles showing the characters facing, reacting to, or interacting with one another.
  • Keep conversational partners visible or established in the scene.
  • Characters must make eye contact, turn toward each other, gesture, or react directly to what was said.
  • Facial expressions and mouth poses must match speaking/listening dynamics (speaking character with mouth slightly open, listening character reacting to the line).

4. EXECUTION:
For EVERY line in the Script sequentially (L1, L2, ...):
- Construct a rich 3-4 sentence prompt following the rules above.
- Call `create_image(prompt=prompt, filename=line.id)`.
- Never skip any line.
"""

    image_agent = create_agent(
        model=_build_model("ReAct"),
        tools=[create_image],
        system_prompt=IMAGE_AGENT_PROMPT,
        middleware=[
            ModelRetryMiddleware(max_retries=3),
            ToolRetryMiddleware(max_retries=2),
        ],
    )

    prompt_content = (
        f"Generate an illustration for each line in this script using create_image:\n\n"
        f"BRIEF: {state['query']}\n\n"
        f"SCRIPT:\n{json.dumps(script_data, indent=2)}"
    )

    print("\n>    executing image generation agent...")
    try:
        image_agent.invoke({"messages": [{"role": "user", "content": prompt_content}]})
    except Exception as e:
        print(f"> Image agent execution notice: {e}")

    images = []
    speakers_in_script = list({l.speaker for l in script.lines})
    # Ensure every single line has its .png file saved in Day 7/out/
    for line in script.lines:
        target_file = OUT_DIR / f"{line.id}.png"
        if not target_file.exists():
            print(f"\n>    generating image for line {line.id} via create_image tool:")
            other_speaker = next((s for s in speakers_in_script if s != line.speaker), "the listener")
            prompt = f"Anime conversation scene, {line.speaker} talking to {other_speaker}, {line.visual_cue}, {line.emotion} expression, two-shot conversational angle, consistent shared setting, cinematic lighting, masterpiece"
            try:
                saved_path = create_image.invoke({"prompt": prompt, "filename": line.id})
                images.append({
                    "id": line.id,
                    "path": str(saved_path),
                    "prompt": prompt,
                    "ok": True,
                })
                print(f"    {line.id} -> {saved_path}")
            except Exception as e:
                images.append({
                    "id": line.id,
                    "path": None,
                    "prompt": prompt,
                    "ok": False,
                    "error": str(e),
                })
                print(f"    {line.id} FAILED: {e}")
        else:
            images.append({
                "id": line.id,
                "path": str(target_file),
                "ok": True,
            })
            print(f"    {line.id} -> {target_file}")

    print("> generate_images node is Done! All image files saved directly in Day 7/out.\n")
    return {"images": images}


# ============================================================================
# AGENT 3: VOICE GENERATION AGENT (HAS TOOL: generate_voice)
# ============================================================================
def generate_voices_node(state: AgentState) -> dict:
    print("> inside generate_voices node")
    print("> creating voice generation agent with generate_voice tool (Agent 3)")

    script_data = state["script"]
    script = Script.model_validate(script_data)

    VOICE_AGENT_PROMPT = """You are an audio director and voice synthesis supervisor.
        You have access to the tool `generate_voice(text: str, speaker: str, emotion: str, filename: str)`.

        Your job:
        For EVERY spoken line in the provided Script, you MUST call `generate_voice` to synthesize speech audio:
        - Provide the exact line text as `text`.
        - Provide the character name as `speaker`.
        - Provide the emotional tone as `emotion`.
        - Provide the line ID as `filename` (e.g., filename="L1", filename="L2"). It saves directly to Day 7/out/<filename>.mp3.
        - Call `generate_voice` for ALL lines in the script sequentially. Do not skip any line.
        """

    voice_agent = create_agent(
        model=_build_model("ReAct"),
        tools=[generate_voice],
        system_prompt=VOICE_AGENT_PROMPT,
        middleware=[
            ModelRetryMiddleware(max_retries=3),
            ToolRetryMiddleware(max_retries=2),
        ],
    )

    prompt_content = (
        f"Synthesize voice audio for every line in this script using generate_voice:\n\n"
        f"SCRIPT:\n{json.dumps(script_data, indent=2)}"
    )

    print("\n>    executing voice generation agent...")
    try:
        voice_agent.invoke({"messages": [{"role": "user", "content": prompt_content}]})
    except Exception as e:
        print(f"> Voice agent execution notice: {e}")

    audio = []
    # Ensure every single line has its .mp3 file saved in Day 7/out/
    for line in script.lines:
        target_file = OUT_DIR / f"{line.id}.mp3"
        if not target_file.exists():
            print(f"\n>    generating voice for line {line.id} via generate_voice tool:")
            try:
                saved_path = generate_voice.invoke({
                    "text": line.text,
                    "speaker": line.speaker,
                    "emotion": line.emotion,
                    "filename": line.id,
                })
                audio.append({
                    "id": line.id,
                    "speaker": line.speaker,
                    "text": line.text,
                    "emotion": line.emotion,
                    "path": str(saved_path),
                    "ok": True,
                })
                print(f"    {line.id} -> {saved_path}")
            except Exception as e:
                audio.append({
                    "id": line.id,
                    "speaker": line.speaker,
                    "text": line.text,
                    "emotion": line.emotion,
                    "path": None,
                    "ok": False,
                    "error": str(e),
                })
                print(f"    {line.id} FAILED: {e}")
        else:
            audio.append({
                "id": line.id,
                "speaker": line.speaker,
                "text": line.text,
                "emotion": line.emotion,
                "path": str(target_file),
                "ok": True,
            })
            print(f"    {line.id} -> {target_file}")

    print("> generate_voices node is Done! All audio files saved directly in Day 7/out.\n")
    return {"audio": audio}


# ============================================================================
# STATEGRAPH WORKFLOW (3 AGENTS)
# ============================================================================
flow = StateGraph(AgentState)

flow.add_node("write_script", write_script_node)
flow.add_node("generate_images", generate_images_node)
flow.add_node("generate_voices", generate_voices_node)

flow.add_edge(START, "write_script")
flow.add_edge("write_script", "generate_images")
flow.add_edge("generate_images", "generate_voices")
flow.add_edge("generate_voices", END)

app = flow.compile()


if __name__ == "__main__":
    content = "Hatsune Miku is flirting with Teto, six lines"
    print(f"\n==================== STARTING MULTI-AGENT RUN ====================")
    print(f"> Query: {content}")
    res = app.invoke({"messages": [HumanMessage(content=content)], "query": content})

    # Save structured metadata JSON files directly in Day 7/out
    dump_json(res["script"], OUT_DIR / "script.json")
    dump_json(res["images"], OUT_DIR / "images.json")
    dump_json(res["audio"], OUT_DIR / "audio.json")

    print(f"\n==================== RUN COMPLETED SUCCESSFULLY ====================")
    print(f"📁 All files saved directly in: {OUT_DIR}")
    print(f"   • Metadata:  script.json, images.json, audio.json")
    print(f"   • Images:    L1.png, L2.png, ...")
    print(f"   • Audio:     L1.mp3, L2.mp3, ...")
    print(f"====================================================================\n")
