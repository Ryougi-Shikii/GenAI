from __future__ import annotations
import os
import re
import sys
import time
import shutil
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

# Ensure repository root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from langchain_core.tools import tool
except ImportError:
    from langchain.tools import tool

OPENROUTER_SPEECH_URL = "https://openrouter.ai/api/v1/audio/speech"
DEFAULT_FISH_MODEL = os.environ.get("FISH_AUDIO_MODEL", "fish-audio/s2.1-pro-free:free")

# Verified Fish Audio voice models
FISH_AUDIO_SPEAKERS: Dict[str, Dict[str, str]] = {
    "female_cute_anime": {
        "id": "d8f3b4c1199f4568a767ec54112d2b17",
        "gender": "female",
    },
    "female_expressive": {
        "id": "933563129e564b19a115bedd57b7406a",
        "gender": "female",
    },
    "female_egirl": {
        "id": "ca3007f96ae7499ab87d27ea3599956a",
        "gender": "female",
    },
    "male_hero_energetic": {
        "id": "802e3bc2b27e49c2995d23ef70e6ac89",
        "gender": "male",
    },
    "male_natural_adult": {
        "id": "536d3a5e000945adb7038665781a4aca",
        "gender": "male",
    },
    "male_smooth_cool": {
        "id": "bf322df2096a46f18c579d0baa36f41d",
        "gender": "male",
    },
    "male_detective_gritty": {
        "id": "59c2ef52360c4131bc7cb75644ebc225",
        "gender": "male",
    },
    "male_narrator": {
        "id": "e686ae649ee44f219a108aacba206c1a",
        "gender": "male",
    },
}

SPEAKER_POOLS: Dict[str, List[str]] = {
    "female": ["female_cute_anime", "female_egirl", "female_expressive"],
    "male": ["male_hero_energetic", "male_smooth_cool", "male_natural_adult", "male_detective_gritty"],
}


class VoiceClip(BaseModel):
    line_id: str
    speaker: str
    speaker_id: str = ""
    speaker_label: str = ""
    text: str
    emotion: str = ""
    path: str
    duration_ms: int = 2000


def get_speaker_id_by_key(key: str) -> Optional[str]:
    """Resolve a key to its 32-character Fish Audio hex ID."""
    if not key:
        return None
    if len(key) == 32 and all(c in "0123456789abcdefABCDEF" for c in key):
        return key.lower()
    speaker_info = FISH_AUDIO_SPEAKERS.get(key.lower())
    if speaker_info:
        return speaker_info["id"]
    return None


def resolve_speaker_for_character(
    char_name: str,
    gender: str = "female",
    age_archetype: str = "young_adult",
    used_speaker_ids: Optional[set] = None,
) -> tuple[str, str]:
    """Dynamically resolve a consistent Fish Audio speaker ID."""
    if used_speaker_ids is None:
        used_speaker_ids = set()

    clean_gender = (gender or "female").strip().lower()
    pool_key = "female" if "female" in clean_gender else "male"
    pool = SPEAKER_POOLS.get(pool_key, SPEAKER_POOLS["female"])

    chosen_key = pool[0]
    for key in pool:
        spk_id = FISH_AUDIO_SPEAKERS[key]["id"]
        if spk_id not in used_speaker_ids:
            chosen_key = key
            break

    chosen_id = FISH_AUDIO_SPEAKERS[chosen_key]["id"]
    used_speaker_ids.add(chosen_id)
    return chosen_id, chosen_key


def _probe_duration_ms(path: Path) -> int:
    """Probe audio file duration."""
    if shutil.which("ffprobe") and path.exists():
        try:
            out = subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ], text=True).strip()
            return max(500, int(float(out) * 1000))
        except Exception:
            pass
    if path.exists():
        size_bytes = path.stat().st_size
        return max(800, int(size_bytes / 16))
    return 2000


def build_fish_prompt(text: str, emotion: str | None = None) -> str:
    """Format dialogue with emotional delivery tags."""
    text = text.strip()
    if text.startswith("[") and "]" in text:
        return text
    if emotion and emotion.strip():
        clean_emotion = emotion.strip().rstrip(",")
        return f"[{clean_emotion}] {text}"
    return text


async def _synthesize_fish_audio(
    prompt: str,
    out_path: Path,
    speaker_id: str | None = None,
    model: str = DEFAULT_FISH_MODEL,
    max_retries: int = 3,
) -> bool:
    """Synthesize speech using Fish Audio via OpenRouter API."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("    Fish Audio: OPENROUTER_API_KEY not set, skipping")
        return False

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/universal-ai-studio",
        "X-Title": "Universal AI Production Pipeline",
    }
    payload = {
        "model": model,
        "input": prompt,
        "response_format": "mp3",
    }
    if speaker_id:
        payload["voice"] = speaker_id

    async with httpx.AsyncClient(timeout=45.0) as client:
        for attempt in range(1, max_retries + 1):
            try:
                res = await client.post(OPENROUTER_SPEECH_URL, headers=headers, json=payload)
                print(f"    Fish Audio attempt {attempt}: HTTP {res.status_code}, {len(res.content)} bytes")
                if res.status_code == 200 and len(res.content) > 1000:
                    out_path.write_bytes(res.content)
                    return True
                if res.status_code == 429:
                    await asyncio.sleep(2.0 * attempt)
                else:
                    await asyncio.sleep(1.0)
            except Exception:
                await asyncio.sleep(1.0)

    return False


async def _synthesize_edge_tts_fallback(text: str, out_path: Path, gender: str = "female") -> bool:
    """Safeguard fallback using Edge TTS."""
    try:
        import edge_tts
        clean_text = re.sub(r"\[.*?\]", "", text).strip() or text
        voice = "en-US-AnaNeural" if "female" in (gender or "female").lower() else "en-US-GuyNeural"
        print(f"    Edge TTS: voice={voice}, text={clean_text[:60]}...")
        comm = edge_tts.Communicate(text=clean_text, voice=voice)
        await comm.save(str(out_path))
        exists = out_path.exists()
        size = out_path.stat().st_size if exists else 0
        print(f"    Edge TTS: file exists={exists}, size={size} bytes")
        return exists and size > 500
    except Exception as e:
        print(f"    Edge TTS error: {type(e).__name__}: {e}")
        return False


async def synthesize_line(
    line_id: str,
    speaker: str,
    text: str,
    emotion: str = "",
    speaker_id: str | None = None,
    speaker_label: str = "",
    out_path: Path | None = None,
    gender: str = "female",
    model: str = DEFAULT_FISH_MODEL,
) -> VoiceClip:
    """Synthesize one line of dialogue with character voice and emotional delivery."""
    if out_path is None:
        out_dir = ROOT_DIR / "Day 7" / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{line_id}.mp3"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not speaker_id:
        speaker_id, speaker_label = resolve_speaker_for_character(char_name=speaker, gender=gender)

    fish_prompt = build_fish_prompt(text, emotion=emotion)
    print(f"    Fish Audio prompt: {fish_prompt[:80]}...")
    success = await _synthesize_fish_audio(fish_prompt, out_path, speaker_id=speaker_id, model=model)
    print(f"    Fish Audio success: {success}")
    if not success:
        print(f"    OpenRouter Fish Audio unavailable, using Edge TTS fallback for {line_id}...")
        fb_success = await _synthesize_edge_tts_fallback(text, out_path, gender=gender)
        print(f"    Edge TTS fallback success: {fb_success}")
        if not fb_success or not out_path.exists():
            raise RuntimeError(f"Audio synthesis failed for {line_id}. Could not write {out_path}")

    duration_ms = _probe_duration_ms(out_path)
    return VoiceClip(
        line_id=line_id,
        speaker=speaker,
        speaker_id=speaker_id or "",
        speaker_label=speaker_label or "",
        text=text,
        emotion=emotion,
        path=str(out_path),
        duration_ms=duration_ms,
    )


def _run_async(coro):
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing event loop (e.g. Jupyter, LangGraph internals).
        # Create a brand new loop in a separate thread to avoid conflicts.
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=120)
    else:
        return asyncio.run(coro)


@tool
def generate_voice(
    text: str,
    speaker: str = "Character",
    emotion: str = "",
    filename: str = "",
    voice: str = "female_cute_anime",
    gender: str = "female",
    model: str = DEFAULT_FISH_MODEL,
) -> str:
    """Synthesize speech audio for spoken dialogue.
    
    Args:
        text: The spoken dialogue or line text.
        speaker: Name of the character speaking the line.
        emotion: Emotional delivery descriptor (e.g. 'cheerful', 'angry', 'deadpan').
        filename: Output filename or line ID (e.g. 'L1'). Saves directly to Day 7/out/<filename>.mp3.
        voice: Voice preset key or speaker ID.
        gender: Gender of character ('female' or 'male').
        model: Model name for Fish Audio speech synthesis.
        
    Returns:
        The string path to the saved audio file.
    """
    print("> inside generate_voice tool")
    if not filename:
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"voice_{ts}"

    f_path = Path(filename)
    if f_path.suffix or f_path.parent != Path("."):
        out_path = f_path.with_suffix(".mp3") if not f_path.suffix else f_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = ROOT_DIR / "Day 7" / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{filename}.mp3"

    speaker_id = get_speaker_id_by_key(voice)
    speaker_label = voice
    if not speaker_id:
        speaker_id, speaker_label = resolve_speaker_for_character(char_name=speaker, gender=gender)

    print(f"\n>    synthesizing voice:")
    print(f"    speaker:  {speaker} ({speaker_label})")
    print(f"    emotion:  {emotion if emotion else 'neutral'}")
    print(f"    text:     {text}")
    print(f"    target:   {out_path}")

    try:
        clip = _run_async(
            synthesize_line(
                line_id=filename,
                speaker=speaker,
                text=text,
                emotion=emotion,
                speaker_id=speaker_id,
                speaker_label=speaker_label,
                out_path=out_path,
                gender=gender,
                model=model,
            )
        )
        print(f"    {filename} -> {clip.path} (duration: {clip.duration_ms}ms)")
        print(f"> generate_voice tool is done!\n")
        return str(clip.path)
    except Exception as e:
        print(f"    {filename} FAILED: {e}")
        print(f"> generate_voice tool encountered an error!\n")
        raise e


if __name__ == "__main__":
    print("> Testing generate_voice tool independently...")
    test_text = "Good evening! Voice generation system is fully operational."
    result = generate_voice.invoke({
        "text": test_text,
        "speaker": "Hatsune Miku",
        "emotion": "cheerful",
        "filename": "test_voice",
    })
    print(f"> Result: {result}")
