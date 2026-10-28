from __future__ import annotations
import os
import sys
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

# Ensure repository root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from gradio_client import Client
except ImportError:
    Client = None

try:
    from langchain_core.tools import tool
except ImportError:
    from langchain.tools import tool

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, deformed, bad anatomy, bad hands, missing fingers, extra digits, "
    "blurry, mutated, text, watermark, signature, speech bubble"
)

_gradio_client = None


def get_animagine_client():
    """Connect to Animagine XL 3.1 via Private Google Colab (Unlimited) or HF ZeroGPU Space."""
    global _gradio_client
    if _gradio_client is None and Client is not None:
        colab_url = os.environ.get("ANIMAGINE_URL") or os.environ.get("COLAB_GRADIO_URL")
        if colab_url:
            colab_url = colab_url.strip().rstrip("/")
            try:
                _gradio_client = Client(colab_url)
                return _gradio_client
            except Exception:
                pass

        token = os.environ.get("huggingface_colab") or os.environ.get("HF_TOKEN")
        try:
            if token:
                _gradio_client = Client("cagliostrolab/animagine-xl-3.1", token=token)
            else:
                _gradio_client = Client("cagliostrolab/animagine-xl-3.1")
        except Exception:
            try:
                _gradio_client = Client("cagliostrolab/animagine-xl-3.1")
            except Exception:
                _gradio_client = None
    return _gradio_client


def _create_cinematic_fallback(prompt: str, out_path: Path):
    """Fallback backdrop if external APIs are unreachable."""
    img = Image.new("RGB", (1280, 720), color=(15, 18, 30))
    draw = ImageDraw.Draw(img)
    
    for y in range(720):
        r = int(15 + (y / 720) * 15)
        g = int(18 + (y / 720) * 20)
        b = int(30 + (y / 720) * 35)
        draw.line([(0, y), (1280, y)], fill=(r, g, b))

    summary_text = prompt[:80] + "..." if len(prompt) > 80 else prompt
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except Exception:
        font = ImageFont.load_default()

    draw.text((80, 320), f"Scene: {summary_text}", fill=(200, 220, 255), font=font)
    fmt = "PNG" if out_path.suffix.lower() == ".png" else "JPEG"
    img.save(out_path, format=fmt, quality=95)


def generate_single_image(
    prompt: str,
    out_path: Path,
    seed: int = 42,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    ) -> Path:
    
    """Generate image using Animagine XL 3.1 or PIL fallback."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Animagine XL 3.1 Space
    client = get_animagine_client()
    if client is not None:
        for attempt in range(1, 3):
            try:
                res = client.predict(
                    prompt[:500],
                    negative_prompt,
                    seed,
                    1344,
                    768,
                    7.0,
                    20,
                    'Euler a',
                    '1344 x 768',
                    'Anime',
                    'Standard v3.1',
                    False,
                    0.5,
                    1.2,
                    True,
                    api_name='/run'
                )
                if res and isinstance(res, (tuple, list)) and len(res) > 0 and len(res[0]) > 0:
                    img_item = res[0][0]
                    temp_p = None
                    if isinstance(img_item, dict):
                        temp_p = img_item.get("image") or img_item.get("path") or img_item.get("name")
                    elif isinstance(img_item, str):
                        temp_p = img_item
                    elif isinstance(img_item, (tuple, list)) and len(img_item) > 0:
                        temp_p = img_item[0]

                    if temp_p and Path(temp_p).exists():
                        shutil.copy(temp_p, out_path)
                        return out_path
            except Exception:
                time.sleep(1)

    # 4. Fallback safeguard
    _create_cinematic_fallback(prompt, out_path)
    return out_path


@tool
def create_image(prompt: str, filename: str = "", seed: int = 42) -> str:
    """Generate an image from a detailed text prompt, saving the native output directly.
    
    Args:
        prompt: Visual prompt describing the scene, characters, setting, lighting, and mood.
        filename: Optional output filename or line ID (without extension).
        seed: Random seed for visual consistency.
        
    Returns:
        The string path to the generated image file.
    """
    print("> inside create_image tool")
    if not filename:
        ts = time.strftime("%Y%m%d_%H%M%S")
        filename = f"img_{ts}"
    
    f_path = Path(filename)
    if f_path.suffix or f_path.parent != Path("."):
        out_path = f_path.with_suffix(".png") if not f_path.suffix else f_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = ROOT_DIR / "Day 7" / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{filename}.png"

    print(f"\n>    generating image:")
    print(f"    filename: {filename}")
    print(f"    prompt:   {prompt}")
    print(f"    target:   {out_path}")
    
    try:
        saved_path = generate_single_image(prompt=prompt, out_path=out_path, seed=seed)
        print(f"    {filename} -> {saved_path}")
        print(f"> create_image tool is done!\n")
        return str(saved_path)
    except Exception as e:
        print(f"    {filename} FAILED: {e}")
        print(f"> create_image tool encountered an error!\n")
        raise e


if __name__ == "__main__":
    print("> Testing create_image tool independently...")
    test_prompt = "Futuristic laboratory with glowing blue monitors and neon lights"
    result = create_image.invoke({"prompt": test_prompt, "filename": "test_standalone_image"})
    print(f"> Result: {result}")
