"""Take a window screenshot via macOS, describe it via xAI grok-vision.

The xAI realtime/voice API doesn't accept image inputs directly, so we
use the REST chat-completions endpoint with a vision-capable model to
turn the screenshot into a textual description, which we then inject
into the realtime conversation as a user message.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from PIL import Image

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]


# xAI Responses endpoint (newer unified API; replaces /v1/chat/completions
# for vision). grok-4.3 is the current general-purpose multimodal model
# and supports images up to 20 MiB.
VISION_MODEL = "grok-4.3"
VISION_ENDPOINT = "https://api.x.ai/v1/responses"
VISION_PROMPT = (
    "You provide visual context to a conversational voice agent. "
    "Describe this screenshot in 2-3 short sentences. Name the app or "
    "site and summarize the most relevant content (key text, code, "
    "errors, or UI state). Be specific and concrete; skip filler — "
    "the agent will answer the user using only your description."
)


def capture_window() -> Optional[Path]:
    """Run macOS interactive window screenshot. Returns the PNG path, or
    None if the user cancelled (pressed Esc) or capture failed.

    `screencapture -i -w` puts macOS into the same modal window-picker
    UI that's used by ⌘⇧4 then space. `-x` suppresses the shutter sound.
    """
    out = Path(tempfile.gettempdir()) / (
        f"ara-screenshot-{os.getpid()}-{int(time.time() * 1000)}.png"
    )
    try:
        subprocess.run(
            ["screencapture", "-i", "-w", "-x", str(out)],
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None
    if not out.exists() or out.stat().st_size == 0:
        # User pressed Esc — screencapture creates no file in that case.
        return None
    return out


def _extract_text(data: dict) -> str:
    """Pull the text out of an xAI Responses payload.

    Tries the convenience `output_text` field first, then walks
    `output[*].content[*]` for any text-bearing items. Robust against
    minor shape changes in the response.
    """
    if isinstance(data.get("output_text"), str):
        return data["output_text"].strip()
    parts = []
    for item in data.get("output", []):
        for c in item.get("content", []) or []:
            text = c.get("text") or c.get("output_text")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _shrink_for_vision(path: Path) -> bytes:
    """Resize and recompress the raw screenshot before upload.

    A retina window screenshot is often a 2–5 MB PNG; base64-encoding and
    JSON-serializing that on the asyncio executor thread holds the GIL
    long enough to starve the audio callback, which manifests as audible
    chops during the post-screenshot response. Shrinking to a max 1280px
    dimension and recompressing as JPEG drops the payload ~10× without
    losing the visual fidelity vision models need to read UI text.
    """
    MAX_DIM = 1280
    JPEG_QUALITY = 85
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_DIM:
        if w >= h:
            new_w, new_h = MAX_DIM, max(1, int(h * MAX_DIM / w))
        else:
            new_w, new_h = max(1, int(w * MAX_DIM / h)), MAX_DIM
        img = img.resize((new_w, new_h), LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()


def describe_image_sync(path: Path, api_key: str) -> str:
    """Synchronous: POST to xAI Responses endpoint, return description text."""
    jpeg_bytes = _shrink_for_vision(path)
    b64 = base64.b64encode(jpeg_bytes).decode()
    body = json.dumps({
        "model": VISION_MODEL,
        "input": [{
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
                {"type": "input_text", "text": VISION_PROMPT},
            ],
        }],
    }).encode()

    req = urllib.request.Request(
        VISION_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    text = _extract_text(data)
    if not text:
        raise RuntimeError(f"Empty vision response: {data}")
    return text


async def capture_and_describe(api_key: str) -> Optional[str]:
    """Full pipeline: pick a window → describe it. Returns the description
    or None if cancelled / failed. Runs the blocking syscalls in the
    default executor so the asyncio loop stays responsive.
    """
    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(None, capture_window)
    if path is None:
        return None
    try:
        description = await loop.run_in_executor(
            None, describe_image_sync, path, api_key
        )
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode(errors="replace")[:500]
        except Exception:
            pass
        print(f"⚠️  Vision API error: {e.code} {e.reason}"
              + (f" — {body}" if body else ""))
        return None
    except Exception as e:
        print(f"⚠️  Vision call failed: {type(e).__name__}: {e}")
        return None
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
    return description
