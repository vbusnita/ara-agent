"""Take a screenshot and either OCR it locally OR send it to xAI's
vision model for a visual description. Two complementary modes:

  • "Capture Text" — local OCR via Apple Vision. ~100 ms, no API call,
    zero token cost, no audio-thread GIL contention. Ara receives the
    literal characters on screen. Best for code, errors, docs, anything
    text-heavy where the user wants Ara to read exact content.

  • "Capture Image" — Pillow-shrunk JPEG sent to xAI grok-4.3 via the
    Responses API. ~2-3 s, costs tokens, briefly contends GIL during
    base64+JSON. Ara receives a 2–3 sentence description. Best for
    charts, diagrams, UI layouts, anything where visuals matter.

The capture step (`screencapture -i`) is shared — user can drag a
region OR press Space to switch to window-pick mode, ESC to cancel.
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


# ----- shared capture -----

def capture_screen() -> Optional[Path]:
    """Launch macOS interactive screenshot.

    `screencapture -i` starts in crosshair drag-region mode.
      • Drag to select a region, click to capture.
      • Press SPACE to toggle window-pick mode.
      • Press ESC to cancel.
    `-x` suppresses the shutter sound.

    Returns the PNG path, or None if cancelled / capture failed.
    """
    out = Path(tempfile.gettempdir()) / (
        f"ara-screenshot-{os.getpid()}-{int(time.time() * 1000)}.png"
    )
    try:
        subprocess.run(
            ["screencapture", "-i", "-x", str(out)],
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None
    if not out.exists() or out.stat().st_size == 0:
        return None
    return out


# ----- text mode: local OCR via Apple Vision -----

def ocr_sync(path: Path) -> str:
    """Extract text from an image using macOS Vision (local, no API).
    Returns recognized text joined by newlines, or empty string if Vision
    found no text or failed."""
    try:
        from Vision import (
            VNImageRequestHandler,
            VNRecognizeTextRequest,
        )
        from Foundation import NSURL
    except ImportError:
        print("⚠️  pyobjc-framework-Vision missing — install it to enable OCR")
        return ""

    url = NSURL.fileURLWithPath_(str(path))
    handler = VNImageRequestHandler.alloc().initWithURL_options_(url, {})

    request = VNRecognizeTextRequest.alloc().init()
    # 1 = VNRequestTextRecognitionLevelAccurate. Slower than "fast" but
    # materially better for code, UI text, and small fonts.
    request.setRecognitionLevel_(1)
    request.setUsesLanguageCorrection_(True)

    handler.performRequests_error_([request], None)

    lines = []
    for obs in (request.results() or []):
        candidates = obs.topCandidates_(1)
        if candidates and len(candidates) > 0:
            lines.append(candidates[0].string())
    return "\n".join(lines)


async def capture_and_extract_text() -> Optional[str]:
    """Capture → OCR locally → return text. None if cancelled or no text."""
    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(None, capture_screen)
    if path is None:
        return None
    try:
        text = await loop.run_in_executor(None, ocr_sync, path)
    except Exception as e:
        print(f"⚠️  OCR failed: {type(e).__name__}: {e}")
        return None
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
    text = text.strip() if text else ""
    return text if text else None


# ----- image mode: xAI grok-4.3 via Responses API -----

VISION_MODEL = "grok-4.3"
VISION_ENDPOINT = "https://api.x.ai/v1/responses"
VISION_PROMPT = (
    "You provide visual context for a voice agent that will speak about "
    "this screen. Reply in ONE plain sentence (~25 words max, no "
    "markdown, no lists, no paragraph breaks). Name the app and the "
    "single most relevant thing visible — the problem, the question, "
    "or the key content. Nothing else."
)


def _shrink_for_vision(path: Path) -> bytes:
    """Resize and recompress the raw screenshot before upload.

    A retina screenshot can be 2–5 MB PNG; base64-encoding and JSON-
    serializing that on the asyncio executor thread holds the GIL long
    enough to starve the audio callback (audible chops during post-
    screenshot response). Shrinking to max 1280px + JPEG q85 drops the
    payload ~10× while keeping enough fidelity for the vision model to
    read UI text.
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


def _extract_text(data: dict) -> str:
    """Pull text out of an xAI Responses payload. Tolerant of minor
    schema variation between `output_text` convenience field and the
    full `output[*].content[*]` walk."""
    if isinstance(data.get("output_text"), str):
        return data["output_text"].strip()
    parts = []
    for item in data.get("output", []):
        for c in item.get("content", []) or []:
            text = c.get("text") or c.get("output_text")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def describe_image_sync(path: Path, api_key: str) -> str:
    """POST to xAI Responses endpoint, return description text."""
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
    """Capture → vision-describe → return description. None if cancelled."""
    loop = asyncio.get_running_loop()
    path = await loop.run_in_executor(None, capture_screen)
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
