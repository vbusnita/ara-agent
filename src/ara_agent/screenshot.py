"""Screen-perception flows for ara-agent. Two entry points share the
same interactive `screencapture -i` step:

  • capture_and_describe() — Pillow-shrunk JPEG → xAI grok-4.3 vision.
    Returns a one-sentence semantic description suitable for injecting
    as conversation context. The default capture path triggered by the
    overlay's Capture menu / hotkey.

  • capture_and_clean_for_reading() — Apple Vision OCR (local, exact
    transcription via `ocr_sync`) → xAI grok-4.3 chat (strips code,
    URLs, chrome, timestamps, etc.). Returns prose-only text suitable
    for reading aloud. Invoked by the read_screen_region_text tool
    when the user asks the agent to read a wall of text.

Apple OCR has one job here: faithfully transcribe pixels into the
exact text on screen, which the cleaner then judges for "readability."
Vision models would paraphrase that first step — wrong tool for it.

The capture step is interactive: drag to select a region, Space to
toggle window-pick mode, ESC to cancel.
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
from typing import Optional, Tuple

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


async def capture_and_describe(
    api_key: str,
) -> Optional[Tuple[str, Path]]:
    """Capture → vision-describe → return (description, screenshot_path).
    None if cancelled or describe failed.

    The path is RETURNED to the caller rather than deleted here so the
    read_screen_region_text tool can reuse the same screenshot for OCR
    without forcing the user to drag-select the same region twice. The
    caller is responsible for unlinking it (AraAgent owns the lifecycle
    via self._last_capture_path)."""
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
        # Failure → caller won't get a path to cache, so clean up here.
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None
    except Exception as e:
        print(f"⚠️  Vision call failed: {type(e).__name__}: {e}")
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None
    return description, path


# ----- read-aloud mode: Apple OCR + grok-4.3 prose cleaning -----
#
# Apple OCR gives us exact transcription (its strength); grok-4.3 gives
# us "what's worth speaking aloud" judgment (its strength). Compose the
# two: OCR for fidelity, model for taste.

READING_MODEL = "grok-4.3"
READING_ENDPOINT = "https://api.x.ai/v1/responses"

# Sentinel the cleaner returns when nothing on screen is worth reading
# (pure code, all chrome, etc.). Tool layer translates it to a friendly
# user message instead of having the voice agent say literal junk.
NO_PROSE_SENTINEL = "NO_PROSE_FOUND"

READING_PROMPT = (
    "You are preparing text that will be spoken aloud to a user by a "
    "voice assistant. Below is text OCR'd from the user's screen. "
    "Return ONLY the natural-language prose that should be read aloud "
    "— strip everything else.\n\n"
    "STRIP OUT:\n"
    "- Source code, code snippets, syntax markers (braces, semicolons, "
    "type annotations, imports, function signatures)\n"
    "- URLs, file paths, shell commands, terminal output\n"
    "- Navigation chrome: isolated menu items, button labels, tab "
    "labels, breadcrumbs, side-nav links\n"
    "- Timestamps, log lines, structured numeric data, tables\n"
    "- Cookie notices, ad placeholders, copyright footers, version "
    "numbers, share-button labels\n"
    "- Repeated separators or garbled OCR fragments\n\n"
    "KEEP:\n"
    "- Article body, email body, message text, document prose\n"
    "- Headings that flow into the prose\n"
    "- Bullet points that are full sentences\n"
    "- Direct quotes\n\n"
    "FORMAT:\n"
    "Return only the cleaned text exactly as it should be spoken. No "
    "preamble like \"Here is the text:\". No commentary. Punctuation "
    "should support natural speech rhythm. If the screen contains no "
    f"meaningful prose worth reading aloud, return exactly: {NO_PROSE_SENTINEL}\n\n"
    "OCR'd text:\n---\n"
)


def clean_ocr_for_reading_sync(ocr_text: str, api_key: str) -> str:
    """Send OCR text to grok-4.3 with the reading prompt, return the
    model's cleaned output. Raises on transport/API failure."""
    body = json.dumps({
        "model": READING_MODEL,
        "input": [{
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": READING_PROMPT + ocr_text + "\n---\n",
            }],
        }],
    }).encode()
    req = urllib.request.Request(
        READING_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    text = _extract_text(data)
    if not text:
        raise RuntimeError(f"Empty reading-cleaner response: {data}")
    return text


async def capture_and_clean_for_reading(
    api_key: str,
    existing_path: Optional[Path] = None,
) -> Optional[str]:
    """Apple OCR → grok-4.3 cleaning → return prose-only text.

    If `existing_path` is provided (and the file still exists on disk),
    OCR runs against that screenshot instead of prompting the user to
    drag-select a new region. This is the cached-last-capture path: the
    user hits ⌘⇧A once, then asks Ara to read — same screenshot is
    reused, no second selection required. If the cached file is missing
    or no path was passed, falls back to a fresh interactive capture.

    Returns None on user-cancel, empty OCR, or API failure. Returns the
    literal NO_PROSE_SENTINEL string when the screen has no speakable
    prose — caller should turn that into a user-facing message rather
    than reading the sentinel aloud."""
    loop = asyncio.get_running_loop()

    using_cached = (
        existing_path is not None and existing_path.exists()
    )
    if using_cached:
        path = existing_path
    else:
        path = await loop.run_in_executor(None, capture_screen)
        if path is None:
            return None
    try:
        raw_text = await loop.run_in_executor(None, ocr_sync, path)
    except Exception as e:
        print(f"⚠️  OCR failed: {type(e).__name__}: {e}")
        return None
    finally:
        # Only clean up screenshots WE took. The cached path is owned
        # by AraAgent and will be cleaned up when the next capture
        # replaces it (or on agent shutdown).
        if not using_cached:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return None
    try:
        cleaned = await loop.run_in_executor(
            None, clean_ocr_for_reading_sync, raw_text, api_key,
        )
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode(errors="replace")[:500]
        except Exception:
            pass
        print(f"⚠️  Reading cleaner API error: {e.code} {e.reason}"
              + (f" — {body}" if body else ""))
        return None
    except Exception as e:
        print(f"⚠️  Reading cleaner failed: {type(e).__name__}: {e}")
        return None
    cleaned = cleaned.strip()
    return cleaned if cleaned else None
