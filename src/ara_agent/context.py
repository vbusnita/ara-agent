"""Workflow Context Layer (v1) — enrich captures with active-window
metadata so Ara knows which app/window the user is looking at, not
just the raw bytes.

This is intentionally tiny: one dataclass, two functions, a couple of
template strings. The goal is to close the biggest current information
gap (app/window awareness) without introducing memory systems, goal
tracking, or persistent storage. Those grow from here when there's
evidence we need them.

All future "what context does Ara know about this turn?" enrichments
should funnel through build_capture_packet — that's the chokepoint.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class ActiveContext:
    """Snapshot of what the user is looking at, taken at capture time."""

    app: str = "unknown"
    window: str = ""
    captured_at: float = field(default_factory=time.time)

    def short(self) -> str:
        """One-line human-readable summary for embedding in a prompt."""
        parts = [f"App: {self.app}"]
        if self.window:
            parts.append(f"Window: {self.window}")
        parts.append(time.strftime("%H:%M:%S", time.localtime(self.captured_at)))
        return " | ".join(parts)


# AppleScript that returns "app_name|||window_title" for the frontmost
# process. The triple-pipe delimiter keeps things parseable when window
# titles contain pipes, hyphens, etc.
_FRONTMOST_SCRIPT = (
    'tell application "System Events"\n'
    '  set fa to first application process whose frontmost is true\n'
    '  set an to name of fa\n'
    '  try\n'
    '    set wn to name of first window of fa\n'
    '  on error\n'
    '    set wn to ""\n'
    '  end try\n'
    'end tell\n'
    'return an & "|||" & wn'
)


def probe_active_window(timeout: float = 4.0) -> ActiveContext:
    """Frontmost app + window title via AppleScript / System Events.

    Returns a default ActiveContext (app="unknown", window="") on any
    failure — never raises. Worst case is degraded context, never a
    blocked capture.

    Requires Accessibility permission for the host process (which we
    already need for `run_applescript` and the global hotkey).
    """
    try:
        r = subprocess.run(
            ["osascript", "-e", _FRONTMOST_SCRIPT],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            log.warning(
                "probe_active_window: osascript exit %d, stderr=%r",
                r.returncode, r.stderr.strip()[:200],
            )
            return ActiveContext()
        parts = r.stdout.strip().split("|||")
        if len(parts) >= 2:
            ctx = ActiveContext(
                app=parts[0].strip() or "unknown",
                window=parts[1].strip(),
            )
            log.debug("active window: %s", ctx.short())
            return ctx
    except subprocess.TimeoutExpired:
        log.warning("probe_active_window: osascript timed out")
    except Exception:
        log.exception("probe_active_window failed")
    return ActiveContext()


# Packet template for vision captures. Single source of truth for how
# screen context is framed for the model — change it here, not at call
# sites. Substitutions: {ctx} (ActiveContext.short()) and {content}
# (the vision model's 1-sentence description).
#
# We used to support a second "ocr_text" template that injected raw
# Apple-OCR'd screen text directly into the realtime conversation. That
# path was deleted after every OCR-injection turn in three live tests
# stalled xAI's grok-voice-latest backend (Stream idle timeout / gRPC
# UNAVAILABLE) while every vision turn succeeded. Voice models want
# semantic context, not raw bytes.

_VISION_TEMPLATE = (
    "[Context — {ctx} | source: vision]\n"
    "Here is a brief description of what the user is looking at "
    "on their screen:\n\n"
    "{content}"
)


def build_capture_packet(kind: str, content: str) -> str:
    """Wrap a vision capture into a single block of text ready to inject
    into the realtime conversation. Includes app/window metadata and a
    timestamp so the model can reason about which surface the user is
    pointing at (and disambiguate from earlier captures).

    `kind` is kept for forward compatibility (future packet types) but
    currently only "vision" is meaningful — other kinds use the same
    template."""
    ctx = probe_active_window()
    packet = _VISION_TEMPLATE.format(ctx=ctx.short(), content=content)
    log.info(
        "context packet: kind=%s, app=%r, window=%r, content_len=%d",
        kind, ctx.app, ctx.window, len(content),
    )
    return packet
