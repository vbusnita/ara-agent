"""
ara-agent — Voice-first Grok computer agent
Core real-time voice client using xAI Voice Agent API
"""

import asyncio
import base64
import collections
import getpass
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

import websockets
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
import keyring

load_dotenv()

def get_api_key() -> str:
    try:
        key = keyring.get_password("xai-api-key", getpass.getuser())
        if key:
            return key
    except Exception:
        pass
    key = os.getenv("XAI_API_KEY")
    if key:
        return key
    raise ValueError(
        "No API key found. Either:\n"
        "  1. Store in macOS Keychain:\n"
        "     security add-generic-password -a \"$USER\" -s \"xai-api-key\" -w \"your-key\"\n"
        "  2. Or set XAI_API_KEY in your .env file"
    )


XAI_API_KEY = get_api_key()
VOICE = "ara"
MODEL = "grok-voice-latest"
ENDPOINT = f"wss://api.x.ai/v1/realtime?model={MODEL}"
SAMPLE_RATE = 24000

# Environment context exposed to Ara in the system prompt so she can
# refer to the right paths (instead of guessing or using literal "~"
# which subprocess.run / open() don't expand without shell=True).
USER_NAME = getpass.getuser()
HOME_DIR = os.path.expanduser("~")
LAUNCH_CWD = os.getcwd()


def _normalize_path(p: str) -> str:
    """Expand ~ and $VAR in a user-supplied path. Tools that call
    open()/os.listdir() directly need this since shell expansion only
    happens for subprocess.run(shell=True)."""
    if not p:
        return p
    return os.path.expanduser(os.path.expandvars(p))


# Ara's system prompt. Kept at module scope so it's easy to find and
# evolve as her capabilities grow. Key things she needs to know:
#   1. Her environment (whose Mac, home dir, launch dir)
#   2. What tools she has
#   3. How the screen-capture flows work — because they're driven by
#      the user (not by Ara calling a tool), she needs to recognize
#      the incoming `[Context — App: …]` packets as captures rather
#      than treating them as confusing user messages.
SYSTEM_PROMPT = (
    "You are Ara, a warm, friendly, and highly competent macOS sysadmin. "
    "Speak naturally and conversationally. Be helpful and concise.\n\n"

    f"You're running on {USER_NAME}'s Mac. Their home directory is "
    f"{HOME_DIR}. You were launched from {LAUNCH_CWD}. Use absolute "
    "paths in tool calls (not '~'). Always warn before running "
    "potentially destructive commands.\n\n"

    "## Tools\n"
    "You have function tools to run bash commands, read files, list "
    "directories, open macOS apps, run AppleScript, and read text from "
    "the user's screen aloud (read_screen_region_text — see the screen-"
    "capture section below). Use them freely to accomplish what the "
    "user asks. After a tool runs, the system automatically asks you "
    "to follow up — talk about what you found or did, don't just go "
    "silent.\n\n"

    "## Screen captures (important — these arrive automatically)\n"
    "The user has a Capture hotkey and menu item that take a screenshot "
    "and pipe a one-sentence SEMANTIC DESCRIPTION of what they're "
    "looking at directly into this conversation as a user message "
    "prefixed `[Context — App: … | Window: … | source: vision]`. You "
    "never need to tell them to save a file, attach an image, paste "
    "text, or take a screenshot with macOS — the host app handles "
    "capture and feeds it in automatically. When the user says they'll "
    "'show you' or 'share' something, just wait briefly — the capture "
    "is coming. Don't give save-as instructions; don't ask them to "
    "paste.\n\n"

    "## Reading screen content aloud\n"
    "The vision-description capture above is a SUMMARY — it paraphrases "
    "and is not suitable for reading verbatim. When the user asks you "
    "to READ text on their screen ('read this article to me', 'read "
    "this email out loud', 'read me what's on screen', 'read it word "
    "for word', etc.) you MUST call the `read_screen_region_text` "
    "tool — it returns the actual prose, cleaned of code / URLs / UI "
    "chrome / timestamps, ready to be spoken verbatim. Do this even if "
    "you already have a vision description of the screen, because the "
    "description is not the literal text. After the tool returns, read "
    "what it gave you naturally and verbatim — do not paraphrase, "
    "summarize, or editorialize unless the user asks you to."
)


def _format_tool_call(tool_name: str, args: dict) -> str:
    """Terminal-style representation of a tool call for the HUD.
    bash commands look like `$ cmd`; other tools as `tool args`."""
    if tool_name == "run_bash":
        return f"$ {args.get('command', '')}"
    if tool_name == "read_file":
        return f"read  {args.get('path', '')}"
    if tool_name == "list_files":
        return f"ls    {args.get('path', '.')}"
    if tool_name == "open_app":
        return f"open  {args.get('app_name', '')}"
    if tool_name == "run_applescript":
        snippet = (args.get("script", "") or "").split("\n")[0]
        return f"osascript  {snippet}"
    flat = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"{tool_name}  {flat}"


def _clean_tool_result(result: str) -> str:
    """Strip the verbose 'stdout: / stderr: / exit:' framing from
    run_bash results so the HUD shows just the actual output, the way
    you'd see it in a real terminal."""
    if not result:
        return ""
    # run_bash result format: "stdout: <s>\nstderr: <s>\nexit: <n>"
    if result.startswith("stdout: "):
        lines = result.split("\n")
        stdout_part = ""
        stderr_part = ""
        for line in lines:
            if line.startswith("stdout: "):
                stdout_part = line[len("stdout: "):]
            elif line.startswith("stderr: "):
                stderr_part = line[len("stderr: "):]
            # ignore "exit: N" — exit code rarely interesting interactively
        pieces = []
        if stdout_part:
            pieces.append(stdout_part)
        if stderr_part:
            pieces.append(stderr_part)
        return "\n".join(pieces) if pieces else ""
    return result

# Jitter-buffer pre-roll: don't start playing until ~800ms of audio has
# accumulated at the START of each response. After that, mid-stream
# underruns silence-pad just the missing samples and resume immediately
# — they do NOT re-engage pre-roll, because doing so turns a 40ms WS
# hiccup into a long audible gap.
#
# 800ms is chosen to absorb most observed server-side TTS hiccups and
# typical network jitter. Beyond this, the remaining audio gaps are
# either truly long server pauses, AirPods firmware quirks (verified
# Beats don't have them), or environmental network issues.
PREROLL_MS = 800
PREROLL_SAMPLES = SAMPLE_RATE * PREROLL_MS // 1000

# Hard cap on the playback ring buffer. xAI's TTS server frequently
# bursts audio at 2-3× real-time, so a 50s response can land 30s+ of
# pending audio in the buffer mid-stream — the previous 10s cap was
# dropping that excess and causing audible mid-word audio skips
# (perceived as "random blanks", confirmed via deficit -35s in
# telemetry). 180s is large enough to hold any realistic single
# response in full while still bounding memory (~8.6 MB) against a
# runaway server.
MAX_BUFFERED_SAMPLES = SAMPLE_RATE * 180

# Substrings that suggest a device is Bluetooth audio. macOS forces the
# whole BT connection into low-quality HFP mode the moment any app opens
# the BT device's mic — which is *the* reason AirPods get choppy on Macs
# during voice calls. We avoid that by opening the mic on a non-BT device.
_BT_HINTS = ("airpods", "bluetooth", "buds", "beats", "headset")


def _pick_input_device() -> Optional[int]:
    """Pick a non-Bluetooth mic for the agent, unconditionally.

    Why not respect the system default? Because macOS silently switches
    the default input to AirPods the moment they connect, regardless of
    what you set in Sound preferences. Any BT mic capture then forces the
    connection into low-quality HFP mode and trashes the output. So we
    always look for a built-in / USB / wired mic and pin to that.

    Preference order:
      1. Built-in MacBook mic ("MacBook Pro Microphone", "Built-in Mic", …)
      2. Other non-Bluetooth input (USB, wired headset, etc.)
      3. None — meaning no non-BT mic exists; caller falls through to
         the system default and prints a warning.
    """
    try:
        devices = sd.query_devices()
    except Exception:
        return None

    non_bt = []
    for i, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) < 1:
            continue
        name = (dev.get("name") or "").lower()
        if any(h in name for h in _BT_HINTS):
            continue
        non_bt.append((i, name))

    # Prefer the built-in mic over USB or virtual devices (e.g. Continuity
    # Camera's "me Microphone" is non-BT but routes via the network and is
    # less reliable than the local hardware mic).
    preferred = ("macbook", "built-in", "built in", "internal")
    for i, name in non_bt:
        if any(p in name for p in preferred):
            return i

    return non_bt[0][0] if non_bt else None

# Tools are now defined in ara_agent/tools/definitions.py for sharing with the Brain layer.
# We still import here for backward compatibility during the transition.
from ara_agent.tools.definitions import TOOLS as _SHARED_TOOLS
TOOLS = _SHARED_TOOLS

# Lazy import to avoid circular dependency during the architecture transition
try:
    from ara_agent.brain import Brain
except Exception:
    Brain = None  # type: ignore


class AraAgent:
    def __init__(
        self,
        state_callback: Optional[Callable[[str], None]] = None,
        event_callback: Optional[Callable[[str, str], None]] = None,
        cost_callback: Optional[Callable[[dict], None]] = None,
        brain: Optional["Brain"] = None,
        run_id: Optional[str] = None,
    ):
        self.ws = None
        self.is_running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.mic_queue: Optional[asyncio.Queue] = None
        # Continuous playback ring buffer. A single OutputStream callback
        # drains this every audio tick, so CoreAudio (and Bluetooth) get a
        # steady byte stream instead of start/stop sessions per chunk.
        self._play_buffer: "collections.deque[np.ndarray]" = collections.deque()
        self._play_buffer_pos: int = 0   # position inside the front chunk
        self._buffered_samples: int = 0  # running total for O(1) preroll check
        self._is_streaming: bool = False # past pre-roll? toggles on underrun
        self._play_lock = threading.Lock()
        # Audio-thread → asyncio-thread status pipe. The callback stashes
        # PortAudio's status flags here instead of calling print() directly,
        # which would block the realtime thread and worsen underruns.
        self._audio_status_pending: Optional[str] = None
        # Counter of how many callback frames silence-padded because the
        # buffer was empty mid-stream. Useful for diagnosing whether
        # audio gaps come from buffer starvation (our fault) or from the
        # device side (BT/AirPods hardware quirk).
        self._underrun_frames: int = 0
        # Per-response telemetry — reset on response.created.
        self._resp_stats: dict = {}

        # Session-level token / cost tracking for the dedicated cost HUD panel.
        # These are accumulated from server usage reports (ground truth for
        # what this WS session consumed). Reset on new connection.
        self._session_input_tokens: int = 0
        self._session_output_tokens: int = 0
        self._session_total_tokens: int = 0
        self._last_usage: Optional[dict] = None  # most recent usage blob from server
        self._recent_vision_packet: bool = False

        # Called whenever the agent transitions states (idle/listening/thinking/speaking).
        # Invoked on the asyncio thread, so the listener must marshal back to main if needed.
        self._state_callback = state_callback or (lambda _state: None)
        # Called when something noteworthy happens — tool calls, results,
        # screenshot captures. Args: (kind, text) where kind is one of
        # "call" / "result" / "say" / "info" / "warn". Same threading
        # caveat as state_callback.
        self._event_callback = event_callback or (lambda _k, _t: None)
        # Dedicated cost callback: receives the raw usage dict from the server.
        # The CostHUD subscribes to this so it can do proper session/lifetime math.
        self._cost_callback = cost_callback
        self.brain = brain
        self._run_id = run_id or "n/a"
        # True between the first audio chunk of a response and response.done.
        # Used by the player to avoid flickering to "listening" between audio chunks.
        self._response_active = False
        # True from function_call_arguments.done through to the moment we
        # send response.create for the follow-up. Holds the UI state at
        # "thinking" so it doesn't briefly flip to "listening" between the
        # server's initial response.done and the follow-up audio arriving.
        self._in_tool_flow = False
        # Stall watchdog: when a response begins producing items but no
        # audio/transcript delta arrives within STALL_THRESHOLD_S, we
        # surface a warning and cancel the response. Built because the
        # xAI Realtime endpoint occasionally goes silent mid-response
        # without emitting an error event — see handle_messages for the
        # diagnostic flow.
        self._stall_watchdog: Optional["asyncio.Task"] = None
        self.STALL_THRESHOLD_S = 5.0
        # Set to True when we declare the current response dead (watchdog
        # fire or user barge-in). The server doesn't always honor our
        # response.cancel — late audio/transcript chunks keep arriving,
        # which without this flag would (a) play phantom audio after we
        # told the user we cancelled and (b) re-arm the watchdog into a
        # fire loop. Cleared on the next response.created or response.done.
        self._abandoned_response = False
        # Path to the most recent screenshot still on disk. Set when the
        # vision-capture flow completes; reused by read_screen_region_text
        # so the user doesn't have to drag-select the same region twice
        # in one breath. Cleaned up when a new capture replaces it or
        # on agent close.
        self._last_capture_path: Optional[Path] = None
        # Auto-retry state for *programmatic* response turns we care
        # about: capture-induced turns (vision packet → speak) and
        # tool-follow-up turns (tool result → speak). When the watchdog
        # cancels one of these, we quietly resend response.create up
        # to MAX_RESPONSE_RETRIES additional times before giving up.
        # xAI's voice backend has been flaky on text-injection turns
        # (stalls/idle-timeouts) and retries cover most transient
        # failures without the user noticing.
        #
        # Not armed for user-voice turns — if the user spoke and the
        # response stalls, they're right there and can just speak again,
        # and the user-cancellation flow already handles barge-in.
        # Shape when set: {"attempts": int, "max_attempts": int}
        self._pending_response_retry: Optional[dict] = None
        self.MAX_RESPONSE_RETRIES = 2  # → 3 attempts total

        # xAI service-health tracking. We've stacked four layers of
        # client-side resilience around xAI's Realtime backend; further
        # workarounds have diminishing returns. Instead we just OBSERVE
        # the failure rate and tell the user when the service is
        # degraded enough that retries probably won't save them — they
        # can make their own call to pause or keep trying.
        #
        # _health_events holds timestamps of recent xAI failures
        # (watchdog fires + server error events) within HEALTH_WINDOW_S.
        # When count ≥ HEALTH_DEGRADED_THRESHOLD we flip into degraded
        # state and emit a one-time warning. _health_degraded debounces
        # the warning so we don't re-fire on every event while in the
        # degraded state.
        self._health_events: list[float] = []
        self._health_degraded = False
        self.HEALTH_WINDOW_S = 60.0
        self.HEALTH_DEGRADED_THRESHOLD = 2

    def _set_state(self, state: str) -> None:
        try:
            self._state_callback(state)
        except Exception:
            pass  # never let UI errors kill the audio loop

    def _emit_event(self, kind: str, text: str) -> None:
        """Fire a UI event for surfaces like the Output HUD. Swallows
        all exceptions so a broken UI never crashes the audio loop."""
        try:
            self._event_callback(kind, text)
        except Exception:
            pass

    def _record_xai_failure(self, kind: str) -> None:
        """Note a watchdog stall or server error against the health
        window. If we cross the degraded threshold, surface a clear
        one-time warning so the user knows it's not just bad luck on
        this turn — the service itself is unhappy."""
        now = time.time()
        cutoff = now - self.HEALTH_WINDOW_S
        self._health_events = [t for t in self._health_events if t >= cutoff]
        self._health_events.append(now)
        if (not self._health_degraded
                and len(self._health_events) >= self.HEALTH_DEGRADED_THRESHOLD):
            self._health_degraded = True
            msg = (
                f"xAI voice service appears degraded — "
                f"{len(self._health_events)} failures in the last "
                f"{int(self.HEALTH_WINDOW_S)}s ({kind} most recent). "
                f"Audio may be unreliable; consider pausing for a minute."
            )
            log.warning("health: %s", msg)
            self._emit_event("warn", f"⚠ {msg}")
            # Stdout too — user may have HUD hidden but always sees terminal.
            print(f"⚠️  {msg}")

    def _record_xai_success(self) -> None:
        """A clean response with audio landed. If we'd been degraded
        and the failure window is now empty, declare recovery — gives
        the user a clear signal that things are working again."""
        if not self._health_degraded:
            return
        now = time.time()
        cutoff = now - self.HEALTH_WINDOW_S
        self._health_events = [t for t in self._health_events if t >= cutoff]
        if not self._health_events:
            self._health_degraded = False
            log.info("health: xAI service recovered (window cleared)")
            self._emit_event("info", "✓ xAI service looks healthy again.")
            print("✓ xAI service looks healthy again.")

    def report_turn_start(self, label: str = "Turn") -> None:
        """Notify the cost system that we are starting a turn that is expected
        to consume tokens (vision packet, tool follow-up, etc.). This allows
        the dedicated cost panel to show 'in-flight' activity.
        """
        if self._cost_callback:
            try:
                self._cost_callback({"type": "turn_start", "label": label})
            except Exception:
                log.exception("cost_callback (turn_start) failed")

    def _accumulate_usage(self, usage: dict) -> None:
        """Record token usage reported by the xAI Realtime server for one response.
        This is the ground-truth data for what this session consumed.
        We accumulate across all responses for the lifetime of the connection.
        """
        if not usage:
            return

        input_t = usage.get("input_tokens", 0) or usage.get("input_tokens_details", {}).get("text_tokens", 0) or 0
        output_t = usage.get("output_tokens", 0) or usage.get("output_tokens_details", {}).get("text_tokens", 0) or 0

        # Some voice endpoints report separate audio tokens
        audio_input = usage.get("input_tokens_details", {}).get("audio_tokens", 0) or 0
        audio_output = usage.get("output_tokens_details", {}).get("audio_tokens", 0) or 0

        total = usage.get("total_tokens", input_t + output_t) or 0

        self._session_input_tokens += input_t + audio_input
        self._session_output_tokens += output_t + audio_output
        self._session_total_tokens += total
        self._last_usage = usage

        log.info(
            "usage: +%d in / +%d out (total session: %d in, %d out, %d total)",
            input_t + audio_input,
            output_t + audio_output,
            self._session_input_tokens,
            self._session_output_tokens,
            self._session_total_tokens,
        )

        # Feed the raw usage dict to the dedicated CostHUD (if wired)
        if self._cost_callback:
            try:
                self._cost_callback({"type": "usage", "usage": usage})
            except Exception:
                log.exception("cost_callback failed")

        # Also emit the old string event for anything still listening
        cost_text = (
            f"Session: {self._session_input_tokens} in / {self._session_output_tokens} out "
            f"({self._session_total_tokens} total tokens)"
        )
        self._emit_event("cost", cost_text)

    async def stop(self) -> None:
        """Cleanly shut down: stop loops and close the WS so run() returns."""
        self.is_running = False
        if self.mic_queue is not None:
            self.mic_queue.put_nowait(None)  # wake mic_sender
        with self._play_lock:
            self._play_buffer.clear()
            self._play_buffer_pos = 0
            self._buffered_samples = 0
            self._is_streaming = False
        # Tempfile cleanup — the cached screenshot is in /tmp and will
        # eventually be reclaimed by the OS anyway, but be tidy.
        self._clear_last_capture()
        if self.ws is not None:
            await self.ws.close()

    async def _yield_to_new_turn(self, label: str) -> None:
        """Make room for a new user turn by ending whatever response is
        currently in flight.

        Background: every screenshot path was sending response.create
        unconditionally. When the user fires a capture while a prior
        response is still streaming (or hung waiting for the server),
        the realtime API silently cancels response #1 and starts #2 —
        which we saw correlate with response #2 itself stalling out.
        Explicitly closing the previous turn ourselves gives the server
        a clean slate to react to the new context.

        Reuses cancel_response() (which clears the play buffer too) and
        then re-asserts the "thinking" state we want during the upcoming
        capture flow. The short sleep gives the server a moment to
        register the cancel before our new conversation.item.create
        arrives — empirically reduces the chance of the next response
        going silent."""
        if not self._response_active:
            return
        log.info(
            "%s capture: cancelling in-flight response before new turn",
            label,
        )
        await self.cancel_response()
        await asyncio.sleep(0.05)
        self._set_state("thinking")

    def _bump_progress(self) -> None:
        """Reset the stall watchdog. Called whenever we see proof that
        the server's response is making forward progress (audio chunk,
        transcript delta). Implements a sliding "no progress for N
        seconds" timeout rather than a fixed deadline, so a slow-but-
        steady response isn't killed.

        No-op while a tool call is in flight (we're legitimately waiting
        on our own tool, not on server output — no audio/transcript will
        arrive until execute_tool finishes and we send response.create).

        No-op if the current response has already been abandoned (watchdog
        fired or user cancelled). Late chunks that arrive after our
        response.cancel must not re-arm a new watchdog — otherwise the
        server's misbehavior produces a fire loop on our side."""
        if self._in_tool_flow or self._abandoned_response:
            return
        if self._stall_watchdog is not None and not self._stall_watchdog.done():
            self._stall_watchdog.cancel()
        self._stall_watchdog = asyncio.create_task(self._watch_for_stall())

    def _clear_stall_watchdog(self) -> None:
        """Cancel the watchdog without restarting. Called on response.done
        or error — the response is no longer active so there's nothing
        to watch."""
        if self._stall_watchdog is not None and not self._stall_watchdog.done():
            self._stall_watchdog.cancel()
        self._stall_watchdog = None

    async def _watch_for_stall(self) -> None:
        """If we don't see progress (audio/transcript delta) within
        STALL_THRESHOLD_S, infer the server is hung and bail.

        xAI's Realtime endpoint occasionally accepts a packet, starts a
        response (response.output_item.added + content_part.added), and
        then goes silent indefinitely while pings continue. No `error`
        event is emitted — the connection just stops producing content.
        Without this watchdog the user sees a frozen overlay and has no
        way to tell whether their input was lost, the model is thinking,
        or the session is permanently wedged. With it, we surface the
        inferred state and free the session so the next turn can land."""
        try:
            await asyncio.sleep(self.STALL_THRESHOLD_S)
        except asyncio.CancelledError:
            return  # Progress arrived in time — exactly what we want.
        last_progress = getattr(self, '_last_progress_ts', None)
        time_since = (time.time() - last_progress) if last_progress else None

        log.warning(
            "stall watchdog fired | run_id=%s | threshold=%.1fs | time_since_last_progress=%.1fs | "
            "last_event_context=%s",
            self._run_id,
            self.STALL_THRESHOLD_S,
            time_since or -1,
            "recent_vision" if getattr(self, '_recent_vision_packet', False) else "normal",
        )
        self._emit_event(
            "warn",
            f"Model stalled ({self.STALL_THRESHOLD_S:.0f}s no audio). "
            "Server may be degraded — cancelling.",
        )
        self._record_xai_failure("stall watchdog")
        try:
            if self.ws is not None:
                await self.ws.send(json.dumps({"type": "response.cancel"}))
        except Exception:
            log.exception("stall watchdog: response.cancel send failed")
        # Mark this response dead. Without this flag, any late audio
        # chunk the server dribbles out after our cancel would re-arm
        # the watchdog (via _bump_progress) and we'd fire repeatedly —
        # exactly the pattern we saw in the field: 4 stall warnings for
        # one stalled response, plus phantom audio playing post-cancel.
        # Cleared on the next response.created / response.done.
        self._abandoned_response = True
        self._stall_watchdog = None
        # Drop anything still buffered so the user doesn't hear stale
        # audio start playing seconds after the "cancelling" warning.
        with self._play_lock:
            self._play_buffer.clear()
            self._play_buffer_pos = 0
            self._buffered_samples = 0
            self._is_streaming = False
        self._response_active = False
        self._set_state("listening")
        # If this stall was on a turn we care about retrying (capture-
        # induced or tool follow-up), quietly resend response.create.
        # Most xAI flakiness on text-injection turns is transient.
        if self._pending_response_retry is not None:
            asyncio.create_task(self._auto_retry_response_turn())

    async def _auto_retry_response_turn(self) -> None:
        """Resend response.create on a programmatic turn (capture or
        tool follow-up) that the watchdog just cancelled, up to
        MAX_RESPONSE_RETRIES additional times.

        We don't re-send any conversation.item.create — the server
        already has the user message / function_call_output in its
        history. Just ask for a fresh response on the existing turn.

        Brief backoff first so the server has time to fully tear down
        the cancelled response on its side. Aborts if a real new
        response started in the interim (e.g., user spoke), so user
        input always wins over our retry."""
        pending = self._pending_response_retry
        if pending is None:
            return
        if pending["attempts"] >= pending["max_attempts"]:
            log.warning(
                "auto-retry: exhausted (%d/%d attempts)",
                pending["attempts"], pending["max_attempts"],
            )
            self._emit_event(
                "warn",
                f"Response failed after {pending['attempts']} attempts. "
                "xAI's voice backend looks unhealthy right now — try again "
                "in a moment.",
            )
            self._pending_response_retry = None
            return
        await asyncio.sleep(1.0)
        # Conditions may have changed while we slept.
        if self._pending_response_retry is None:
            return  # Cleared by success or explicit error
        if self._response_active:
            log.info("auto-retry: skipping, a new response is already active")
            return
        pending["attempts"] += 1
        log.warning(
            "auto-retry response (attempt %d/%d)",
            pending["attempts"], pending["max_attempts"],
        )
        self._emit_event(
            "info",
            f"Retrying response (attempt {pending['attempts']}/"
            f"{pending['max_attempts']})…",
        )
        # Allow the next response.created/output_item.added through.
        self._abandoned_response = False
        self._set_state("thinking")
        try:
            if self.ws is not None:
                await self.ws.send(json.dumps({"type": "response.create"}))
        except Exception:
            log.exception("auto-retry: response.create send failed")
            self._pending_response_retry = None

    async def cancel_response(self) -> None:
        """Barge-in: cancel the in-flight server response. Clears the
        playback buffer too, so the user hears Ara stop immediately
        rather than draining whatever's already queued."""
        if self.ws is None:
            return
        if not self._response_active:
            return
        try:
            await self.ws.send(json.dumps({"type": "response.cancel"}))
        except Exception as e:
            print(f"⚠️  Cancel send failed: {type(e).__name__}: {e}")
        self._clear_stall_watchdog()
        # Same reason as the watchdog path: drop any late server output
        # for this response on the floor instead of letting it leak into
        # the next turn's UI/audio.
        self._abandoned_response = True
        with self._play_lock:
            self._play_buffer.clear()
            self._play_buffer_pos = 0
            self._buffered_samples = 0
            self._is_streaming = False
        self._response_active = False
        self._set_state("listening")

    def _clear_last_capture(self) -> None:
        """Drop the cached screenshot file (if any). Called when a new
        capture replaces it or when the agent shuts down. Failure to
        delete is non-fatal — it's a tempfile."""
        if self._last_capture_path is None:
            return
        try:
            self._last_capture_path.unlink(missing_ok=True)
        except Exception:
            log.exception(
                "failed to clean up previous capture at %s",
                self._last_capture_path,
            )
        self._last_capture_path = None

    async def _send_capture_turn(self, framed: str) -> None:
        """Send a user input_text item + response.create as one capture
        turn. Extracted so the watchdog's auto-retry can reuse the
        response.create half without duplicating logic."""
        if self.ws is None:
            return
        await self.ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": framed}],
            },
        }))
        await self.ws.send(json.dumps({"type": "response.create"}))

    async def request_screenshot_description(self) -> None:
        """Hotkey / menu capture path. Take an interactive screenshot,
        send the image to xAI grok-4.3 for a one-sentence semantic
        description, inject that description into the realtime
        conversation, and cache the screenshot file so the read tool
        can OCR the same image without forcing the user to drag-select
        again.

        Arms the auto-retry counter so if the xAI backend stalls on the
        response (which it intermittently does on text-injection turns),
        the watchdog can quietly resend response.create up to N times
        before giving up."""
        from ara_agent.context import build_capture_packet
        from ara_agent.screenshot import (
            ScreenCaptureBusyError,
            capture_and_describe,
        )

        if self.ws is None:
            return

        self._set_state("thinking")
        triggered_response = False
        try:
            self._emit_event("info", "Describing image…")
            self.report_turn_start("Vision description")
            try:
                result = await capture_and_describe(XAI_API_KEY)
            except ScreenCaptureBusyError:
                # macOS rejected our `screencapture -i` because another
                # interactive capture is already running system-wide.
                # Usually leftover state from a previous agent process.
                log.warning(
                    "capture aborted: another interactive screencapture "
                    "is already in progress"
                )
                self._emit_event(
                    "warn",
                    "Another screen capture is already in progress — "
                    "dismiss it (ESC) and try again.",
                )
                return
            if result is None:
                self._emit_event("info", "Capture cancelled")
                return
            description, screenshot_path, vision_usage = result if len(result) == 3 else (*result, {})
            # Replace any previous cached screenshot with this fresh one
            # so read_screen_region_text reuses what the user just selected.
            self._clear_last_capture()
            self._last_capture_path = screenshot_path
            self._emit_event(
                "info", f"Image  {description[:120]}",
            )

            # Feed the actual usage from the vision description call (this is real spend)
            if vision_usage:
                if self._cost_callback:
                    self._cost_callback({"type": "usage", "usage": vision_usage})

            # === Hybrid model path (preferred for complex work) ===
            brain = self.brain
            if brain is None:
                # Create a lightweight Brain on the fly for this screenshot.
                # In a fuller implementation the OverlayController would own a persistent Brain.
                try:
                    brain = Brain(
                        api_key=XAI_API_KEY,
                        tools=TOOLS,
                        system_prompt=(
                            "You are Ara, a warm and competent macOS sysadmin. "
                            "The user just showed you a screenshot. Respond naturally "
                            "and helpfully about what you see. Be concise."
                        ),
                    )
                except Exception:
                    brain = None

            if brain is not None:
                log.info("[hybrid] Using Brain for screenshot response (run_id=%s)", self._run_id)
                context = build_capture_packet("vision", description)
                spoken = brain.respond_to_screenshot(
                    description=description,
                    app_context=context,
                )
                log.info("[hybrid] Brain produced response (%d chars)", len(spoken) if spoken else 0)
                # Speak via the Realtime connection
                if self.ws is not None:
                    await self.ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "text", "text": spoken}]
                        }
                    }))
                    await self.ws.send(json.dumps({"type": "response.create"}))
                return   # Skip the old injection path when we have a Brain

            # === Legacy / pure Realtime path (being phased out for complex flows) ===
            # Same chokepoint — probes app/window so the description
            # doesn't float context-free.
            framed = build_capture_packet("vision", description)
            await self._yield_to_new_turn("vision")
            log.info("sending vision packet to model: %d chars", len(framed))
            self.report_turn_start("Vision injection")
            self._recent_vision_packet = True
            await self._send_capture_turn(framed)
            log.info("vision packet sent, awaiting response")
            # Arm auto-retry. The watchdog will resend response.create
            # if the server stalls before producing audio.
            self._pending_response_retry = {
                "attempts": 1,
                "max_attempts": self.MAX_RESPONSE_RETRIES + 1,
            }
            triggered_response = True
        except Exception as e:
            print(f"⚠️  Image-describe flow failed: {type(e).__name__}: {e}")
        finally:
            if not triggered_response and not self._response_active:
                self._set_state("listening")

    async def connect(self):
        headers = {"Authorization": f"Bearer {XAI_API_KEY}"}
        
        # Connect without specifying model (let xAI use default for Voice)
        self.ws = await websockets.connect(ENDPOINT, additional_headers=headers)

        await self.ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "voice": VOICE,
                "instructions": SYSTEM_PROMPT,
                "turn_detection": {"type": "server_vad"},
                "tools": TOOLS
            }
        }))

        log.info("connected to Realtime | run_id=%s", self._run_id)
        print("✅ Connected to Ara (xAI Voice Agent)")
        self._set_state("listening")

    async def mic_sender(self):
        """Pull mic chunks off the thread-safe queue and forward to the WS."""
        assert self.mic_queue is not None
        while self.is_running:
            chunk = await self.mic_queue.get()
            if chunk is None:
                return
            b64_audio = base64.b64encode(chunk.tobytes()).decode()
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": b64_audio
            }))

    async def player(self):
        """Open one continuous OutputStream and let its audio thread drain
        the playback ring buffer. This eliminates per-chunk CoreAudio
        teardown/setup, which was the dominant source of choppiness on
        Bluetooth — a single steady byte stream is what BT audio wants.

        The asyncio side just polls the buffer to drive the speaking ↔
        listening state transitions.
        """

        def on_audio(outdata, frames, time, status):
            # Runs on CoreAudio's realtime thread. NEVER call print() or any
            # blocking I/O here — that would slow the audio thread and cascade
            # underruns. Just stash flags for the watcher to log later.
            if status:
                self._audio_status_pending = str(status)

            with self._play_lock:
                # Case 1: nothing buffered. Silence-pad and return.
                # Critical: don't touch _is_streaming here — if we set it
                # True during idle, (a) idle frames falsely count as
                # underruns (inflated starvation telemetry), and (b) the
                # next response bypasses pre-roll entirely.
                if self._buffered_samples == 0:
                    outdata[:, 0] = 0
                    if self._is_streaming:
                        # Genuinely mid-stream underrun — count it.
                        self._underrun_frames += frames
                    return

                # Case 2: have audio. If we haven't started streaming
                # this response yet, gate on pre-roll cushion.
                if not self._is_streaming:
                    if (self._buffered_samples < PREROLL_SAMPLES
                            and self._response_active):
                        outdata[:, 0] = 0
                        return
                    # Pre-roll satisfied (or response already complete and
                    # we're just draining the tail) — start streaming.
                    self._is_streaming = True

                # Drain buffer into outdata.
                idx = 0
                while idx < frames and self._play_buffer:
                    chunk = self._play_buffer[0]
                    remaining = len(chunk) - self._play_buffer_pos
                    take = min(remaining, frames - idx)
                    outdata[idx:idx + take, 0] = chunk[
                        self._play_buffer_pos:self._play_buffer_pos + take
                    ]
                    idx += take
                    self._play_buffer_pos += take
                    self._buffered_samples -= take
                    if self._play_buffer_pos >= len(chunk):
                        self._play_buffer.popleft()
                        self._play_buffer_pos = 0

                if idx < frames:
                    # Mid-stream underrun: silence-pad the missing samples
                    # and resume on the next callback as soon as data is
                    # available. Do NOT reset _is_streaming — that would
                    # re-engage pre-roll and extend a 40ms hiccup into a
                    # long gap.
                    outdata[idx:, 0] = 0
                    self._underrun_frames += (frames - idx)

        # Grace period for an empty buffer before we force the UI back
        # to "listening" in case response.done never fires. Bumped from
        # 2s → 5s because real responses can have mid-stream server
        # pauses of 2-4 seconds (verified via per-response stats); 2s
        # was triggering false-positive transitions during normal flow.
        EMPTY_GRACE_SECONDS = 5.0

        with sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16,
            callback=on_audio,
            # ~43 ms at 24 kHz — large enough for BT to stay fed, small
            # enough that state transitions feel snappy.
            blocksize=1024,
            latency="high",  # let CoreAudio buffer generously for BT
        ):
            currently_playing = False
            empty_since: Optional[float] = None
            loop = asyncio.get_running_loop()

            while self.is_running:
                await asyncio.sleep(0.05)

                # Drain any audio-thread status flags (safe to log from here).
                if self._audio_status_pending:
                    print(f"⚠️  Audio: {self._audio_status_pending}")
                    self._audio_status_pending = None

                with self._play_lock:
                    has_audio = bool(self._play_buffer)

                if has_audio:
                    empty_since = None
                    if not currently_playing:
                        self._set_state("speaking")
                        currently_playing = True
                elif currently_playing:
                    # Buffer empty mid- or post-response: transition back to
                    # "listening" when either response.done fired (clean) or
                    # the grace period elapsed (recovery, in case the server
                    # never sent response.done).
                    if empty_since is None:
                        empty_since = loop.time()
                    elapsed = loop.time() - empty_since
                    # _in_tool_flow holds state at "thinking" through the
                    # tool execution + follow-up trigger. Don't transition
                    # to listening while that's set, regardless of how
                    # long the buffer's been empty.
                    response_truly_done = (
                        not self._response_active and not self._in_tool_flow
                    )
                    grace_fired = (
                        elapsed > EMPTY_GRACE_SECONDS
                        and not self._in_tool_flow
                    )

                    if response_truly_done or grace_fired:
                        # Report whether this response had buffer-starvation
                        # gaps. >0ms means we ran out of data mid-response;
                        # 0ms means cuts (if any) were downstream of us.
                        if self._underrun_frames > 0:
                            ms = self._underrun_frames * 1000 / SAMPLE_RATE
                            print(f"⏱️  buffer-starvation this response: {ms:.0f} ms total")
                            log.info("buffer_starvation: %.0f ms (run_id=%s)", ms, self._run_id)
                        self._underrun_frames = 0
                        self._set_state("listening")
                        currently_playing = False
                        empty_since = None
                        # Critical: only reset preroll state if the response is
                        # genuinely over (response.done fired). If we transition
                        # via grace timeout but the server is still streaming,
                        # the next chunk should resume playback IMMEDIATELY,
                        # not wait another 800ms for pre-roll — that would
                        # compound a 5s server gap into a 5.8s perceived gap.
                        if response_truly_done:
                            self._is_streaming = False
                        else:
                            # Force-clear stuck flag so next chunk's arrival
                            # cleanly enters a new response cycle.
                            self._response_active = False
                            print("⚠️  Grace timeout fired with response still "
                                  "active — server may have dropped response.done.")

    async def handle_messages(self):
        loop = asyncio.get_running_loop()
        async for message in self.ws:
            data = json.loads(message)
            msg_type = data["type"]

            if msg_type == "error":
                # Realtime API surfaces server-side problems (oversized
                # input, bad item shape, rate limits, model errors) as
                # an "error" event. Without this branch they were silently
                # dropped by the if/elif chain — the agent would just go
                # quiet after sending a packet. Log loudly and let the
                # user see it in the HUD so we can debug.
                err = data.get("error", data)
                err_msg = (
                    err.get("message") if isinstance(err, dict) else str(err)
                )
                log.error("server error event: %s", err)
                self._emit_event("warn", f"Server error: {err_msg}")
                self._record_xai_failure("server error")
                self._clear_stall_watchdog()
                # Server has admitted the response failed — ignore anything
                # else it dribbles out for the same turn.
                self._abandoned_response = True
                self._response_active = False
                # An explicit server error is "the request itself is bad"
                # territory (oversized input, rate limit, content rejection),
                # not a transient stall. Retrying the same packet will hit
                # the same wall. Drop pending retry so we don't loop.
                if self._pending_response_retry is not None:
                    log.info(
                        "clearing pending retry on server error "
                        "(retry would hit the same failure)"
                    )
                    self._pending_response_retry = None
                self._set_state("listening")
                continue

            if msg_type == "input_audio_buffer.speech_started":
                # User is talking → any pending auto-retry from a prior
                # programmatic turn is no longer wanted. User input
                # always wins over our retry attempts.
                if self._pending_response_retry is not None:
                    log.info(
                        "user started speaking — abandoning pending "
                        "response retry"
                    )
                    self._pending_response_retry = None
                continue

            if msg_type == "response.created":
                log.info("response.created — server starting new response")
                # New turn begins → previous abandonment (if any) is now
                # irrelevant. Clearing here means stale chunks from the
                # PREVIOUS response would still be dropped (since they'd
                # have to arrive before response.created of the next one),
                # but the new response is processed normally.
                self._abandoned_response = False
                # Arm the stall watchdog NOW, not just at output_item.added.
                # When xAI's backend is degraded it sometimes acknowledges
                # response.create with response.created but never commits
                # to producing an output_item — we saw a retry get stuck
                # in exactly that pre-content limbo for 77 seconds before
                # the server's own idle-timeout fired. Healthy traffic
                # gets from response.created to output_item.added in
                # under 2s; the 5s watchdog gives plenty of headroom.
                # output_item.added / audio.delta / transcript.delta all
                # re-bump the watchdog, so this just catches the gap
                # before the first content event.
                self._bump_progress()
                # Reset per-response telemetry. start = when server signaled
                # response begins; first_audio fills in when first chunk arrives.
                self._resp_stats = {
                    "start": loop.time(),
                    "first_audio": None,
                    "last_chunk": None,
                    "count": 0,
                    "samples": 0,
                    "max_gap_ms": 0.0,
                    "gaps_over_500ms": 0,
                    "had_tool_call": False,
                }
                continue

            if msg_type == "response.output_item.added":
                if self._abandoned_response:
                    log.debug("ignoring output_item.added (response abandoned)")
                    continue
                # Server has committed to producing a response item. Start
                # the stall watchdog here, BEFORE the first audio/transcript
                # arrives — that's the gap where the server has historically
                # hung silently. The watchdog resets on every delta below.
                log.debug("response.output_item.added — arming stall watchdog")
                self._bump_progress()
                continue

            if msg_type == "response.output_audio_transcript.delta":
                if self._abandoned_response:
                    continue
                # Transcript deltas are a valid "progress" signal even when
                # audio chunks are slow to arrive — keep the watchdog alive.
                self._bump_progress()
                continue

            if msg_type == "response.output_audio.delta":
                if self._abandoned_response:
                    # Server is still dribbling audio for a response we
                    # already declared dead. Drop it — don't buffer, don't
                    # bump the watchdog, don't update telemetry.
                    continue
                # xAI/OpenAI realtime variants — accept either field name
                audio_b64 = data.get("delta") or data.get("audio")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    audio = np.frombuffer(audio_bytes, dtype=np.int16)
                    self._response_active = True
                    # Real audio is the strongest progress signal.
                    self._bump_progress()
                    # Audio arrived → the programmatic turn (if there
                    # was one with a pending retry) succeeded. Clear the
                    # retry state so we don't accidentally fire another
                    # retry if a later mid-stream stall happens.
                    if self._pending_response_retry is not None:
                        log.info(
                            "response produced audio after %d attempt(s)"
                            " — clearing retry state",
                            self._pending_response_retry["attempts"],
                        )
                        self._pending_response_retry = None
                    # Audio = service is alive. If we'd previously been
                    # in the degraded state and the window has emptied,
                    # this is recovery.
                    self._record_xai_success()

                    # Telemetry: time of first chunk, inter-chunk gaps.
                    now = loop.time()
                    stats = self._resp_stats
                    if stats:
                        if stats["first_audio"] is None:
                            stats["first_audio"] = now
                            ttfb_ms = (now - stats["start"]) * 1000
                            print(f"📊 First audio after {ttfb_ms:.0f}ms")
                            log.info("first_audio: %dms (run_id=%s)", int(ttfb_ms), self._run_id)
                        elif stats["last_chunk"] is not None:
                            gap_ms = (now - stats["last_chunk"]) * 1000
                            if gap_ms > stats["max_gap_ms"]:
                                stats["max_gap_ms"] = gap_ms
                            if gap_ms > 500:
                                stats["gaps_over_500ms"] += 1
                        stats["last_chunk"] = now
                        stats["count"] += 1
                        stats["samples"] += len(audio)

                    with self._play_lock:
                        # Cap buffer size — drop oldest if pathologically large.
                        while (self._buffered_samples + len(audio)
                                > MAX_BUFFERED_SAMPLES and self._play_buffer):
                            old = self._play_buffer.popleft()
                            self._buffered_samples -= (len(old) - self._play_buffer_pos)
                            self._play_buffer_pos = 0
                        self._play_buffer.append(audio)
                        self._buffered_samples += len(audio)
                    # State="speaking" is set by the player's watcher loop
                    # once the audio actually starts feeding the OutputStream.

            elif msg_type == "response.function_call_arguments.done":
                tool_name = data["name"]
                args = json.loads(data["arguments"])
                log.info("tool call: %s args=%r", tool_name, args)
                self._emit_event("call", _format_tool_call(tool_name, args))
                if self._resp_stats:
                    self._resp_stats["had_tool_call"] = True
                self._set_state("thinking")
                # Hold state at "thinking" through the whole tool dance —
                # see __init__ comment on _in_tool_flow for why.
                self._in_tool_flow = True
                # Disarm the stall watchdog: tool-call output items don't
                # produce audio/transcript deltas, so the watchdog would
                # otherwise fire ~5s later with a "Model stalled" false
                # alarm even though the tool is happily executing. A fresh
                # watchdog will be armed when the post-tool response.create
                # produces its own output_item.added.
                self._clear_stall_watchdog()

                try:
                    try:
                        result = await self.execute_tool(tool_name, args)
                    except Exception as e:
                        log.exception("tool %r raised", tool_name)
                        result = f"Error: {type(e).__name__}: {e}"
                    log.info("tool result (%s): %d chars", tool_name, len(result))
                    # Full result text goes to the HUD; the HUD truncates
                    # for display but stores everything in its scrollable
                    # log.
                    self._emit_event("result", _clean_tool_result(result))

                    await self.ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {
                            "type": "function_call_output",
                            "call_id": data["call_id"],
                            "output": result
                        }
                    }))
                    # Tool output alone doesn't generate a new turn — the
                    # realtime API requires an explicit response.create
                    # to make the model speak about what the tool
                    # returned. Without this, Ara goes silent after the
                    # command runs.
                    self.report_turn_start(f"Tool follow-up ({tool_name})")
                    await self.ws.send(json.dumps({
                        "type": "response.create",
                    }))
                    log.info(
                        "submitted tool output + response.create for %s",
                        tool_name,
                    )
                    # Mark active so the player's watcher doesn't briefly
                    # transition to "listening" before the follow-up's
                    # first audio chunk lands.
                    self._response_active = True
                    # Arm auto-retry for this tool-follow-up turn too.
                    # xAI's text-input pipeline stalls on tool follow-ups
                    # the same way it stalls on capture turns (saw a 30s
                    # idle-timeout on the post-read response in the
                    # field). Cleared on first audio = success.
                    self._pending_response_retry = {
                        "attempts": 1,
                        "max_attempts": self.MAX_RESPONSE_RETRIES + 1,
                    }
                finally:
                    # Always clear the tool-flow flag so response.done for
                    # the follow-up actually transitions us to listening.
                    self._in_tool_flow = False

            elif msg_type == "response.done":
                # If we're mid tool-flow, the server's response.done is
                # only marking the END of the pre-tool half of the turn.
                # Don't flip _response_active off yet — the follow-up
                # response will arrive after our response.create, and we
                # want the watcher to stay in "thinking" through the gap.
                if not self._in_tool_flow:
                    self._response_active = False
                # Response is officially over (success or quietly aborted),
                # so disarm the stall watchdog — nothing left to watch.
                self._clear_stall_watchdog()
                # The server confirming response.done means whatever we
                # were abandoning is fully closed. Safe to clear.
                self._abandoned_response = False
                log.debug(
                    "response.done (in_tool_flow=%s)", self._in_tool_flow,
                )
                stats = self._resp_stats
                if stats and stats.get("first_audio"):
                    wall_ms = (loop.time() - stats["start"]) * 1000
                    audio_ms = stats["samples"] * 1000 / SAMPLE_RATE
                    deficit_ms = wall_ms - audio_ms
                    tool_tag = " [tool call]" if stats["had_tool_call"] else ""
                    print(
                        f"📊 Response: {wall_ms:.0f}ms wall, {audio_ms:.0f}ms audio, "
                        f"deficit {deficit_ms:+.0f}ms, {stats['count']} chunks, "
                        f"max gap {stats['max_gap_ms']:.0f}ms, "
                        f"{stats['gaps_over_500ms']} gaps>500ms{tool_tag}"
                    )
                    log.info(
                        "response_stats: wall=%.0fms audio=%.0fms deficit=%+.0fms chunks=%d "
                        "max_gap=%.0fms gaps>500ms=%d%s (run_id=%s)",
                        wall_ms, audio_ms, deficit_ms, stats['count'],
                        stats['max_gap_ms'], stats['gaps_over_500ms'], tool_tag, self._run_id
                    )

                # Capture any usage the server attached to this response (the ground truth)
                resp = data.get("response", {})
                if isinstance(resp, dict) and resp.get("usage"):
                    self._accumulate_usage(resp["usage"])

                self._recent_vision_packet = False  # reset after a response completes

            else:
                # Unknown / unhandled event type. Log at DEBUG so we don't
                # spam the file with every audio-delta variant, but enough
                # that we can spot new server events when something breaks.
                log.debug("unhandled message type: %s", msg_type)

                # Temporary discovery logging for cost tracking work.
                # If the Realtime server ever sends usage info, we want to see
                # the exact shape so we can accumulate ground-truth tokens.
                if "usage" in data:
                    log.info("USAGE EVENT RECEIVED: %s", data.get("usage"))
                    self._accumulate_usage(data["usage"])
                elif isinstance(data.get("response"), dict) and "usage" in data["response"]:
                    log.info("USAGE IN RESPONSE: %s", data["response"]["usage"])
                    self._accumulate_usage(data["response"]["usage"])

    async def execute_tool(self, name: str, args: dict) -> str:
        """Dispatch to a tool implementation. Sync tools run in the
        default executor (a background thread) — critical, because
        synchronous subprocess.run inside the asyncio loop would block
        WS message processing for the entire tool duration, starving
        the audio playback buffer and producing audible 10+ second
        silence gaps in Ara's response (verified via underrun counter).
        Async tools (like read_screen_region_text, which already hands
        its blocking work off to executors internally) are awaited
        directly here.
        """
        import subprocess

        # Async tools are dispatched inline before the executor-backed
        # sync handlers below.
        if name == "read_screen_region_text":
            from ara_agent.screenshot import (
                NO_PROSE_SENTINEL,
                ScreenCaptureBusyError,
                capture_and_clean_for_reading,
            )
            # Prefer the screenshot from the user's most recent Capture
            # hotkey/menu trigger. If one is on disk we reuse it (no
            # second drag-select); otherwise the tool prompts for a
            # fresh region.
            cached = (
                self._last_capture_path
                if self._last_capture_path is not None
                and self._last_capture_path.exists()
                else None
            )
            if cached is not None:
                self._emit_event(
                    "info", "Reading the screenshot you just captured…",
                )
                log.info(
                    "read_screen_region_text: reusing cached capture %s",
                    cached,
                )
            else:
                self._emit_event(
                    "info", "Select the text you want me to read…",
                )
            try:
                result = await capture_and_clean_for_reading(
                    XAI_API_KEY, existing_path=cached,
                )
            except ScreenCaptureBusyError:
                # Gives the model concrete language so it can tell the
                # user exactly why we couldn't read the screen — not
                # the generic "capture failed" mumble.
                log.warning(
                    "read_screen_region_text: another screencapture "
                    "is already in progress"
                )
                self._emit_event(
                    "warn",
                    "Another screen capture is already in progress.",
                )
                return (
                    "I tried to capture the screen but another interactive "
                    "screen capture is already in progress on this Mac. "
                    "Ask the user to complete or cancel the open capture "
                    "(it's usually a system selection prompt waiting for "
                    "them), then ask me to try again."
                )
            if result is None:
                return (
                    "Capture was cancelled, no text was found on the "
                    "selected region, or the cleaning step failed."
                )
            if result == NO_PROSE_SENTINEL:
                return (
                    "I looked at the selected region but didn't find any prose "
                    "worth reading aloud — it appears to be code, UI chrome, or "
                    "structured data. Let the user know and offer to describe "
                    "what's on screen instead."
                )
            log.info(
                "read_screen_region_text: returning %d chars of cleaned prose",
                len(result),
            )
            return result

        def _run_bash():
            try:
                r = subprocess.run(args["command"], shell=True,
                                   capture_output=True, text=True, timeout=30)
                return f"stdout: {r.stdout}\nstderr: {r.stderr}\nexit: {r.returncode}"
            except Exception as e:
                return f"Error: {str(e)}"

        def _read_file():
            try:
                path = _normalize_path(args["path"])
                with open(path, "r") as f:
                    return f.read()[:2000]
            except Exception as e:
                return f"Error: {str(e)}"

        def _list_files():
            import os
            path = _normalize_path(args.get("path", "."))
            try:
                return "\n".join(os.listdir(path))
            except Exception as e:
                return f"Error: {str(e)}"

        def _open_app():
            app_name = args["app_name"]
            try:
                # `open -a` resolves by display name; non-zero exit = not found.
                r = subprocess.run(
                    ["open", "-a", app_name],
                    capture_output=True, text=True, timeout=10,
                )
                if r.returncode == 0:
                    return f"Opened {app_name}."
                return f"Could not open {app_name}: {r.stderr.strip() or 'unknown error'}"
            except Exception as e:
                return f"Error opening {app_name}: {str(e)}"

        def _run_applescript():
            script = args["script"]
            try:
                r = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True, text=True, timeout=30,
                )
                if r.returncode == 0:
                    return r.stdout.strip() or "(script ran, no output)"
                return f"AppleScript error: {r.stderr.strip()}"
            except Exception as e:
                return f"Error running AppleScript: {str(e)}"

        handlers = {
            "run_bash": _run_bash,
            "read_file": _read_file,
            "list_files": _list_files,
            "open_app": _open_app,
            "run_applescript": _run_applescript,
        }
        handler = handlers.get(name)
        if handler is None:
            return "Unknown tool"

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, handler)

    async def run(self):
        # Install our asyncio exception handler so anything that throws
        # in a coroutine or callback gets logged with a full traceback
        # instead of just being printed to stderr.
        try:
            from ara_agent.log_setup import install_asyncio_hook
            install_asyncio_hook(asyncio.get_running_loop())
        except Exception:
            log.exception("failed to install asyncio exception hook")

        log.info("AraAgent.run() \u2014 connecting to %s", ENDPOINT)
        try:
            await self.connect()
        except Exception as e:
            log.exception("WebSocket connect failed")
            print(f"\n\u274c Connection failed: {e}")
            print("Common fixes:")
            print("  - Make sure your API key has Voice access in console.x.ai")
            print("  - Try storing the key again with the security command")
            return

        self.loop = asyncio.get_running_loop()
        # ~10 s of audio buffer at default sounddevice blocksize/24 kHz.
        self.mic_queue = asyncio.Queue(maxsize=500)
        # Reset playback buffer in case this AraAgent is being reused.
        with self._play_lock:
            self._play_buffer.clear()
            self._play_buffer_pos = 0
            self._buffered_samples = 0
            self._is_streaming = False
        self._audio_status_pending = None
        self.is_running = True

        def _enqueue_mic(chunk):
            # Runs on the asyncio loop thread; QueueFull is raised here, not at
            # call_soon_threadsafe, so this is where we have to catch it.
            try:
                self.mic_queue.put_nowait(chunk)
            except asyncio.QueueFull:
                pass  # drop chunk under backpressure

        def audio_callback(indata, frames, time, status):
            # Runs in sounddevice's CFFI thread — no asyncio loop here.
            if status:
                print(status)
            self.loop.call_soon_threadsafe(_enqueue_mic, indata[:, 0].copy())

        input_device = _pick_input_device()
        if input_device is not None:
            try:
                mic_name = sd.query_devices(input_device)["name"]
                print(f"\U0001f3a7 Mic: {mic_name}  (pinned non-Bluetooth)")
            except Exception:
                pass
        else:
            print("⚠️  No non-Bluetooth mic found — using system default. "
                  "Bluetooth output may degrade to HFP.")

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16,
            device=input_device,
            callback=audio_callback,
        ):
            print("🎤 Listening... (Ctrl+C to stop)")
            try:
                await asyncio.gather(
                    self.handle_messages(),
                    self.mic_sender(),
                    self.player(),
                )
            except websockets.exceptions.ConnectionClosedOK:
                # Normal close — user clicked Stop, server hung up cleanly, etc.
                # Not a crash; just exit run() quietly.
                pass
            except websockets.exceptions.ConnectionClosedError as e:
                print(f"⚠️  Connection dropped: {e.code} {e.reason}")
            finally:
                self.is_running = False


async def main():
    agent = AraAgent()
    try:
        await agent.run()
    except KeyboardInterrupt:
        print("\n\U0001f44b Goodbye!")
    finally:
        if agent.ws:
            await agent.ws.close()


if __name__ == "__main__":
    asyncio.run(main())
