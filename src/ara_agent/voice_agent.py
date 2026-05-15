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

TOOLS = [
    {
        "type": "function",
        "name": "run_bash",
        "description": "Execute a bash command on the local machine (use with caution).",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "reason": {"type": "string"}
            },
            "required": ["command"]
        }
    },
    {
        "type": "function",
        "name": "read_file",
        "description": "Read the contents of a file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    },
    {
        "type": "function",
        "name": "list_files",
        "description": "List files and directories.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}}
        }
    },
    {
        "type": "function",
        "name": "open_app",
        "description": "Open a macOS application by name (e.g. 'Safari', 'Notes', 'Terminal', 'Music').",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "The name of the application to open, as it appears in /Applications."
                }
            },
            "required": ["app_name"]
        }
    },
    {
        "type": "function",
        "name": "run_applescript",
        "description": (
            "Execute arbitrary AppleScript on the local macOS machine. "
            "Powerful — can control any scriptable app, system settings, files, and UI. "
            "Use for anything beyond simple app launching."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "description": "The AppleScript source to run (will be passed to osascript -e)."
                }
            },
            "required": ["script"]
        }
    }
]


class AraAgent:
    def __init__(
        self,
        state_callback: Optional[Callable[[str], None]] = None,
        event_callback: Optional[Callable[[str, str], None]] = None,
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
        # Called whenever the agent transitions states (idle/listening/thinking/speaking).
        # Invoked on the asyncio thread, so the listener must marshal back to main if needed.
        self._state_callback = state_callback or (lambda _state: None)
        # Called when something noteworthy happens — tool calls, results,
        # screenshot captures. Args: (kind, text) where kind is one of
        # "call" / "result" / "say" / "info" / "warn". Same threading
        # caveat as state_callback.
        self._event_callback = event_callback or (lambda _k, _t: None)
        # True between the first audio chunk of a response and response.done.
        # Used by the player to avoid flickering to "listening" between audio chunks.
        self._response_active = False

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
        if self.ws is not None:
            await self.ws.close()

    async def request_screenshot_context(self) -> None:
        """User-triggered (e.g. from the overlay menu): launch the macOS
        interactive screenshot (region or window — Space toggles), OCR
        the result locally via Apple Vision, and inject the recognized
        text into the realtime conversation as user context.

        Local OCR instead of a vision REST call: ~10× faster end to end,
        no token cost, ~1000× less payload (text bytes vs. base64 image),
        and crucially no asyncio-executor GIL contention that used to
        chop audio mid-response.
        """
        from ara_agent.screenshot import capture_and_extract_text

        if self.ws is None:
            return

        self._set_state("thinking")
        triggered_response = False
        try:
            self._emit_event("info", "Capturing screen text…")
            text = await capture_and_extract_text()
            if not text:
                self._emit_event("info", "Capture cancelled")
                return
            line_count = text.count("\n") + 1
            self._emit_event(
                "info",
                f"OCR  {len(text)} chars / {line_count} lines",
            )
            framed = (
                "[The user just captured text from a region of their screen. "
                "Here is exactly what the screen shows:\n\n"
                f"{text}\n\n"
                "Use this as context for what they're asking about.]"
            )
            await self.ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": framed}],
                },
            }))
            await self.ws.send(json.dumps({"type": "response.create"}))
            triggered_response = True
        except Exception as e:
            print(f"⚠️  Screenshot flow failed: {type(e).__name__}: {e}")
        finally:
            # If we didn't kick off a response, revert state ourselves. If
            # we did, the player's watcher will transition us out of
            # thinking when audio starts arriving.
            if not triggered_response and not self._response_active:
                self._set_state("listening")

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
        with self._play_lock:
            self._play_buffer.clear()
            self._play_buffer_pos = 0
            self._buffered_samples = 0
            self._is_streaming = False
        self._response_active = False
        self._set_state("listening")

    async def request_screenshot_description(self) -> None:
        """User-triggered alternate to text capture: take the same
        interactive screenshot, but send the image to xAI grok-4.3 for a
        2-3 sentence visual description, then inject that description as
        user context.

        Use this when the screen has graphics that matter (charts, UI
        layout, diagrams) and the literal text alone wouldn't capture
        what the user is asking about. Slower than OCR and costs tokens,
        so prefer text mode by default.
        """
        from ara_agent.screenshot import capture_and_describe

        if self.ws is None:
            return

        self._set_state("thinking")
        triggered_response = False
        try:
            self._emit_event("info", "Describing image…")
            description = await capture_and_describe(XAI_API_KEY)
            if not description:
                self._emit_event("info", "Capture cancelled")
                return
            self._emit_event(
                "info", f"Image  {description[:120]}",
            )
            framed = (
                "[The user just shared a screenshot of their screen. "
                f"Here is what is visible in it: {description}]"
            )
            await self.ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": framed}],
                },
            }))
            await self.ws.send(json.dumps({"type": "response.create"}))
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
                "instructions": (
                    "You are Ara, a warm, friendly, and highly competent macOS sysadmin. "
                    "Speak naturally and conversationally. Be helpful and concise. "
                    "You can run bash commands, read files, and explore the system. "
                    f"You're running on {USER_NAME}'s Mac. "
                    f"Their home directory is {HOME_DIR}. "
                    f"You were launched from {LAUNCH_CWD}. "
                    "Use absolute paths in tool calls (not '~'). "
                    "Always warn before running potentially destructive commands."
                ),
                "turn_detection": {"type": "server_vad"},
                "tools": TOOLS
            }
        }))

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
                    response_truly_done = not self._response_active
                    grace_fired = elapsed > EMPTY_GRACE_SECONDS

                    if response_truly_done or grace_fired:
                        # Report whether this response had buffer-starvation
                        # gaps. >0ms means we ran out of data mid-response;
                        # 0ms means cuts (if any) were downstream of us.
                        if self._underrun_frames > 0:
                            ms = self._underrun_frames * 1000 / SAMPLE_RATE
                            print(f"⏱️  buffer-starvation this response: {ms:.0f} ms total")
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

            if msg_type == "response.created":
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

            if msg_type == "response.output_audio.delta":
                # xAI/OpenAI realtime variants — accept either field name
                audio_b64 = data.get("delta") or data.get("audio")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    audio = np.frombuffer(audio_bytes, dtype=np.int16)
                    self._response_active = True

                    # Telemetry: time of first chunk, inter-chunk gaps.
                    now = loop.time()
                    stats = self._resp_stats
                    if stats:
                        if stats["first_audio"] is None:
                            stats["first_audio"] = now
                            ttfb_ms = (now - stats["start"]) * 1000
                            print(f"📊 First audio after {ttfb_ms:.0f}ms")
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

                try:
                    result = await self.execute_tool(tool_name, args)
                except Exception as e:
                    log.exception("tool %r raised", tool_name)
                    result = f"Error: {type(e).__name__}: {e}"
                log.info("tool result (%s): %d chars", tool_name, len(result))
                # Full result text goes to the HUD; the HUD truncates for
                # display but stores everything in its scrollable log.
                self._emit_event("result", _clean_tool_result(result))

                await self.ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": data["call_id"],
                        "output": result
                    }
                }))

            elif msg_type == "response.done":
                # State transitions through the HUD now; no need to print.
                self._response_active = False
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

    async def execute_tool(self, name: str, args: dict) -> str:
        """Dispatch to a tool implementation. All implementations run in
        the default executor (a background thread) — critical, because
        synchronous subprocess.run inside the asyncio loop would block
        WS message processing for the entire tool duration, starving
        the audio playback buffer and producing audible 10+ second
        silence gaps in Ara's response (verified via underrun counter).
        """
        import subprocess

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
