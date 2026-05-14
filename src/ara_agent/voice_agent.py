"""
ara-agent — Voice-first Grok computer agent
Core real-time voice client using xAI Voice Agent API
"""

import asyncio
import base64
import collections
import json
import os
import threading
from typing import Callable, Optional

import getpass

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

# Jitter-buffer pre-roll: don't start playing until ~500ms of audio has
# accumulated at the START of each response. After that, mid-stream
# underruns silence-pad just the missing samples and resume immediately
# — they do NOT re-engage pre-roll, because doing so turns a 40ms WS
# hiccup into a 540ms audible gap (the actual cause of the residual
# choppiness through v1).
PREROLL_MS = 500
PREROLL_SAMPLES = SAMPLE_RATE * PREROLL_MS // 1000

# Hard cap on the playback ring buffer so a long WS burst (e.g. during
# reconnect) can't grow it unboundedly. 10 s is far more than any normal
# response.
MAX_BUFFERED_SAMPLES = SAMPLE_RATE * 10

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
    def __init__(self, state_callback: Optional[Callable[[str], None]] = None):
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
        # Called whenever the agent transitions states (idle/listening/thinking/speaking).
        # Invoked on the asyncio thread, so the listener must marshal back to main if needed.
        self._state_callback = state_callback or (lambda _state: None)
        # True between the first audio chunk of a response and response.done.
        # Used by the player to avoid flickering to "listening" between audio chunks.
        self._response_active = False

    def _set_state(self, state: str) -> None:
        try:
            self._state_callback(state)
        except Exception:
            pass  # never let UI errors kill the audio loop

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
        """User-triggered (e.g. from the overlay menu): capture a window
        screenshot, describe it via xAI vision, and inject the description
        as a user message in the realtime conversation so Ara can talk
        about what's on screen.

        The realtime voice API can't ingest images directly, so we OCR-like
        the screenshot through grok-2-vision and feed the text result.
        """
        from ara_agent.screenshot import capture_and_describe

        if self.ws is None:
            return

        self._set_state("thinking")
        triggered_response = False
        try:
            description = await capture_and_describe(XAI_API_KEY)
            if not description:
                return
            print(f"\U0001f4f8 Screenshot: {description}")
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
            print(f"⚠️  Screenshot flow failed: {type(e).__name__}: {e}")
        finally:
            # If we didn't kick off a response, revert state ourselves. If
            # we did, the player's watcher will transition us out of
            # thinking when audio starts arriving.
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
                # Pre-roll gate: don't start streaming until we have a cushion
                # of audio buffered. Once response.done, drain whatever's
                # left immediately — no point pre-rolling the tail.
                if not self._is_streaming:
                    if (self._buffered_samples < PREROLL_SAMPLES
                            and self._response_active):
                        outdata[:, 0] = 0
                        return
                    self._is_streaming = True

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
                    # 500ms gap. Pre-roll is only for the start of each
                    # response, where bursty initial arrival is normal.
                    outdata[idx:, 0] = 0

        # Grace period for an empty buffer before we assume response.done
        # was missed and force-recover the state. Natural mid-response
        # pauses are encoded as silence inside chunks (buffer stays
        # non-empty), so this only kicks in when the server genuinely
        # stopped streaming.
        EMPTY_GRACE_SECONDS = 2.0

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
                    if not self._response_active or elapsed > EMPTY_GRACE_SECONDS:
                        self._set_state("listening")
                        currently_playing = False
                        empty_since = None
                        self._response_active = False  # un-stick if it was stuck
                        # Reset for the next response so it gets its own
                        # initial pre-roll cushion against bursty start.
                        self._is_streaming = False

    async def handle_messages(self):
        async for message in self.ws:
            data = json.loads(message)

            if data["type"] == "response.output_audio.delta":
                # xAI/OpenAI realtime variants — accept either field name
                audio_b64 = data.get("delta") or data.get("audio")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    audio = np.frombuffer(audio_bytes, dtype=np.int16)
                    self._response_active = True
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

            elif data["type"] == "response.function_call_arguments.done":
                tool_name = data["name"]
                args = json.loads(data["arguments"])
                print(f"\n\U0001f527 Tool call: {tool_name}({args})")
                self._set_state("thinking")

                result = await self.execute_tool(tool_name, args)
                print(f"   Result: {result[:150]}...")

                await self.ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": data["call_id"],
                        "output": result
                    }
                }))

            elif data["type"] == "response.done":
                print("\n\U0001f5e3️  Ara finished speaking")
                # Just flip the flag; the player will set state="listening"
                # once the playback queue drains.
                self._response_active = False

    async def execute_tool(self, name: str, args: dict) -> str:
        if name == "run_bash":
            import subprocess
            try:
                result = subprocess.run(args["command"], shell=True, capture_output=True, text=True, timeout=30)
                return f"stdout: {result.stdout}\nstderr: {result.stderr}\nexit: {result.returncode}"
            except Exception as e:
                return f"Error: {str(e)}"

        elif name == "read_file":
            try:
                with open(args["path"], "r") as f:
                    return f.read()[:2000]
            except Exception as e:
                return f"Error: {str(e)}"

        elif name == "list_files":
            import os
            path = args.get("path", ".")
            try:
                return "\n".join(os.listdir(path))
            except Exception as e:
                return f"Error: {str(e)}"

        elif name == "open_app":
            import subprocess
            app_name = args["app_name"]
            try:
                # `open -a` resolves the app by display name; non-zero exit means not found.
                result = subprocess.run(
                    ["open", "-a", app_name],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    return f"Opened {app_name}."
                return f"Could not open {app_name}: {result.stderr.strip() or 'unknown error'}"
            except Exception as e:
                return f"Error opening {app_name}: {str(e)}"

        elif name == "run_applescript":
            import subprocess
            script = args["script"]
            try:
                # osascript -e accepts the script inline; stdout carries the result,
                # stderr carries compilation/runtime errors.
                result = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    return result.stdout.strip() or "(script ran, no output)"
                return f"AppleScript error: {result.stderr.strip()}"
            except Exception as e:
                return f"Error running AppleScript: {str(e)}"

        return "Unknown tool"

    async def run(self):
        try:
            await self.connect()
        except Exception as e:
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
