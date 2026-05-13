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

# Substrings that suggest a device is Bluetooth audio. macOS forces the
# whole BT connection into low-quality HFP mode the moment any app opens
# the BT device's mic — which is *the* reason AirPods get choppy on Macs
# during voice calls. We avoid that by opening the mic on a non-BT device.
_BT_HINTS = ("airpods", "bluetooth", "buds", "beats", "headset")


def _pick_input_device() -> Optional[int]:
    """Return a sounddevice device index for the mic, or None for default.

    If the system's default *output* looks like Bluetooth, find a non-BT
    input device (built-in mic, USB mic, etc.) so the BT link stays in
    high-quality A2DP for output.
    """
    try:
        default_out = sd.default.device[1]
        if default_out is None or default_out < 0:
            return None
        out_name = (sd.query_devices(default_out).get("name") or "").lower()
    except Exception:
        return None

    if not any(h in out_name for h in _BT_HINTS):
        return None  # output isn't BT — system default mic is fine

    try:
        devices = sd.query_devices()
    except Exception:
        return None

    candidates = []
    for i, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) < 1:
            continue
        name = (dev.get("name") or "").lower()
        if any(h in name for h in _BT_HINTS):
            continue
        candidates.append((i, name))

    # Prefer the built-in mic over weirder virtual devices (e.g. Continuity
    # Camera mic is also non-BT but flaky).
    preferred = ("macbook", "built-in", "built in", "internal")
    for i, name in candidates:
        if any(p in name for p in preferred):
            return i

    return candidates[0][0] if candidates else None

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
        self._play_buffer_pos: int = 0  # position inside the front chunk
        self._play_lock = threading.Lock()
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
        if self.ws is not None:
            await self.ws.close()

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
            # Runs on CoreAudio's realtime thread — keep this lean.
            if status:
                print(f"Output status: {status}")
            with self._play_lock:
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
                    if self._play_buffer_pos >= len(chunk):
                        self._play_buffer.popleft()
                        self._play_buffer_pos = 0
                if idx < frames:
                    outdata[idx:, 0] = 0  # silence-pad rather than underrun

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
                        self._play_buffer.append(audio)
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
                print(f"\U0001f3a7 Using mic: {mic_name} "
                      f"(keeps Bluetooth output in A2DP — avoids HFP downgrade)")
            except Exception:
                pass

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
