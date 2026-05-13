"""
ara-agent — Voice-first Grok computer agent
Core real-time voice client using xAI Voice Agent API
"""

import asyncio
import json
import os
import base64
from typing import Optional

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
    def __init__(self):
        self.ws = None
        self.is_running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.mic_queue: Optional[asyncio.Queue] = None
        self.playback_queue: Optional[asyncio.Queue] = None

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
        """Play queued response audio without blocking the event loop."""
        assert self.playback_queue is not None
        loop = asyncio.get_running_loop()
        while self.is_running:
            audio = await self.playback_queue.get()
            if audio is None:
                return
            # sd.play is non-blocking, sd.wait blocks — run wait in a thread
            sd.play(audio, samplerate=SAMPLE_RATE)
            await loop.run_in_executor(None, sd.wait)

    async def handle_messages(self):
        async for message in self.ws:
            data = json.loads(message)

            if data["type"] == "response.output_audio.delta":
                # xAI/OpenAI realtime variants — accept either field name
                audio_b64 = data.get("delta") or data.get("audio")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    audio = np.frombuffer(audio_bytes, dtype=np.int16)
                    await self.playback_queue.put(audio)

            elif data["type"] == "response.function_call_arguments.done":
                tool_name = data["name"]
                args = json.loads(data["arguments"])
                print(f"\n\U0001f527 Tool call: {tool_name}({args})")

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
        self.playback_queue = asyncio.Queue()
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

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16,
            callback=audio_callback,
        ):
            print("🎤 Listening... (Ctrl+C to stop)")
            try:
                await asyncio.gather(
                    self.handle_messages(),
                    self.mic_sender(),
                    self.player(),
                )
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
