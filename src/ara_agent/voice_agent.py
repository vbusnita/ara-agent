"""
ara-agent — Voice-first Grok computer agent
Core real-time voice client using xAI Voice Agent API
"""

import asyncio
import json
import os
import base64
from typing import Optional

import websockets
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY not found in environment")

VOICE = "ara"  # Warm, friendly, conversational
MODEL = "grok-voice-latest"
ENDPOINT = "wss://api.x.ai/v1/realtime"

# Basic tools for MVP
TOOLS = [
    {
        "type": "function",
        "name": "run_bash",
        "description": "Execute a bash command on the local machine (use with caution).",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run"},
                "reason": {"type": "string", "description": "Why you're running this command"}
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
            "properties": {
                "path": {"type": "string", "description": "Path to the file"}
            },
            "required": ["path"]
        }
    },
    {
        "type": "function",
        "name": "list_files",
        "description": "List files and directories.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: .)"}
            }
        }
    }
]


class AraAgent:
    def __init__(self):
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.audio_queue = asyncio.Queue()
        self.is_running = False

    async def connect(self):
        """Connect to xAI Voice Agent API"""
        headers = {"Authorization": f"Bearer {XAI_API_KEY}"}
        self.ws = await websockets.connect(
            f"{ENDPOINT}?model={MODEL}",
            additional_headers=headers
        )

        # Initialize session
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

    async def send_audio(self, audio_data: np.ndarray):
        """Send audio chunk to the API"""
        if self.ws:
            # Convert to base64 (assuming 16kHz mono int16)
            audio_bytes = audio_data.tobytes()
            b64_audio = base64.b64encode(audio_bytes).decode()
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": b64_audio
            }))

    async def handle_messages(self):
        """Handle incoming messages from the API"""
        async for message in self.ws:
            data = json.loads(message)

            if data["type"] == "response.output_audio.delta":
                # Play audio chunk
                audio_b64 = data["delta"]
                audio_bytes = base64.b64decode(audio_b64)
                audio = np.frombuffer(audio_bytes, dtype=np.int16)
                sd.play(audio, samplerate=24000)
                sd.wait()

            elif data["type"] == "response.function_call_arguments.done":
                # Tool call received
                tool_name = data["name"]
                args = json.loads(data["arguments"])
                print(f"\n\U0001f527 Tool call: {tool_name}({args})")

                # Execute tool (basic for MVP)
                result = await self.execute_tool(tool_name, args)
                print(f"   Result: {result[:100]}...")

                # Send result back
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
        """Execute tools (MVP implementation)"""
        if name == "run_bash":
            import subprocess
            try:
                result = subprocess.run(
                    args["command"],
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                return f"stdout: {result.stdout}\nstderr: {result.stderr}\nexit: {result.returncode}"
            except Exception as e:
                return f"Error: {str(e)}"

        elif name == "read_file":
            try:
                with open(args["path"], "r") as f:
                    return f.read()[:2000]  # Limit output for MVP
            except Exception as e:
                return f"Error: {str(e)}"

        elif name == "list_files":
            import os
            path = args.get("path", ".")
            try:
                return "\n".join(os.listdir(path))
            except Exception as e:
                return f"Error: {str(e)}"

        return "Unknown tool"

    async def run(self):
        """Main loop"""
        await self.connect()

        # Start audio input (microphone)
        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            asyncio.create_task(self.send_audio(indata[:, 0]))  # Mono

        with sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype=np.int16,
            callback=audio_callback
        ):
            print("🎤 Listening... (Ctrl+C to stop)")
            self.is_running = True
            await self.handle_messages()


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
