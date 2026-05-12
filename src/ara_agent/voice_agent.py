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
import keyring

load_dotenv()

def get_api_key() -> str:
    try:
        key = keyring.get_password("ara-agent", "xai-api-key")
        if key:
            return key
    except Exception:
        pass
    key = os.getenv("XAI_API_KEY")
    if key:
        return key
    raise ValueError(
        "No API key found. Store it with:\n"
        "security add-generic-password -a \"$USER\" -s \"xai-api-key\" -w \"your-key\""
    )


XAI_API_KEY = get_api_key()
VOICE = "ara"
ENDPOINT = "wss://api.x.ai/v1/realtime"

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
    }
]


class AraAgent:
    def __init__(self):
        self.ws = None
        self.is_running = False

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

    async def send_audio(self, audio_data: np.ndarray):
        if self.ws:
            audio_bytes = audio_data.tobytes()
            b64_audio = base64.b64encode(audio_bytes).decode()
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": b64_audio
            }))

    async def handle_messages(self):
        async for message in self.ws:
            data = json.loads(message)

            if data["type"] == "response.output_audio.delta":
                audio_b64 = data["delta"]
                audio_bytes = base64.b64decode(audio_b64)
                audio = np.frombuffer(audio_bytes, dtype=np.int16)
                sd.play(audio, samplerate=24000)
                sd.wait()

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

        def audio_callback(indata, frames, time, status):
            if status:
                print(status)
            asyncio.create_task(self.send_audio(indata[:, 0]))

        with sd.InputStream(samplerate=16000, channels=1, dtype=np.int16, callback=audio_callback):
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
