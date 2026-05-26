"""
Brain — the strong reasoning layer for ara-agent (hybrid architecture).

This module is responsible for the "heavy lifting":
- Understanding screenshots + context via the regular Responses API
- Reasoning and planning
- Calling tools
- Deciding what the user should hear

The Realtime voice layer (voice_agent.py) becomes mostly responsible for:
- Low-latency audio input/output
- Barge-in
- Speaking whatever text the Brain tells it to say

This separation makes the agent much more reliable for real work
(screenshots → actual actions on the Mac, network ops, security tasks, etc.).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Optional

import urllib.request

log = logging.getLogger(__name__)

# For now we use the same model family as the vision descriptions.
# Later we can make this configurable or use a stronger model for the Brain.
BRAIN_MODEL = os.getenv("ARA_BRAIN_MODEL", "grok-4.3")
BRAIN_ENDPOINT = "https://api.x.ai/v1/responses"


class Brain:
    """
    High-level reasoning agent that uses the regular xAI Responses API
    (much more reliable than Realtime for complex work).
    """

    def __init__(
        self,
        api_key: str,
        tools: list[dict[str, Any]],
        system_prompt: str,
        on_speak: Optional[Callable[[str], None]] = None,
        on_tool_call: Optional[Callable[[str, dict], Any]] = None,
    ):
        self.api_key = api_key
        self.tools = tools
        self.system_prompt = system_prompt
        self.on_speak = on_speak          # callback to make the voice layer speak
        self.on_tool_call = on_tool_call  # callback to execute tools
        self.conversation: list[dict[str, Any]] = []

    def add_user_message(self, text: str, image_b64: Optional[str] = None) -> None:
        """Add a user message, optionally with an image."""
        content: list[dict[str, Any]] = []
        if image_b64:
            content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{image_b64}",
                "detail": "high",
            })
        content.append({"type": "input_text", "text": text})

        self.conversation.append({
            "role": "user",
            "content": content
        })

    def run_turn(self) -> str:
        """
        Send the current conversation to the strong model and get a response.
        Returns the text the agent should speak (if any).
        """
        messages = [{"role": "system", "content": self.system_prompt}] + self.conversation

        body = json.dumps({
            "model": BRAIN_MODEL,
            "input": messages,
            "tools": self.tools,
            "tool_choice": "auto",
        }).encode()

        req = urllib.request.Request(
            BRAIN_ENDPOINT,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)

        # Very simplified parsing — real implementation needs to handle
        # tool calls, multiple outputs, etc.
        output_text = ""
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        output_text += c.get("text", "")

        if output_text:
            self.conversation.append({
                "role": "assistant",
                "content": [{"type": "output_text", "text": output_text}]
            })

        return output_text

    def respond_to_screenshot(
        self,
        description: str,
        app_context: str = "",
        window_context: str = "",
    ) -> str:
        """
        High-level helper for the common case: user showed a screenshot.
        The Brain reasons about it using the strong model and returns
        text that should be spoken to the user.
        """
        context = ""
        if app_context or window_context:
            context = f"\n[Context — App: {app_context} | Window: {window_context}]"

        prompt = f"The user just showed you a screenshot of their screen.{context}\n\n" \
                 f"Here is a semantic description of what is visible:\n\"{description}\"\n\n" \
                 "Respond naturally and helpfully as if you can see the screen. " \
                 "Be concise unless the user asks for more detail."

        self.add_user_message(prompt)
        spoken_text = self.run_turn()
        return spoken_text or "Got it."

    # TODO: proper tool calling loop, streaming, better error handling, etc.