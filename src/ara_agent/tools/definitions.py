"""
Shared tool definitions for ara-agent.

These are used by both the voice layer (for simple cases) and the
main reasoning Brain (for complex work with screenshots, etc.).
"""

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
        "description": "Read the contents of a file on the local machine.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"}
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
                "path": {"type": "string", "default": "."}
            },
            "required": []
        }
    },
    {
        "type": "function",
        "name": "open_app",
        "description": "Open a macOS application by name.",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {"type": "string"}
            },
            "required": ["app_name"]
        }
    },
    {
        "type": "function",
        "name": "run_applescript",
        "description": "Execute AppleScript on the local Mac.",
        "parameters": {
            "type": "object",
            "properties": {
                "script": {"type": "string"}
            },
            "required": ["script"]
        }
    },
    {
        "type": "function",
        "name": "read_screen_region_text",
        "description": (
            "Read text from the user's screen aloud using OCR. "
            "Use this when the user explicitly asks to read content from the screen."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
]