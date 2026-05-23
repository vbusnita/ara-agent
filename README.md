# ara

**Voice-first Grok agent for macOS that lives on your screen.**

Ara is a floating "blackhole" overlay that listens, watches your screen when you ask, speaks with a warm voice, and can actually do things on your Mac.

The primary interface is a small, always-available, draggable blackhole icon. Click it for actions, use the global hotkey (`Cmd+Shift+A`) to capture context, and talk naturally. There's also an optional inspection HUD for seeing what she's thinking and doing.

## Quick Start

```bash
git clone https://github.com/vbusnita/ara-agent.git
cd ara-agent
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # add your XAI_API_KEY
```

Then launch the overlay (recommended):

```bash
ara start --overlay
# or shorthand:
ara start -o
```

The blackhole will appear. Click it to start listening, or use the menu to show the Inspection HUD.

**Global hotkey:** `Cmd + Shift + A` triggers a screen capture while listening.

## Running Modes

| Command                    | Description                              | Recommended |
|---------------------------|------------------------------------------|-------------|
| `ara start --overlay`     | Floating draggable blackhole + HUD       | **Yes**     |
| `ara start --menu`        | macOS menu bar app                       | -           |
| `ara start`               | Plain terminal / CLI mode                | -           |

We're currently focused on the overlay experience as the main way to use Ara.

## What Ara Can Do

- Real-time voice conversation (barge-in supported)
- Screen capture + semantic understanding (vision)
- Read text on screen aloud (Apple OCR + cleaning)
- Tool calling: run bash, read files, list directories, open apps, run AppleScript
- Confirmation flows for potentially destructive actions
- Persistent floating UI that remembers position across restarts (multi-monitor friendly)

## Development

The source of truth for the project is **[AGENT.md](AGENT.md)** — it contains the current state, vision, known gaps, and priorities.

If you're an AI agent working on this codebase, you should also read **CLAUDE.md** (it contains the operating manual and collaboration rules for this project).

We're currently on the `supervisor-mode` branch working toward making Ara a visible, reliable "grunt worker" you can supervise while it does network mapping, security hardening, and other real operational work.

## Philosophy

- The agent should feel like a calm, competent presence on your Mac — not another chat window.
- Visibility and inspectability matter. You should be able to see what the agent is doing.
- Voice is the primary input, but the agent can also see and act.
- Keep the core tight. Maximum intelligence per line of code.
- The user is the supervisor. The agent proposes and executes, but important actions should feel safe and transparent.

---

*Named after Ara, the warm voice from xAI.*