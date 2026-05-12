# ara-agent

**Voice-first Grok computer agent for macOS.**

Talk to Ara. She listens, thinks, speaks back in her warm voice, and can take actions on your Mac.

## Current Status (MVP)

- Real-time voice conversation with Ara
- Basic tool calling (`run_bash`, `read_file`, `list_files`)
- Clean, minimal architecture
- Launched with a simple `ara` command

## Quick Start

```bash
git clone https://github.com/vbusnita/ara-agent.git
cd ara-agent
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # add your XAI_API_KEY
ara
```

Then just start talking.

## Philosophy

- Voice is the primary interface (no heavy CLI or GUI)
- Safety first, but not in the way
- Start simple, iterate fast
- Built on xAI's Voice Agent API + Grok

## Roadmap

- [x] Basic voice + tool calling
- [ ] Voice confirmation for sensitive actions
- [ ] Session memory (`AGENT.md` style)
- [ ] Hotkey / menu bar launcher
- [ ] Full self-improvement capabilities

---

*Named after Ara — the warm, friendly voice from xAI.*
