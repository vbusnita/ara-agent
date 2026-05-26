# AGENT.md — ara-agent Project Bible

## Project Overview

**ara-agent** is a voice-first, native macOS agent powered by xAI's Grok models. The goal is to build an agent that feels like a true extension of the user — intelligent, context-aware, and delightful to use.

Instead of another CLI tool or heavy GUI, ara-agent lives as a lightweight, floating "blackhole" overlay on the Mac — a small, elegant, always-present presence — with real-time voice conversation and the ability to actually *do* things on the user's behalf.

**Core Philosophy**: Maximum intelligence per line of code, with a strong emphasis on code that is easy for AI agents to maintain and evolve.

## Current State (as of May 16, 2026)

### What We Have (Ground Truth)

- **Voice Input**: Real-time streaming at 24kHz with smart mic selection, barge-in, Bluetooth handling, and pre-roll jitter buffering.
- **Voice Agent Core** (`voice_agent.py`): WebSocket to xAI Realtime, tool calling, state machine (`listening` / `thinking` / `speaking`), and a coherent set of flags (`_in_tool_flow`, `_abandoned_response`, `_pending_capture_retry`) that keep state sane through cancellations and retries.
- **Tool Execution**: `run_bash`, `read_file`, `list_files`, `open_app`, `run_applescript`, plus **`read_screen_region_text`** for reading screen text aloud. Confirmation flow + live HUD feedback.
- **Screen Capture (single path, vision-only)**: Interactive `screencapture -i` → grok-4.3 vision → one-sentence semantic description injected into the conversation. The earlier Apple-OCR-as-context path was deleted after it consistently stalled xAI's Realtime backend on text-injection turns.
- **Read-aloud pipeline**: Apple Vision OCR (exact transcription) → grok-4.3 chat cleaning (strips code, URLs, chrome, timestamps, log lines) → voice model reads the prose verbatim. The last screenshot is cached so the user only drag-selects once even when both "describe this" and "read this" happen in the same breath.
- **Realtime resilience**: Stall watchdog with sliding 5s timeout; response-abandonment flag that drops late chunks instead of fire-looping or playing phantom audio post-cancel; auto-retry on capture-induced stalls (cleared on first audio = success); watchdog stays disarmed during tool-call flow.
- **Diagnostic plumbing**: Explicit handlers for server `error` events, `response.created` logging, catch-all for unknown event types. Every backend stall now produces a concrete event timeline in `~/Library/Logs/ara-agent/agent.log` instead of a silent freeze.
- **Context Layer v1** (`context.py`): App/window metadata attached to every capture packet via AppleScript.
- **UI Layer**: Floating draggable "blackhole" overlay (the canonical surface), global hotkey (Cmd+Shift+A) for capture, rumps menu bar, rotary HUD for action picking, structured Output HUD for inspection.

### Architecture Highlights

- Clean separation between voice engine, tools, screen perception, and UI.
- All major logic lives in `src/ara_agent/`.
- Deliberately minimal dependencies.
- Strong focus on native macOS integration and thread safety (asyncio ↔ AppKit).
- Hybrid perception: Apple's local strengths (exact OCR transcription) + xAI's semantic strengths (cleaning, describing) — each tool used for what it's actually best at.

### What We Don't Have Yet (Known Gaps)

These are the **next priorities** — gating items before we add more capability.

- **HUD visibility / placement**: The Output HUD is currently always-on. It should be **off by default** (the blackhole overlay alone is the agent), **toggleable** via the menu, and **draggable + resizable** so the user can place it wherever they want when they do want to inspect. The HUD is an inspection panel, not the primary surface.
- **First-response latency**: Time-to-first-audio on the opening turn feels too long, and a fluid conversation is the difference between a tool people use daily and one they abandon. Needs investigation: WS connect + `session.update` timing, the voice model's TTFT on cold sessions, pre-roll size on the very first response. This is an audio-pipeline fix, not a Realtime-backend one.

Beyond those:

- No real memory system (short-term or long-term)
- No persistent memory across sessions
- No goal tracking across turns
- No visual screen highlighting / annotation layer
- No ambient context tick — capture flows are still user-triggered only (Context Layer v2 thinking)
- No live token tracking or session-memory panel

## Vision

Build an agent that:
- Understands not just what the user says, but **what they're looking at and trying to do**
- Maintains useful context across turns without bloating the conversation
- Can take meaningful actions on the Mac while staying transparent and trustworthy
- Feels delightful and native, not like another AI tool

Long-term, we want to explore multi-agent orchestration on the xAI API, with ara-agent as both a product and a proving ground for advanced agent workflows.

## Current Priorities (May 2026)

Voice-first means voice has to feel right *first*. The next two items are gating problems before any new capability lands.

1. **HUD UX**: Hide by default; toggle from the menu; make it draggable + resizable. The blackhole overlay is the agent — the HUD is for inspection.
2. **First-response latency**: Profile and improve TTFT on the opening turn. The conversation has to feel alive from the first second, every time.

After those land:

3. **Ambient context awareness (Context Layer v2)**: probe app/window on every user turn, not only on screenshots, so Ara passively knows what the user is doing without a manual capture.
4. **Memory** when there's clear evidence of need.
5. **Richer observability**: live token tracking, per-session memory panel, response telemetry surfaced beyond stdout.

## Key Files

- `src/ara_agent/voice_agent.py` — Core orchestration, tool dispatch, Realtime state machine, resilience plumbing
- `src/ara_agent/screenshot.py` — Vision capture + Apple OCR + read-aloud cleaning pipeline
- `src/ara_agent/context.py` — Capture packet builder with app/window context
- `src/ara_agent/overlay.py` — Floating overlay, menu, hotkey wiring
- `src/ara_agent/output_hud.py` — Structured event log (next target for the HUD UX work)
- `CLAUDE.md` — How to work with Claude on this project
- `docs/adr/` — Architecture Decision Records (important design choices and their rationale)

## Philosophy & Constraints

- **Elon's 5-step algorithm** is our default thinking framework for meaningful work.
- Prioritize **simplicity and maintainability** for both humans and agents.
- Be cost-conscious with tokens during development.
- Never build "piles of shit." Every piece of code should be something we're proud to maintain long-term.
- **Diagnose, don't guess**: when something fails, reach for the event timeline first. Inferring system state from observable event flow is core to both the agent's intelligence and our debugging style.
- Keep the vision front and center: a truly intelligent, delightful macOS agent.

---

*This file is the single source of truth for the project. Update it regularly as we make progress.*
