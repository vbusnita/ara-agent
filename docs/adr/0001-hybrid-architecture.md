# ADR-0001: Adopt Hybrid Architecture (Realtime Voice + Strong Model Brain)

**Date:** 2026-05-24  
**Status:** Accepted  
**Deciders:** Victor Busnita + Grok (as engineering partner)

## Context

The initial architecture of ara-agent centered on xAI's Realtime Voice API (`grok-voice-latest` + "ara" voice) as the primary intelligence and orchestration layer. 

When the user triggered a screenshot:
1. A separate call was made to the regular Responses API (grok-4.3) to generate a semantic description of the screen.
2. That description was injected as a text `conversation.item` into the persistent Realtime voice session.
3. The Realtime model was then asked to respond to the injected context.

The floating blackhole overlay + global hotkey (`Cmd+Shift+A`) + voice interaction formed the core user experience.

## Problem

This design repeatedly produced "service degraded" states and stalled responses, particularly after vision (screenshot) turns. The Realtime Voice endpoint proved unreliable once non-trivial text was injected, especially when followed by tool use or further reasoning. 

Symptoms observed over multiple days:
- Responses would start (sometimes even speak a partial acknowledgment) then go silent.
- The stall watchdog would fire consistently on vision-injection turns.
- No usage data would be returned for the failed turn.
- Restarting the agent sometimes helped temporarily, but the problem was systemic.

This made it extremely difficult to build and test the desired "supervisor-mode" capabilities (using screenshots to drive real operational work like network mapping and security tasks).

## Decision

We are adopting a **hybrid architecture**:

- The **Realtime Voice API** remains responsible for low-latency audio input/output, natural barge-in, and high-quality voice synthesis using the "ara" voice.
- A new **Brain** component (using the regular xAI Responses API) becomes responsible for vision understanding, reasoning, planning, and tool calling.

In this model:
- Screenshots and complex context are sent to the Brain (strong model) rather than being text-injected into the Realtime session.
- The Brain decides what should be said or done.
- The voice layer is primarily used to speak the Brain's output and handle voice input.

## Consequences

### Positive
- Significantly higher reliability for flows involving screenshots and real work.
- Clear separation of concerns: Voice I/O vs. Intelligence & Tool Use.
- Easier to use the strongest available model for hard reasoning tasks.
- Better cost observability (we can distinguish "voice turn" cost from "brain reasoning + vision" cost).
- More aligned with the long-term goal of a visible, supervisor-controlled agent that can perform meaningful operational work.

### Negative / Trade-offs
- Increased system complexity (need to coordinate two model sessions).
- Slightly higher latency on "screenshot → first spoken response" paths.
- Requires building new coordination logic between the Voice layer and the Brain.
- The Realtime session may still be used for simpler voice-only interactions.

## Alternatives Considered

1. **Double down on pure Realtime**  
   Add more aggressive recovery, fresh connections on degradation, etc.  
   *Rejected* — the unreliability appeared fundamental to text injection + tool use on the current Realtime endpoint.

2. **Abandon Realtime entirely**  
   Use regular Responses API + separate TTS (e.g., via another service or xAI's TTS).  
   *Rejected for now* — the quality and latency of the current Realtime "ara" voice is a core part of the delightful experience. We want to keep it for voice I/O.

3. **Hybrid model (chosen)**  
   Realtime for voice, strong Responses model for reasoning and vision.

## Related Documents

- [AGENT.md](../../AGENT.md) — Current project vision and priorities
- [CLAUDE.md](../../CLAUDE.md) — Working practices
- Ongoing work on dedicated Cost HUD (for visibility across both voice and brain paths)

## Notes

This decision was made after repeated real-world friction with the pure Realtime + injection approach while trying to build supervisor-mode capabilities. The hybrid model is expected to be the foundation for future "visible agent" work where the user can watch the agent perform actions on screen.