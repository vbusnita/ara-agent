# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the ara-agent project.

## Purpose

ADRs document important architectural choices, their context, rationale, and consequences. They help future contributors (human or AI) understand *why* things are built the way they are.

## Format

Each ADR follows a lightweight but structured format including:
- Status (Proposed / Accepted / Deprecated / Superseded)
- Context
- Decision
- Consequences (positive and negative)
- Alternatives considered

## Numbering

ADRs are numbered sequentially (e.g. `0001-...`, `0002-...`).

When an ADR is superseded, a new one is created and the old one is updated to point to it.

## Current ADRs

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-hybrid-architecture.md) | Adopt Hybrid Architecture (Realtime Voice + Strong Model Brain) | Accepted | 2026-05-24 |

## Related

- [AGENT.md](../../AGENT.md) — Project bible and vision
- [CLAUDE.md](../../CLAUDE.md) — Working guidelines for contributors and AI agents