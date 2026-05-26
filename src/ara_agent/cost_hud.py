"""
Dedicated floating cost / token usage panel for ara-agent.

This is intentionally a completely separate surface from the main
OutputHUD (event log). The user wants a clean, always-glanceable
cost tracker.

It tracks:
- Current session (resets when you Start/Stop Listening)
- Lifetime (keeps growing for the life of the app process)

Pricing is hardcoded and must be updated manually when xAI changes rates.
There is no public xAI API endpoint that returns current model pricing.
"""

from __future__ import annotations

import logging
from typing import Optional

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSFontWeightMedium,
    NSFontWeightRegular,
    NSForegroundColorAttributeName,
    NSKernAttributeName,
    NSPanel,
    NSShadow,
    NSShadowAttributeName,
    NSStatusWindowLevel,
    NSTextAlignmentLeft,
    NSTextAlignmentRight,
    NSTextField,
    NSView,
    NSViewHeightSizable,
    NSViewWidthSizable,
    NSVisualEffectBlendingModeBehindWindow,
    NSVisualEffectMaterialHUDWindow,
    NSVisualEffectStateActive,
    NSVisualEffectView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSMakeRect, NSObject, NSPoint, NSRect, NSSize

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing (must be maintained manually)
#
# xAI does not expose current model pricing via any public API endpoint.
# You have to copy the rates from the xAI console / docs when they change.
#
# These are example/placeholder rates for grok-voice-latest.
# Replace with the real numbers from your xAI account.
# ---------------------------------------------------------------------------

PRICING = {
    # Per 1 million tokens
    "voice_text_input_per_m": 3.00,
    "voice_text_output_per_m": 12.00,
    "voice_audio_input_per_m": 30.00,
    "voice_audio_output_per_m": 120.00,
}

# How often we update the UI (we accumulate in the background anyway)
UPDATE_THROTTLE_SEC = 0.0


def _calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Very rough estimate using the rates above.
    In reality the Realtime voice model has more nuanced tokenization
    (cached tokens, audio vs text, etc.). This is good enough for oversight.
    """
    # For now we treat everything as "text" rate.
    # When we have better breakdown from the usage dict we can improve this.
    text_in = input_tokens
    text_out = output_tokens

    cost = (
        (text_in / 1_000_000) * PRICING["voice_text_input_per_m"]
        + (text_out / 1_000_000) * PRICING["voice_text_output_per_m"]
    )
    return round(cost, 4)


class CostHUD(NSObject):
    """Small dedicated floating panel showing session + lifetime cost."""

    def init(self):
        self = objc.super(CostHUD, self).init()
        if self is None:
            return None

        self._session_in: int = 0
        self._session_out: int = 0
        self._lifetime_in: int = 0
        self._lifetime_out: int = 0

        # In-flight tracking for visibility during degraded / slow turns
        self._in_flight: int = 0
        self._last_turn_label: str = ""

        self._build_panel()
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_usage(self, usage: dict) -> None:
        """Called when the server reports usage for a completed turn."""
        if not usage:
            return

        inp = (
            usage.get("input_tokens")
            or usage.get("input_tokens_details", {}).get("text_tokens", 0)
            or 0
        )
        out = (
            usage.get("output_tokens")
            or usage.get("output_tokens_details", {}).get("text_tokens", 0)
            or 0
        )
        audio_in = usage.get("input_tokens_details", {}).get("audio_tokens", 0) or 0
        audio_out = usage.get("output_tokens_details", {}).get("audio_tokens", 0) or 0

        self._session_in += inp + audio_in
        self._session_out += out + audio_out
        self._lifetime_in += inp + audio_in
        self._lifetime_out += out + audio_out

        if self._in_flight > 0:
            self._in_flight -= 1
            if self._in_flight < 0:
                self._in_flight = 0

        self._update_labels()

    def handle_cost_message(self, msg: dict) -> None:
        """Generic entry point used by the cost_callback from the voice agent.
        Supports both usage reports and turn-start notifications.
        """
        if not msg:
            return
        if msg.get("type") == "turn_start":
            self.start_turn(msg.get("label", "Turn"))
        elif msg.get("type") == "usage":
            self.add_usage(msg.get("usage", {}))

    def start_turn(self, label: str = "Turn") -> None:
        """Called when we are about to start a turn that may consume tokens
        (e.g. sending a vision packet or creating a programmatic response).
        This lets the panel show 'in-flight' activity even before the server
        reports usage.
        """
        self._in_flight += 1
        self._last_turn_label = label
        self._update_labels()

    def new_session(self) -> None:
        """Called when the user starts a fresh listening session.
        Resets only the current session counters + in-flight state.
        Lifetime keeps accumulating.
        """
        self._session_in = 0
        self._session_out = 0
        self._in_flight = 0
        self._last_turn_label = ""
        self._update_labels()

    def show(self) -> None:
        self._panel.orderFront_(None)

    def hide(self) -> None:
        self._panel.orderOut_(None)

    # ------------------------------------------------------------------
    # Internal UI
    # ------------------------------------------------------------------

    def _build_panel(self):
        width, height = 320, 78
        self._panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSRect(NSPoint(0, 0), NSSize(width, height)),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        self._panel.setLevel_(NSStatusWindowLevel)
        self._panel.setOpaque_(False)
        self._panel.setBackgroundColor_(NSColor.clearColor())
        self._panel.setHasShadow_(True)
        self._panel.setMovableByWindowBackground_(True)
        self._panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        content = NSView.alloc().initWithFrame_(NSRect(NSPoint(0, 0), NSSize(width, height)))
        content.setWantsLayer_(True)
        content.layer().setCornerRadius_(10)
        content.layer().setMasksToBounds_(True)
        self._panel.setContentView_(content)

        # Frosted background
        blur = NSVisualEffectView.alloc().initWithFrame_(content.bounds())
        blur.setMaterial_(NSVisualEffectMaterialHUDWindow)
        blur.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        blur.setState_(NSVisualEffectStateActive)
        blur.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        content.addSubview_(blur)

        # Dark tint
        tint = NSView.alloc().initWithFrame_(content.bounds())
        tint.setWantsLayer_(True)
        tint.layer().setBackgroundColor_(
            NSColor.colorWithWhite_alpha_(0.0, 0.45).CGColor()
        )
        tint.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        content.addSubview_(tint)

        # Labels
        self._session_label = self._make_label(12, 52, width - 24, 22, "Session")
        self._lifetime_label = self._make_label(12, 18, width - 24, 22, "Lifetime")

        content.addSubview_(self._session_label)
        content.addSubview_(self._lifetime_label)

        self._update_labels()

    def _make_label(self, x, y, w, h, title):
        label = NSTextField.alloc().initWithFrame_(NSRect(NSPoint(x, y), NSSize(w, h)))
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setFont_(NSFont.monospacedSystemFontOfSize_weight_(11, NSFontWeightRegular))
        label.setTextColor_(NSColor.whiteColor())
        label.setStringValue_(f"{title}: 0 in / 0 out  ($0.0000)")
        return label

    def _update_labels(self):
        sess_cost = _calculate_cost(self._session_in, self._session_out)
        life_cost = _calculate_cost(self._lifetime_in, self._lifetime_out)

        session_line = (
            f"Session:  {self._session_in:,} in / {self._session_out:,} out   ${sess_cost:.4f}"
        )
        if self._in_flight > 0:
            label = self._last_turn_label or "turn"
            # Keep it short so it fits in the narrow panel
            short_label = label[:18] + "…" if len(label) > 19 else label
            session_line += f"  [+{self._in_flight} {short_label}]"

        self._session_label.setStringValue_(session_line)
        self._lifetime_label.setStringValue_(
            f"Lifetime: {self._lifetime_in:,} in / {self._lifetime_out:,} out   ${life_cost:.4f}"
        )