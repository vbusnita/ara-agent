"""Output HUD for ara-agent.

A persistent floating panel at the bottom-middle of whatever screen the
blackhole overlay lives on. Two visual states:

  • MINIMIZED — a single line of bright text showing the agent's
    current state (LISTENING / THINKING / SPEAKING). Always visible
    while listening. Glanceable, never blocks.

  • EXPANDED — same status line plus a recent-events log: tool calls,
    results, screenshot captures, etc. Click the strip to expand.
    Click again to minimize.

Visual style follows the same minimalist language as the rotary HUD:
no background box, just floating uppercase text with drop shadow for
legibility against any wallpaper. Letter-spacing + weight contrast
give it a technical/instrumented feel.

The HUD subscribes to two callbacks on AraAgent:
  - state_callback  → set_state()
  - event_callback  → log_event()

Geometry is recomputed on every show() against the overlay's current
screen, so dragging the overlay across monitors moves the HUD too.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import objc
from AppKit import (
    NSAnimationContext,
    NSAttributedString,
    NSBackingStoreBuffered,
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSFontWeightMedium,
    NSFontWeightRegular,
    NSForegroundColorAttributeName,
    NSKernAttributeName,
    NSMutableAttributedString,
    NSMutableParagraphStyle,
    NSPanel,
    NSParagraphStyleAttributeName,
    NSScreen,
    NSScrollView,
    NSShadow,
    NSShadowAttributeName,
    NSStatusWindowLevel,
    NSTextAlignmentCenter,
    NSTextAlignmentLeft,
    NSTextField,
    NSTextView,
    NSView,
    NSViewHeightSizable,
    NSViewMinYMargin,
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
from Foundation import (
    NSMakeRange,
    NSObject,
    NSOperationQueue,
    NSPoint,
    NSRect,
    NSSize,
)
from Quartz import CGColorCreateGenericGray

import logging

log = logging.getLogger(__name__)


# Geometry
WIDTH = 720
HEIGHT_MIN = 38
HEIGHT_EXPANDED = 280
BOTTOM_MARGIN = 56
CORNER_RADIUS = 12.0
TINT_ALPHA = 0.42        # darker tint for higher text contrast

# Animation
ANIM_DURATION = 0.22

# Typography
STATUS_SIZE = 13.0
STATUS_KERN = 2.6
EVENT_SIZE = 11.0
EVENT_LINE_SPACING = 2.0
SHADOW_BLUR = 5.0
SHADOW_ALPHA = 0.85

# Event log
# Keep the text storage bounded so a runaway response doesn't grow it
# forever. ~80 KB ≈ many screens of terminal history; we trim the oldest
# half when we cross the cap.
MAX_LOG_CHARS = 80_000
TRIM_TO_CHARS = 40_000

# Text styling alpha — uniform, no fade. Commands brightest, output dim.
COMMAND_ALPHA = 1.0
OUTPUT_ALPHA = 0.78
INFO_ALPHA = 0.66


# ----- attribute helpers -----

def _shadow():
    """Soft drop shadow for legibility on any background."""
    s = NSShadow.alloc().init()
    s.setShadowColor_(NSColor.colorWithWhite_alpha_(0.0, SHADOW_ALPHA))
    s.setShadowOffset_(NSSize(0, -1))
    s.setShadowBlurRadius_(SHADOW_BLUR)
    return s


def _paragraph(alignment, line_spacing=0.0):
    p = NSMutableParagraphStyle.alloc().init()
    p.setAlignment_(alignment)
    if line_spacing:
        p.setLineSpacing_(line_spacing)
    return p


def _status_string(text: str) -> NSAttributedString:
    font = NSFont.systemFontOfSize_weight_(STATUS_SIZE, NSFontWeightMedium)
    attrs = {
        NSFontAttributeName: font,
        NSForegroundColorAttributeName: NSColor.whiteColor(),
        NSKernAttributeName: STATUS_KERN,
        NSShadowAttributeName: _shadow(),
        NSParagraphStyleAttributeName: _paragraph(NSTextAlignmentCenter),
    }
    return NSAttributedString.alloc().initWithString_attributes_(
        text.upper(), attrs,
    )


def _monospace_font(weight=NSFontWeightRegular):
    """SF Mono via the system monospaced font helper (macOS 10.15+).
    Gives the proper 'terminal' feel and lines up tabular output."""
    return NSFont.monospacedSystemFontOfSize_weight_(EVENT_SIZE, weight)


def _command_string(text: str) -> NSAttributedString:
    """Bright monospace line for tool calls / commands."""
    font = _monospace_font(NSFontWeightMedium)
    attrs = {
        NSFontAttributeName: font,
        NSForegroundColorAttributeName: NSColor.colorWithWhite_alpha_(
            1.0, COMMAND_ALPHA
        ),
        NSShadowAttributeName: _shadow(),
        NSParagraphStyleAttributeName: _paragraph(
            NSTextAlignmentLeft, line_spacing=EVENT_LINE_SPACING,
        ),
    }
    return NSAttributedString.alloc().initWithString_attributes_(text, attrs)


def _output_string(text: str) -> NSAttributedString:
    """Slightly dimmer monospace block for tool output."""
    font = _monospace_font(NSFontWeightRegular)
    attrs = {
        NSFontAttributeName: font,
        NSForegroundColorAttributeName: NSColor.colorWithWhite_alpha_(
            1.0, OUTPUT_ALPHA
        ),
        NSShadowAttributeName: _shadow(),
        NSParagraphStyleAttributeName: _paragraph(
            NSTextAlignmentLeft, line_spacing=EVENT_LINE_SPACING,
        ),
    }
    return NSAttributedString.alloc().initWithString_attributes_(text, attrs)


def _info_string(text: str) -> NSAttributedString:
    """Muted line for system messages — captures, status, etc."""
    font = _monospace_font(NSFontWeightRegular)
    attrs = {
        NSFontAttributeName: font,
        NSForegroundColorAttributeName: NSColor.colorWithWhite_alpha_(
            1.0, INFO_ALPHA
        ),
        NSShadowAttributeName: _shadow(),
        NSParagraphStyleAttributeName: _paragraph(
            NSTextAlignmentLeft, line_spacing=EVENT_LINE_SPACING,
        ),
    }
    return NSAttributedString.alloc().initWithString_attributes_(text, attrs)


# ----- click-target view -----

class _HUDClickView(NSView):
    """NSView subclass that forwards mouseDown to a Python callback.
    The plain contentView from NSPanel doesn't itself respond to clicks;
    swapping in this subclass gives us a place to handle the toggle."""

    def initWithFrame_(self, frame):
        self = objc.super(_HUDClickView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._on_click = None
        return self

    @objc.python_method
    def set_on_click(self, callback):
        self._on_click = callback

    def mouseDown_(self, event):
        if self._on_click is not None:
            self._on_click()


class _PassThroughBlur(NSVisualEffectView):
    """Visual-effect blur that declines hits so clicks pass through to
    the underlying _HUDClickView."""

    def hitTest_(self, point):
        return None


class _PassThroughLabel(NSTextField):
    """NSTextField that declines hits — even non-editable labels normally
    absorb mouse events. This lets clicks anywhere on the HUD reach the
    contentView's mouseDown_."""

    def hitTest_(self, point):
        return None


class _PassThroughTint(NSView):
    """Plain view used for a translucent black tint layered above the
    blur for extra darkness. Click-through."""

    def hitTest_(self, point):
        return None


# ----- HUD controller -----

class OutputHUD(NSObject):
    """Manages the HUD panel: lifecycle, state line, event log."""

    def initWithAnchor_(self, anchor_panel):
        self = objc.super(OutputHUD, self).init()
        if self is None:
            return None
        self._anchor_panel = anchor_panel
        self._expanded = False
        self._state_text = "IDLE"
        self._build_panel()
        return self

    # ---- panel construction ----

    @objc.python_method
    def _build_panel(self):
        rect = NSRect(NSPoint(0, 0), NSSize(WIDTH, HEIGHT_MIN))
        self._panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        self._panel.setLevel_(NSStatusWindowLevel)
        self._panel.setOpaque_(False)
        self._panel.setBackgroundColor_(NSColor.clearColor())
        self._panel.setHasShadow_(False)
        self._panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        # Custom contentView so we can intercept clicks (toggle expand).
        click_view = _HUDClickView.alloc().initWithFrame_(rect)
        click_view.set_on_click(self._on_click)
        self._panel.setContentView_(click_view)
        click_view.setWantsLayer_(True)
        # Clip everything to the rounded rectangle.
        click_view.layer().setCornerRadius_(CORNER_RADIUS)
        click_view.layer().setMasksToBounds_(True)

        # Layer 1 (back): system HUD blur material — gives the "frosted"
        # texture and reads sharper against busy wallpapers.
        blur = _PassThroughBlur.alloc().initWithFrame_(click_view.bounds())
        blur.setMaterial_(NSVisualEffectMaterialHUDWindow)
        blur.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        blur.setState_(NSVisualEffectStateActive)
        blur.setAutoresizingMask_(
            NSViewWidthSizable | NSViewHeightSizable
        )
        click_view.addSubview_(blur)

        # Layer 2 (middle): translucent black tint above the blur for the
        # "frosted BLACK" feel — pushes the backdrop darker so white text
        # always has high contrast regardless of wallpaper.
        tint = _PassThroughTint.alloc().initWithFrame_(click_view.bounds())
        tint.setWantsLayer_(True)
        # CGColorCreateGenericGray returns a properly-typed CGColorRef,
        # avoiding pyobjc's "PyObjCPointer created" warning we'd get from
        # NSColor.CGColor() (which pyobjc can't fully bridge).
        tint.layer().setBackgroundColor_(
            CGColorCreateGenericGray(0.0, TINT_ALPHA)
        )
        tint.setAutoresizingMask_(
            NSViewWidthSizable | NSViewHeightSizable
        )
        click_view.addSubview_(tint)

        # Layer 3a (front): status label — pinned to the top, fixed height.
        status_height = 22
        status_y = HEIGHT_MIN - status_height - 4
        self._status_label = _PassThroughLabel.alloc().initWithFrame_(
            NSRect(NSPoint(0, status_y), NSSize(WIDTH, status_height))
        )
        self._configure_label(self._status_label, NSTextAlignmentCenter)
        self._status_label.setAutoresizingMask_(
            NSViewMinYMargin | NSViewWidthSizable
        )
        click_view.addSubview_(self._status_label)

        # Layer 3b (front): scrollable terminal-style log. NSScrollView
        # wraps an NSTextView so the user can drag-scroll long output,
        # select & copy text, and we auto-scroll to the bottom on each
        # new event. Sits below the status, fills the rest.
        events_top_pad = 4
        events_bottom_pad = 6
        events_height = max(0, HEIGHT_MIN - status_height - 4
                            - events_top_pad - events_bottom_pad)
        events_frame = NSRect(
            NSPoint(12, events_bottom_pad),
            NSSize(WIDTH - 24, events_height),
        )

        self._events_scroll = NSScrollView.alloc().initWithFrame_(events_frame)
        self._events_scroll.setBorderType_(0)  # NSNoBorder
        self._events_scroll.setDrawsBackground_(False)
        self._events_scroll.setHasVerticalScroller_(True)
        self._events_scroll.setHasHorizontalScroller_(False)
        self._events_scroll.setAutohidesScrollers_(True)
        self._events_scroll.setAutoresizingMask_(
            NSViewWidthSizable | NSViewHeightSizable
        )

        self._events_view = NSTextView.alloc().initWithFrame_(
            self._events_scroll.bounds()
        )
        self._events_view.setEditable_(False)
        self._events_view.setSelectable_(True)
        self._events_view.setDrawsBackground_(False)
        self._events_view.setTextContainerInset_(NSSize(0, 0))
        self._events_view.setVerticallyResizable_(True)
        self._events_view.setHorizontallyResizable_(False)
        self._events_view.setAutoresizingMask_(NSViewWidthSizable)
        # Match content width to the scroll view so long lines wrap
        # rather than spawn a horizontal scrollbar.
        text_container = self._events_view.textContainer()
        text_container.setWidthTracksTextView_(True)
        text_container.setContainerSize_(
            NSSize(events_frame.size.width, 1.0e7)
        )

        self._events_scroll.setDocumentView_(self._events_view)
        click_view.addSubview_(self._events_scroll)

        # Initial render
        self._render_status()

    @objc.python_method
    def _configure_label(self, label, alignment, multiline: bool = False):
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setAlignment_(alignment)
        if multiline:
            label.setUsesSingleLineMode_(False)
            label.setMaximumNumberOfLines_(0)
            label.setLineBreakMode_(0)  # NSLineBreakByWordWrapping
        else:
            label.setUsesSingleLineMode_(True)

    # ---- public API ----

    @objc.python_method
    def show(self):
        """Position the HUD and bring it to front. Always starts minimized."""
        self._dispatch_main(self._show_main)

    @objc.python_method
    def _show_main(self):
        try:
            self._expanded = False
            self._reposition(animate=False)
            self._panel.orderFront_(None)
        except Exception:
            log.exception("show failed")

    @objc.python_method
    def hide(self):
        self._dispatch_main(self._hide_main)

    @objc.python_method
    def _hide_main(self):
        try:
            self._expanded = False
            self._panel.orderOut_(None)
        except Exception:
            log.exception("hide failed")

    @objc.python_method
    def set_state(self, state: str):
        """Update the bright status line. May be called from any thread;
        AppKit text mutation is strictly main-thread, so we dispatch."""
        self._dispatch_main(lambda: self._set_state_main(state))

    @objc.python_method
    def _set_state_main(self, state: str):
        try:
            self._state_text = state.upper()
            self._render_status()
        except Exception:
            log.exception("set_state failed (state=%r)", state)

    @objc.python_method
    def log_event(self, kind: str, text: str):
        """Append an event to the scrollable log. May be called from any
        thread (typically the asyncio loop). Marshals the actual NSText
        mutation onto the main thread — NSTextView.textStorage is NOT
        thread-safe and will crash if you mutate it from a worker."""
        self._dispatch_main(lambda: self._log_event_main(kind, text))

    @objc.python_method
    def _log_event_main(self, kind: str, text: str):
        try:
            if kind == "call":
                line = f"\n› {text}\n"
                attr = _command_string(line)
            elif kind == "result":
                line = self._indent_output(text)
                attr = _output_string(line)
            elif kind == "info":
                line = f"· {text}\n"
                attr = _info_string(line)
            elif kind == "warn":
                line = f"⚠ {text}\n"
                attr = _info_string(line)
            else:
                line = f"{text}\n"
                attr = _info_string(line)
            self._append_to_log(attr)
        except Exception:
            log.exception(
                "log_event failed (kind=%r, text=%r)", kind, text[:200],
            )

    @staticmethod
    @objc.python_method
    def _dispatch_main(callable_):
        """Schedule a Python callable on the main thread.
        Returns immediately."""
        NSOperationQueue.mainQueue().addOperationWithBlock_(callable_)

    @objc.python_method
    def _append_to_log(self, attr: NSAttributedString):
        storage = self._events_view.textStorage()
        storage.beginEditing()
        storage.appendAttributedString_(attr)
        # Cap total length — trim the oldest half when we cross the cap.
        if storage.length() > MAX_LOG_CHARS:
            excess = storage.length() - TRIM_TO_CHARS
            storage.deleteCharactersInRange_(NSMakeRange(0, excess))
        storage.endEditing()
        # Auto-scroll to the bottom so the newest line is visible.
        end = storage.length()
        self._events_view.scrollRangeToVisible_(NSMakeRange(end, 0))

    @staticmethod
    @objc.python_method
    def _indent_output(text: str) -> str:
        """Indent multi-line tool output 2 spaces so it visually nests
        under its preceding command line."""
        if not text:
            return "  (no output)\n"
        lines = text.rstrip("\n").split("\n")
        return "".join(f"  {line}\n" for line in lines)

    # ---- rendering ----

    @objc.python_method
    def _render_status(self):
        self._status_label.setAttributedStringValue_(
            _status_string(self._state_text)
        )

    # ---- expand / collapse ----

    @objc.python_method
    def _on_click(self):
        self._expanded = not self._expanded
        self._reposition(animate=True)

    @objc.python_method
    def _reposition(self, animate: bool):
        """Recompute frame: centered horizontally on the overlay's screen,
        anchored to the bottom edge. Height depends on expand state."""
        af = self._anchor_panel.frame()
        screen = self._screen_for_point(af.origin) or NSScreen.mainScreen()
        sf = screen.visibleFrame()

        height = HEIGHT_EXPANDED if self._expanded else HEIGHT_MIN
        new_x = sf.origin.x + (sf.size.width - WIDTH) / 2
        new_y = sf.origin.y + BOTTOM_MARGIN
        new_frame = NSRect(NSPoint(new_x, new_y), NSSize(WIDTH, height))

        if animate:
            NSAnimationContext.beginGrouping()
            NSAnimationContext.currentContext().setDuration_(ANIM_DURATION)
            self._panel.animator().setFrame_display_(new_frame, True)
            NSAnimationContext.endGrouping()
        else:
            self._panel.setFrame_display_(new_frame, True)

    @staticmethod
    @objc.python_method
    def _screen_for_point(point):
        from Foundation import NSPointInRect
        for screen in NSScreen.screens():
            if NSPointInRect(point, screen.frame()):
                return screen
        return None
