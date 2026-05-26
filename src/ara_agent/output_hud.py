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
    NSWindowStyleMaskResizable,
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


# Geometry. The HUD is now an inspection panel the user opts into via
# the menu — draggable + resizable like a normal window. WIDTH/HEIGHT
# are the initial dimensions on first show; MIN_WIDTH/MIN_HEIGHT bound
# how small the user can drag-resize it before things look broken.
WIDTH = 720
HEIGHT = 280
MIN_WIDTH = 360
MIN_HEIGHT = 120
BOTTOM_MARGIN = 56
CORNER_RADIUS = 12.0
TINT_ALPHA = 0.42        # darker tint for higher text contrast

# Persistence (NSUserDefaults)
HUD_FRAME_KEY = "ara.hudLastFrame"  # "x,y,width,height" — survives restarts

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
        self._state_text = "IDLE"
        # True until the very first show() — used to center the HUD on
        # screen one time, then leave user-chosen position alone forever
        # after (drag + resize survive hide/show cycles via AppKit).
        self._first_show = True
        self._saved_frame_rect = None   # loaded at init, applied on first show
        self._build_panel()
        self._load_saved_frame()        # read from defaults, but don't apply yet
        return self

    # ---- panel construction ----

    @objc.python_method
    def _build_panel(self):
        rect = NSRect(NSPoint(0, 0), NSSize(WIDTH, HEIGHT))
        self._panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            (NSWindowStyleMaskBorderless
             | NSWindowStyleMaskNonactivatingPanel
             | NSWindowStyleMaskResizable),
            NSBackingStoreBuffered,
            False,
        )
        self._panel.setLevel_(NSStatusWindowLevel)
        self._panel.setOpaque_(False)
        self._panel.setBackgroundColor_(NSColor.clearColor())
        self._panel.setHasShadow_(False)
        # Drag from anywhere on the panel body — the previous click-to-
        # expand toggle is gone (resize handles supersede it).
        self._panel.setMovableByWindowBackground_(True)
        self._panel.setMinSize_(NSSize(MIN_WIDTH, MIN_HEIGHT))
        self._panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        # Persist last user-chosen position/size across launches.
        # We observe move + resize and save via NSUserDefaults.
        from Foundation import NSNotificationCenter
        nc = NSNotificationCenter.defaultCenter()
        nc.addObserver_selector_name_object_(
            self, "savedFrameChanged:", "NSWindowDidMoveNotification", self._panel
        )
        nc.addObserver_selector_name_object_(
            self, "savedFrameChanged:", "NSWindowDidResizeNotification", self._panel
        )

        # Plain NSView as contentView, layer-backed so we can clip the
        # subviews (blur, tint, label, scroll) to the rounded corners.
        # (The previous _HUDClickView subclass existed only to forward
        # mouseDown to a click-to-expand toggle — that toggle was
        # retired when the panel became user-resizable.)
        content_view = NSView.alloc().initWithFrame_(rect)
        self._panel.setContentView_(content_view)
        content_view.setWantsLayer_(True)
        content_view.layer().setCornerRadius_(CORNER_RADIUS)
        content_view.layer().setMasksToBounds_(True)

        # Layer 1 (back): system HUD blur material — gives the "frosted"
        # texture and reads sharper against busy wallpapers.
        blur = _PassThroughBlur.alloc().initWithFrame_(content_view.bounds())
        blur.setMaterial_(NSVisualEffectMaterialHUDWindow)
        blur.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        blur.setState_(NSVisualEffectStateActive)
        blur.setAutoresizingMask_(
            NSViewWidthSizable | NSViewHeightSizable
        )
        content_view.addSubview_(blur)

        # Layer 2 (middle): translucent black tint above the blur for the
        # "frosted BLACK" feel — pushes the backdrop darker so white text
        # always has high contrast regardless of wallpaper.
        tint = _PassThroughTint.alloc().initWithFrame_(content_view.bounds())
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
        content_view.addSubview_(tint)

        # Layer 3a (front): status label — pinned to the top of the
        # panel. Autoresizing mask keeps it pinned as the window grows.
        status_height = 22
        status_y = HEIGHT - status_height - 4
        self._status_label = _PassThroughLabel.alloc().initWithFrame_(
            NSRect(NSPoint(0, status_y), NSSize(WIDTH, status_height))
        )
        self._configure_label(self._status_label, NSTextAlignmentCenter)
        self._status_label.setAutoresizingMask_(
            NSViewMinYMargin | NSViewWidthSizable
        )
        content_view.addSubview_(self._status_label)

        # Layer 3a.5 (front): small persistent cost line for the dedicated
        # session cost / token usage panel. Updated live when the voice
        # agent reports usage from the xAI Realtime server.
        cost_height = 18
        cost_y = status_y - cost_height - 2
        self._cost_label = _PassThroughLabel.alloc().initWithFrame_(
            NSRect(NSPoint(12, cost_y), NSSize(WIDTH - 24, cost_height))
        )
        self._configure_label(self._cost_label, NSTextAlignmentLeft)
        self._cost_label.setFont_(_monospace_font(NSFontWeightRegular))
        self._cost_label.setStringValue_("Session cost: —")
        self._cost_label.setAutoresizingMask_(
            NSViewMinYMargin | NSViewWidthSizable
        )
        content_view.addSubview_(self._cost_label)

        # Layer 3b (front): scrollable terminal-style log. NSScrollView
        # wraps an NSTextView so the user can drag-scroll long output,
        # select & copy text, and we auto-scroll to the bottom on each
        # new event. Fills the space below the status; grows with the
        # window via the autoresizing mask set further down.
        events_top_pad = 4
        events_bottom_pad = 6
        # Account for status + cost line
        cost_line_height = 18
        events_height = max(0, HEIGHT - status_height - cost_line_height - 6
                            - events_top_pad - events_bottom_pad)
        # Place the log below the cost line
        cost_line_height = 18
        events_y = events_bottom_pad + cost_line_height + 2
        events_frame = NSRect(
            NSPoint(12, events_y),
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
        content_view.addSubview_(self._events_scroll)

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
        """Bring the HUD to front. Centers on screen on the very first
        show; subsequent shows preserve whatever position/size the user
        has dragged it to."""
        self._dispatch_main(self._show_main)

    @objc.python_method
    def _show_main(self):
        try:
            # Apply user-saved position/size on the very first show.
            # Doing it here (instead of at construction) is much more
            # reliable on multi-monitor Macs — the window server and
            # display layout are stable by the time show() is called.
            if self._saved_frame_rect is not None:
                self._panel.setFrame_display_(self._saved_frame_rect, True)
                self._saved_frame_rect = None
                self._first_show = False
            elif self._first_show:
                self._center_on_anchor_screen()
                self._first_show = False
            self._panel.orderFront_(None)
        except Exception:
            log.exception("show failed")

    @objc.python_method
    def hide(self):
        self._dispatch_main(self._hide_main)

    @objc.python_method
    def _hide_main(self):
        try:
            self._panel.orderOut_(None)
        except Exception:
            log.exception("hide failed")

    @objc.python_method
    def is_visible(self) -> bool:
        try:
            return bool(self._panel.isVisible())
        except Exception:
            return False

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
            elif kind == "cost":
                # Dedicated live cost line — update the persistent label
                # and still append to the log for history.
                self._cost_label.setStringValue_(text)
                line = f"💰 {text}\n"
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

    # ---- initial placement ----

    @objc.python_method
    def _center_on_anchor_screen(self):
        """One-shot positioning on first show: centered horizontally on
        whatever screen the overlay is on, anchored a comfortable margin
        above the bottom edge. After this, user drag wins — we never
        reposition automatically again."""
        af = self._anchor_panel.frame()
        screen = self._screen_for_point(af.origin) or NSScreen.mainScreen()
        sf = screen.visibleFrame()

        current = self._panel.frame()
        width = current.size.width
        height = current.size.height
        new_x = sf.origin.x + (sf.size.width - width) / 2
        new_y = sf.origin.y + BOTTOM_MARGIN
        new_frame = NSRect(NSPoint(new_x, new_y), NSSize(width, height))
        self._panel.setFrame_display_(new_frame, True)

    @staticmethod
    @objc.python_method
    def _screen_for_point(point):
        from Foundation import NSPointInRect
        for screen in NSScreen.screens():
            if NSPointInRect(point, screen.frame()):
                return screen
        return None

    # ---- persistence helpers (frame only; visibility lives in Overlay) ----

    @objc.python_method
    def _load_saved_frame(self):
        """Read the last user position/size from defaults.
        We deliberately do NOT apply it here (too early in app lifecycle
        on multi-monitor setups). Application happens in _show_main so the
        frame + orderFront are atomic and the window server is ready."""
        from Foundation import NSUserDefaults
        defaults = NSUserDefaults.standardUserDefaults()
        frame_str = defaults.stringForKey_(HUD_FRAME_KEY)
        if not frame_str:
            self._saved_frame_rect = None
            return
        try:
            parts = [float(x) for x in frame_str.split(",")]
            if len(parts) != 4:
                self._saved_frame_rect = None
                return
            x, y, w, h = parts
            candidate = NSRect(NSPoint(x, y), NSSize(w, h))
            if self._frame_is_usable(candidate):
                self._saved_frame_rect = candidate
            else:
                self._saved_frame_rect = None
        except Exception:
            log.exception("failed to load saved HUD frame")
            self._saved_frame_rect = None

    @objc.python_method
    def _frame_is_usable(self, frame) -> bool:
        """True if the rect intersects any current screen's visible area
        (handles multi-monitor, external displays being unplugged, etc.)."""
        from Foundation import NSPointInRect
        for screen in NSScreen.screens():
            vf = screen.visibleFrame()
            # Simple intersection check
            if (frame.origin.x < vf.origin.x + vf.size.width and
                    frame.origin.x + frame.size.width > vf.origin.x and
                    frame.origin.y < vf.origin.y + vf.size.height and
                    frame.origin.y + frame.size.height > vf.origin.y):
                return True
        return False

    def savedFrameChanged_(self, _note):
        """Called on any user drag or resize. Fire-and-forget save."""
        self._save_current_frame()

    @objc.python_method
    def _save_current_frame(self):
        try:
            f = self._panel.frame()
            s = f"{f.origin.x},{f.origin.y},{f.size.width},{f.size.height}"
            from Foundation import NSUserDefaults
            defaults = NSUserDefaults.standardUserDefaults()
            defaults.setObject_forKey_(s, HUD_FRAME_KEY)
            # No explicit synchronize needed in modern macOS for small prefs
        except Exception:
            log.exception("failed to save HUD frame")
