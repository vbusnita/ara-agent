"""Rotary-wheel HUD that appears next to the overlay icon.

Shows three stacked labels at once: the previously-selectable item
(top, half-opaque), the currently-selected item (middle, full
opacity), and the next-selectable item (bottom, half-opaque). Each
hotkey press cycles selection by one. After a brief idle period
(no further presses), the currently-selected item fires and the HUD
fades out.

Positioning respects multi-screen setups — the HUD finds the screen
the anchor overlay is on, then sits on the opposite side of that
screen's midline so it never clips off-screen.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import objc
from AppKit import (
    NSAttributedString,
    NSBackingStoreBuffered,
    NSColor,
    NSFont,
    NSFontWeightMedium,
    NSFontWeightLight,
    NSPanel,
    NSScreen,
    NSShadow,
    NSStatusWindowLevel,
    NSTextAlignmentLeft,
    NSTextAlignmentRight,
    NSTextField,
    NSTimer,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import (
    NSDictionary,
    NSObject,
    NSPoint,
    NSRect,
    NSSize,
)


# Neuralink-inspired: no box, just floating text. Heavy letter-spacing,
# uppercase selected item, generous size and weight contrast against
# dimmer adjacent items. Drop shadow gives legibility on any wallpaper
# without resorting to a background panel.

HUD_WIDTH = 240
HUD_HEIGHT = 120
LABEL_HEIGHT = 24
LABEL_GAP = 18                # extra vertical space between labels so the
                              # 3D tilt doesn't visually overlap them
CONFIRMATION_TIMEOUT = 0.9    # seconds idle before selected item fires
EDGE_MARGIN = 14              # gap between overlay and HUD

SELECTED_SIZE = 14.0
SELECTED_KERN = 2.6           # letter-spacing in points, "spread out" feel
ADJACENT_SIZE = 10.0
ADJACENT_KERN = 1.4
ADJACENT_ALPHA = 0.22         # almost-ghost above/below

SHADOW_BLUR = 5.0
SHADOW_ALPHA = 0.85

# In-plane tilt (Z-axis rotation) gives a fan-out look that doesn't
# depend on perspective. Adjacent labels rotate around their center;
# selected stays horizontal. Fan opens to the side the HUD sits on
# relative to the icon (we flip the sign when the HUD is on the icon's
# left vs right) so the items appear to radiate FROM the icon.
TILT_DEGREES = 14.0


# String attribute keys — pyobjc accepts the canonical NSAttributedString
# attribute name constants from AppKit's text system.
from AppKit import (
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSKernAttributeName,
    NSShadowAttributeName,
    NSParagraphStyleAttributeName,
    NSMutableParagraphStyle,
)


def _make_rotation_z(angle_rad: float):
    """Build a CATransform3D for rotation around the Z-axis (in-plane).
    Returned as the 16-element flat tuple matching CATransform3D's struct
    layout (row-major: m11, m12, m13, m14, m21, m22, ...)."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return (
         c,  s,  0.0, 0.0,
        -s,  c,  0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0,
    )


def _make_label_string(text: str, selected: bool, alignment) -> NSAttributedString:
    """Build the NSAttributedString for one rotary label.

    Selected line: medium weight, larger, full opacity, more letter-spacing.
    Adjacent lines: light weight, smaller, low alpha — visually they fade
    so the eye lands on the selected line.

    Drop shadow goes on the text glyphs (not the view), via the attributed
    string's shadow attribute. That way no background panel is needed and
    the text remains legible against any wallpaper.
    """
    if selected:
        font = NSFont.systemFontOfSize_weight_(SELECTED_SIZE, NSFontWeightMedium)
        kern = SELECTED_KERN
        color = NSColor.whiteColor()
    else:
        font = NSFont.systemFontOfSize_weight_(ADJACENT_SIZE, NSFontWeightLight)
        kern = ADJACENT_KERN
        color = NSColor.colorWithWhite_alpha_(1.0, ADJACENT_ALPHA)

    shadow = NSShadow.alloc().init()
    shadow.setShadowColor_(NSColor.colorWithWhite_alpha_(0.0, SHADOW_ALPHA))
    shadow.setShadowOffset_(NSSize(0, -1))
    shadow.setShadowBlurRadius_(SHADOW_BLUR)

    para = NSMutableParagraphStyle.alloc().init()
    para.setAlignment_(alignment)

    attrs = {
        NSFontAttributeName: font,
        NSForegroundColorAttributeName: color,
        NSKernAttributeName: kern,
        NSShadowAttributeName: shadow,
        NSParagraphStyleAttributeName: para,
    }
    return NSAttributedString.alloc().initWithString_attributes_(text, attrs)


class RotaryHUD(NSObject):
    """Floating panel showing a 3-item rotary menu."""

    def init(self):
        self = objc.super(RotaryHUD, self).init()
        if self is None:
            return None

        self._items: List[Tuple[str, Callable[[], None]]] = []
        self._index = 0
        self._anchor_panel = None  # the overlay panel; set per-show
        self._confirm_timer: Optional[NSTimer] = None
        self._alignment = NSTextAlignmentLeft  # updated per-show based on side
        self._build_panel()
        return self

    # ---- panel construction ----

    @objc.python_method
    def _build_panel(self):
        rect = NSRect(NSPoint(0, 0), NSSize(HUD_WIDTH, HUD_HEIGHT))
        self._panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect,
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        self._panel.setLevel_(NSStatusWindowLevel)
        self._panel.setOpaque_(False)
        self._panel.setBackgroundColor_(NSColor.clearColor())
        # No window shadow — the "box" feeling we want gone comes from
        # both the background view and the OS-drawn window shadow.
        self._panel.setHasShadow_(False)
        self._panel.setIgnoresMouseEvents_(True)
        self._panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        content = self._panel.contentView()
        content.setWantsLayer_(True)
        # Force subview drawing through the layer hierarchy. Without
        # this, NSTextField's internal rendering may bypass parent layer
        # transforms (which is what was eating my X-axis tilt before).
        content.setCanDrawSubviewsIntoLayer_(True)

        # Each label goes in its own NSView container; the container —
        # not the NSTextField — receives the tilt transform. NSControl
        # subclasses have internal rendering layers that don't honor
        # a direct setTransform_ on the field's own layer; wrapping in a
        # plain NSView gives a transformable parent layer that the
        # label cleanly composites inside.
        self._labels: List[NSTextField] = []
        for slot in range(3):
            frame = self._frame_for_slot(slot)
            container = NSView.alloc().initWithFrame_(frame)
            container.setWantsLayer_(True)
            container.layer().setMasksToBounds_(False)

            # In-plane (Z-axis) rotation — definitely visible, no
            # perspective needed. Adjacent slots fan out symmetrically.
            tilt = math.radians(TILT_DEGREES)
            if slot == 0:
                # Top label tilts CCW: text leans up to the right.
                container.layer().setTransform_(_make_rotation_z(tilt))
            elif slot == 2:
                # Bottom label tilts CW: text leans down to the right.
                container.layer().setTransform_(_make_rotation_z(-tilt))

            label = NSTextField.labelWithString_("")
            label.setBezeled_(False)
            label.setDrawsBackground_(False)
            label.setEditable_(False)
            label.setSelectable_(False)
            label.setFrame_(NSRect(
                NSPoint(0, 0),
                container.bounds().size,
            ))
            container.addSubview_(label)
            content.addSubview_(container)
            self._labels.append(label)

    @objc.python_method
    def _frame_for_slot(self, slot: int) -> NSRect:
        """Return the NSRect for the container at the given slot.

        slot: 0 = top (prev), 1 = middle (selected), 2 = bottom (next).
        AppKit y origin is bottom-left; slot 0 sits highest. Adjacent
        slots get extra room because the tilt foreshortens them visually."""
        center_y = HUD_HEIGHT / 2
        offsets = {
            0:  LABEL_HEIGHT + LABEL_GAP,
            1:  0,
            2: -(LABEL_HEIGHT + LABEL_GAP),
        }
        y = center_y + offsets[slot] - LABEL_HEIGHT / 2
        # Inset slightly from edges so kerning + shadow don't clip.
        return NSRect(
            NSPoint(8, y),
            NSSize(HUD_WIDTH - 16, LABEL_HEIGHT),
        )

    # ---- public API ----

    @objc.python_method
    def present(
        self,
        items: List[Tuple[str, Callable[[], None]]],
        anchor_panel,
    ):
        """Show (or refresh) the HUD with the given items. Anchor panel
        is used to position the HUD relative to its current frame."""
        self._items = list(items)
        self._index = 0
        self._anchor_panel = anchor_panel
        self._refresh_labels()
        self._position_relative_to_anchor()
        self._panel.orderFront_(None)
        self._restart_confirm_timer()

    @objc.python_method
    def advance(self):
        """Cycle selection one position. No-op if HUD isn't currently shown."""
        if not self._items or not self._panel.isVisible():
            return
        self._index = (self._index + 1) % len(self._items)
        self._refresh_labels()
        self._restart_confirm_timer()

    @objc.python_method
    def hide(self):
        self._cancel_confirm_timer()
        self._panel.orderOut_(None)

    # ---- internal ----

    @objc.python_method
    def _refresh_labels(self):
        n = len(self._items)
        if n == 0:
            return

        if n == 1:
            # Solo item: just show it in the middle, blanks above and below.
            self._set_label(0, "", selected=False)
            self._set_label(1, self._items[0][0], selected=True)
            self._set_label(2, "", selected=False)
        elif n == 2:
            # Two items: avoid the wrap-around mirror (where prev and next
            # would both be the same item). Show the selected one in the
            # middle and the alternative on whichever side it naturally
            # belongs given the current index.
            if self._index == 0:
                # Item 0 selected → blank above, item 1 below.
                self._set_label(0, "", selected=False)
                self._set_label(1, self._items[0][0], selected=True)
                self._set_label(2, self._items[1][0], selected=False)
            else:
                # Item 1 selected → item 0 above, blank below.
                self._set_label(0, self._items[0][0], selected=False)
                self._set_label(1, self._items[1][0], selected=True)
                self._set_label(2, "", selected=False)
        else:
            # 3+ items: normal rotary with prev / current / next.
            prev = (self._index - 1) % n
            nxt = (self._index + 1) % n
            self._set_label(0, self._items[prev][0], selected=False)
            self._set_label(1, self._items[self._index][0], selected=True)
            self._set_label(2, self._items[nxt][0], selected=False)

    @objc.python_method
    def _set_label(self, slot: int, text: str, selected: bool):
        """Render the slot's label as an NSAttributedString with the
        chosen weight, kern, alpha, alignment, and drop shadow."""
        if not text:
            self._labels[slot].setStringValue_("")
            return
        attr = _make_label_string(
            text=text.upper(),
            selected=selected,
            alignment=self._alignment,
        )
        self._labels[slot].setAttributedStringValue_(attr)

    @objc.python_method
    def _position_relative_to_anchor(self):
        """Place HUD horizontally on the opposite side from the screen
        midline (so it doesn't clip off-screen), and vertically centered
        on the anchor. Also set text alignment so the labels read
        toward the icon (right-aligned if HUD is to the icon's left,
        left-aligned if to its right) — gives a tight "hugs the icon"
        feel."""
        if self._anchor_panel is None:
            return
        af = self._anchor_panel.frame()
        screen = self._screen_for_point(af.origin) or NSScreen.mainScreen()
        sf = screen.visibleFrame()
        screen_mid_x = sf.origin.x + sf.size.width / 2

        if af.origin.x + af.size.width / 2 > screen_mid_x:
            # overlay on right half → HUD to its left, text right-aligned
            new_x = af.origin.x - HUD_WIDTH - EDGE_MARGIN
            self._alignment = NSTextAlignmentRight
        else:
            # overlay on left half → HUD to its right, text left-aligned
            new_x = af.origin.x + af.size.width + EDGE_MARGIN
            self._alignment = NSTextAlignmentLeft

        new_y = af.origin.y + (af.size.height - HUD_HEIGHT) / 2

        # Clamp to visible frame so we never spill off-screen
        new_x = max(sf.origin.x + 4,
                    min(new_x, sf.origin.x + sf.size.width - HUD_WIDTH - 4))
        new_y = max(sf.origin.y + 4,
                    min(new_y, sf.origin.y + sf.size.height - HUD_HEIGHT - 4))

        self._panel.setFrameOrigin_(NSPoint(new_x, new_y))

    @staticmethod
    @objc.python_method
    def _screen_for_point(point):
        from Foundation import NSPointInRect
        for screen in NSScreen.screens():
            if NSPointInRect(point, screen.frame()):
                return screen
        return None

    # ---- confirmation timer ----

    @objc.python_method
    def _restart_confirm_timer(self):
        self._cancel_confirm_timer()
        self._confirm_timer = (
            NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                CONFIRMATION_TIMEOUT, self, "fireSelected:", None, False
            )
        )

    @objc.python_method
    def _cancel_confirm_timer(self):
        if self._confirm_timer is not None:
            self._confirm_timer.invalidate()
            self._confirm_timer = None

    def fireSelected_(self, _timer):
        self._confirm_timer = None
        if not self._items:
            self._panel.orderOut_(None)
            return
        _, callback = self._items[self._index]
        self._panel.orderOut_(None)
        try:
            callback()
        except Exception as e:
            print(f"⚠️  Rotary action error: {type(e).__name__}: {e}")
