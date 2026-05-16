"""Floating blackhole overlay for ara-agent.

Renders a small, draggable, always-on-top window with the animated
blackhole icon. Click the icon to bring up a context menu (Start /
Stop / Take Screenshot / Quit). Drag to reposition.

Architecture
------------
The overlay runs the macOS event loop on the main thread via
AppHelper.runEventLoop(). The voice agent runs on a background daemon
thread with its own asyncio loop, the same pattern menu_bar.py uses.
State transitions push a string into self._state (atomic in CPython);
an NSTimer on the main thread reads it ~16×/sec and swaps the NSImage
on the icon view.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

import objc
from AppKit import (
    NSApp,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSColor,
    NSEvent,
    NSImage,
    NSImageView,
    NSMenu,
    NSMenuItem,
    NSPanel,
    NSScreen,
    NSStatusWindowLevel,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSObject, NSPoint, NSRect, NSSize, NSTimer
from PyObjCTools import AppHelper

from ara_agent.hotkeys import GlobalHotkey
from ara_agent.icons import ensure_icons
from ara_agent.output_hud import OutputHUD
from ara_agent.rotary_hud import RotaryHUD
from ara_agent.voice_agent import AraAgent


OVERLAY_PT = 72          # icon display size in points
OVERLAY_MARGIN = 10      # padding around the icon in the window
WINDOW_SIZE = OVERLAY_PT + 2 * OVERLAY_MARGIN
TICK_SECONDS = 0.06      # ~16 fps animation
DRAG_THRESHOLD_PX_SQ = 9 # 3 px squared; minimum movement to count as drag


class IconView(NSImageView):
    """NSImageView that distinguishes click from drag.

    A short click triggers the click handler (shows the menu); a drag
    moves the parent window. The mousedown→mouseup screen distance is
    the discriminator.
    """

    def initWithFrame_(self, frame):
        self = objc.super(IconView, self).initWithFrame_(frame)
        if self is None:
            return None
        self._mouse_down_screen = None
        self._initial_window_origin = None
        self._is_drag = False
        self._click_handler = None
        return self

    def setClickHandler_(self, handler):
        self._click_handler = handler

    def mouseDown_(self, event):
        screen = NSEvent.mouseLocation()
        self._mouse_down_screen = (screen.x, screen.y)
        origin = self.window().frame().origin
        self._initial_window_origin = (origin.x, origin.y)
        self._is_drag = False

    def mouseDragged_(self, event):
        if self._mouse_down_screen is None:
            return
        screen = NSEvent.mouseLocation()
        dx = screen.x - self._mouse_down_screen[0]
        dy = screen.y - self._mouse_down_screen[1]
        if (dx * dx + dy * dy) > DRAG_THRESHOLD_PX_SQ:
            self._is_drag = True
        if self._is_drag:
            self.window().setFrameOrigin_(NSPoint(
                self._initial_window_origin[0] + dx,
                self._initial_window_origin[1] + dy,
            ))

    def mouseUp_(self, event):
        if not self._is_drag and self._click_handler is not None:
            self._click_handler(event, self)
        self._mouse_down_screen = None
        self._initial_window_origin = None
        self._is_drag = False

    def rightMouseUp_(self, event):
        if self._click_handler is not None:
            self._click_handler(event, self)


class OverlayController(NSObject):
    """App-level controller. Owns the panel, the agent thread, the menu."""

    def initWithFrames_(self, frame_paths):
        self = objc.super(OverlayController, self).init()
        if self is None:
            return None

        # Pre-load NSImages at overlay size.
        self._frames: Dict[str, List["NSImage"]] = {}
        for state, paths in frame_paths.items():
            imgs = []
            for p in paths:
                img = NSImage.alloc().initWithContentsOfFile_(p)
                img.setSize_(NSSize(OVERLAY_PT, OVERLAY_PT))
                imgs.append(img)
            self._frames[state] = imgs

        self._state = "idle"
        self._frame_idx = 0
        self._agent_thread: Optional[threading.Thread] = None
        self._agent_loop: Optional[asyncio.AbstractEventLoop] = None
        self._agent: Optional[AraAgent] = None
        # Set True while a screenshot capture is mid-flight (either OCR
        # or vision). Prevents hotkey spam from spawning a second
        # `screencapture -i` (which macOS rejects with "cannot run two
        # interactive screen captures at a time").
        self._capture_in_progress = False

        # Panel positioned near the top-right of the main screen.
        screen_frame = NSScreen.mainScreen().visibleFrame()
        win_x = screen_frame.origin.x + screen_frame.size.width - WINDOW_SIZE - 24
        win_y = screen_frame.origin.y + screen_frame.size.height - WINDOW_SIZE - 24
        self._panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSRect(NSPoint(win_x, win_y), NSSize(WINDOW_SIZE, WINDOW_SIZE)),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered,
            False,
        )
        self._panel.setLevel_(NSStatusWindowLevel)
        self._panel.setOpaque_(False)
        self._panel.setBackgroundColor_(NSColor.clearColor())
        self._panel.setHasShadow_(False)
        self._panel.setIgnoresMouseEvents_(False)
        # Float above fullscreen apps and across all spaces.
        try:
            from AppKit import (
                NSWindowCollectionBehaviorCanJoinAllSpaces,
                NSWindowCollectionBehaviorFullScreenAuxiliary,
            )
            self._panel.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces
                | NSWindowCollectionBehaviorFullScreenAuxiliary
            )
        except Exception:
            pass

        # Icon view
        icon_view = IconView.alloc().initWithFrame_(
            NSRect(NSPoint(OVERLAY_MARGIN, OVERLAY_MARGIN),
                   NSSize(OVERLAY_PT, OVERLAY_PT))
        )
        # NSImageScaleProportionallyUpOrDown = 1
        icon_view.setImageScaling_(1)
        first = self._frames["idle"][0]
        # In a floating window there's no surrounding control to provide
        # tint context, so don't use template mode here — render the icon
        # in pure white as drawn.
        first.setTemplate_(False)
        icon_view.setImage_(first)
        icon_view.setClickHandler_(self._handle_click)
        self._icon_view = icon_view
        self._panel.contentView().addSubview_(icon_view)
        self._panel.orderFront_(None)

        # Animation timer (main thread).
        self._timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            TICK_SECONDS, self, "tick:", None, True
        )

        # Rotary HUD for hotkey-driven action selection.
        self._rotary = RotaryHUD.alloc().init()

        # Output HUD — persistent activity panel at bottom-middle of the
        # overlay's screen. Auto-shows on listening, hides on stop.
        self._output_hud = OutputHUD.alloc().initWithAnchor_(self._panel)

        # Global hotkey: Cmd+Shift+A cycles the rotary HUD. Implemented
        # via Carbon's RegisterEventHotKey — no macOS permission needed.
        self._hotkey = GlobalHotkey(on_press=self._on_hotkey)
        installed = self._hotkey.install()
        if installed:
            print("🔑 Hotkey active: ⌘⇧A triggers Capture when listening.")
        else:
            print("⚠️  Hotkey registration failed — check console for details.")

        return self

    # ---- main-thread animation ----

    def tick_(self, _timer):
        frames = self._frames.get(self._state) or self._frames["idle"]
        img = frames[self._frame_idx % len(frames)]
        img.setTemplate_(False)
        self._icon_view.setImage_(img)
        self._frame_idx += 1

    # ---- agent state callback (asyncio thread) ----

    def _on_agent_state(self, state):
        if state != self._state:
            self._state = state
            self._frame_idx = 0
        # Always forward to the HUD — even if same state, it's harmless,
        # and the HUD may have been freshly shown and missed the last set.
        if self._output_hud is not None:
            self._output_hud.set_state(state)

    def _on_agent_event(self, kind, text):
        if self._output_hud is not None:
            self._output_hud.log_event(kind, text)

    # ---- hotkey ----

    def _on_hotkey(self):
        """Cmd+Shift+A pressed. Trigger a vision capture when the agent
        is running. When not running, the hotkey is a no-op — Start /
        Stop Listening are only available via the overlay's click menu,
        so they can't be triggered by accident from the rotary.

        Only one capture mode now (vision); the old Capture Text / Image
        rotary was collapsed after the OCR-injection path was removed.
        We still route through the rotary to keep the visual feedback
        and the dismiss-on-second-press affordance."""
        if not (self._agent_thread and self._agent_thread.is_alive()):
            return
        if self._capture_in_progress:
            # A screencapture UI is open or its result is being processed.
            # Ignore further hotkey presses until it completes.
            return
        items = [
            ("Capture", lambda: self.capture_(None)),
        ]
        if self._rotary._panel.isVisible():
            self._rotary.advance()
        else:
            self._rotary.present(items, self._panel)

    # ---- click → context menu, OR cancel if speaking ----

    def _handle_click(self, event, view):
        # Click during Ara's speech = barge-in (cancel current response)
        # rather than opening the menu. Lets the user interrupt without
        # navigating menu items.
        if self._state == "speaking":
            loop = self._agent_loop
            agent = self._agent
            if loop and agent and loop.is_running():
                asyncio.run_coroutine_threadsafe(agent.cancel_response(), loop)
            return

        menu = NSMenu.alloc().init()
        running = self._agent_thread is not None and self._agent_thread.is_alive()

        if not running:
            start = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Start Listening", "startAgent:", ""
            )
            start.setTarget_(self)
            menu.addItem_(start)
        else:
            shot = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Capture…", "capture:", ""
            )
            shot.setTarget_(self)
            menu.addItem_(shot)

            menu.addItem_(NSMenuItem.separatorItem())

            stop = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
                "Stop Listening", "stopAgent:", ""
            )
            stop.setTarget_(self)
            menu.addItem_(stop)

        menu.addItem_(NSMenuItem.separatorItem())

        quit_it = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit Ara", "quitApp:", ""
        )
        quit_it.setTarget_(self)
        menu.addItem_(quit_it)

        NSMenu.popUpContextMenu_withEvent_forView_(menu, event, view)

    # ---- menu actions (ObjC selectors) ----

    def startAgent_(self, _sender):
        if self._agent_thread and self._agent_thread.is_alive():
            return
        self._on_agent_state("listening")
        if self._output_hud is not None:
            self._output_hud.show()
        self._agent_thread = threading.Thread(
            target=self._run_agent, daemon=True
        )
        self._agent_thread.start()

    def stopAgent_(self, _sender):
        self._shutdown_agent()

    def capture_(self, _sender):
        """Single capture path: take a screenshot and send it to xAI's
        vision model for a one-sentence semantic description, which is
        then injected into the realtime conversation as context.

        This replaced two earlier modes (Apple-OCR text and grok vision).
        The OCR-text path was deleted after consistently stalling xAI's
        Realtime backend on text-input turns. For literal text reading,
        the model uses the read_screen_region_text tool instead."""
        loop = self._agent_loop
        agent = self._agent
        if not (loop and agent and loop.is_running()):
            print("Start listening before capturing.")
            return
        if self._capture_in_progress:
            return
        self._capture_in_progress = True
        future = asyncio.run_coroutine_threadsafe(
            agent.request_screenshot_description(), loop
        )
        future.add_done_callback(lambda _f: self._clear_capture_lock())

    def _clear_capture_lock(self):
        """Called from the asyncio thread when a capture coroutine
        finishes (successful capture, cancellation, or error). Bool
        assignment is atomic in CPython so cross-thread set is safe."""
        self._capture_in_progress = False

    def quitApp_(self, _sender):
        self._shutdown_agent()
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None
        if self._hotkey is not None:
            self._hotkey.uninstall()
            self._hotkey = None
        if self._rotary is not None:
            self._rotary.hide()
        NSApp.terminate_(None)

    # ---- agent lifecycle (background thread) ----

    def _run_agent(self):
        log.info("agent thread starting")
        self._agent_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._agent_loop)
        self._agent = AraAgent(
            state_callback=self._on_agent_state,
            event_callback=self._on_agent_event,
        )
        try:
            self._agent_loop.run_until_complete(self._agent.run())
            log.info("agent thread completed normally")
        except Exception as e:
            log.exception("agent thread raised — top-level catch")
            print(f"Voice agent error: {type(e).__name__}: {e}")
        finally:
            self._on_agent_state("idle")
            try:
                self._agent_loop.close()
            except Exception:
                log.exception("error closing agent loop")
            self._agent_loop = None
            self._agent = None

    def _shutdown_agent(self):
        loop = self._agent_loop
        agent = self._agent
        if loop and agent and loop.is_running():
            asyncio.run_coroutine_threadsafe(agent.stop(), loop)
        self._on_agent_state("idle")
        if self._output_hud is not None:
            self._output_hud.hide()


def run_overlay():
    """Entry point. Creates NSApplication, builds the overlay, runs the
    event loop. Blocks until the user quits."""
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    icon_paths = ensure_icons()
    # Keep the controller pinned so its ObjC references don't get GC'd.
    global _controller
    _controller = OverlayController.alloc().initWithFrames_(icon_paths)

    AppHelper.runEventLoop()


_controller = None


if __name__ == "__main__":
    run_overlay()
