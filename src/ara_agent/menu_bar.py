"""macOS menu bar UI for ara-agent.

Renders a procedurally generated blackhole-style icon that animates per
state. NSImage frames are pre-loaded once and swapped directly on the
NSStatusItem button so the per-tick cost is just a pointer assignment,
not a file load.

Threading model: rumps owns the main thread (NSApplication is main-thread
only on macOS). The voice agent runs on a daemon thread with its own
asyncio loop. State transitions push a string into self._state (atomic
in CPython); a main-thread rumps.Timer reads it ~16×/sec and updates
the icon — no cross-thread UI mutation.
"""

import asyncio
import threading
from typing import Dict, List, Optional

import rumps
from AppKit import NSImage

from ara_agent.icons import ensure_icons
from ara_agent.voice_agent import AraAgent


TICK_SECONDS = 0.06  # ~16 fps
MENU_BAR_PT = 24     # icon display size in the menu bar


class AraMenuBar(rumps.App):
    def __init__(self):
        icon_paths = ensure_icons()

        super().__init__(
            name="Ara",
            icon=icon_paths["idle"][0],
            template=True,  # macOS auto-tints to text color (native feel)
            quit_button=None,
        )
        self.menu = ["Start Listening", "Stop", None, "Quit"]

        # Pre-load every frame so per-tick swaps are a pointer assignment.
        self._frames: Dict[str, List["NSImage"]] = {
            state: [self._load_image(p) for p in paths]
            for state, paths in icon_paths.items()
        }

        self._state = "idle"
        self._frame = 0

        self._agent_thread: Optional[threading.Thread] = None
        self._agent_loop: Optional[asyncio.AbstractEventLoop] = None
        self._agent: Optional[AraAgent] = None

        self._timer = rumps.Timer(self._tick, TICK_SECONDS)
        self._timer.start()

    # -- frame loading --

    @staticmethod
    def _load_image(path: str) -> "NSImage":
        img = NSImage.alloc().initWithContentsOfFile_(path)
        # Source PNGs are @2x; tell AppKit the on-screen size in points so
        # retina displays sample crisply and non-retina downsample cleanly.
        img.setSize_((MENU_BAR_PT, MENU_BAR_PT))
        # Each frame is also a template — rumps sets this on the App's
        # initial icon, but we bypass that path when swapping frames
        # via setImage_, so set it explicitly per image.
        img.setTemplate_(True)
        return img

    # -- main-thread render loop --

    def _tick(self, _sender):
        frames = self._frames.get(self._state) or self._frames["idle"]
        ns_image = frames[self._frame % len(frames)]
        try:
            status_item = self._nsapp.nsstatusitem
            button = status_item.button()
            if button is not None:
                button.setImage_(ns_image)
            else:  # very old macOS fallback
                status_item.setImage_(ns_image)
        except Exception:
            pass
        self._frame += 1

    def _on_state(self, state: str) -> None:
        # Called from the asyncio thread; main thread reads on next tick.
        if state != self._state:
            self._state = state
            self._frame = 0  # restart cycle on transition

    # -- menu actions --

    @rumps.clicked("Start Listening")
    def _start(self, _):
        if self._agent_thread and self._agent_thread.is_alive():
            return
        self._on_state("listening")
        self._agent_thread = threading.Thread(target=self._run_agent, daemon=True)
        self._agent_thread.start()

    @rumps.clicked("Stop")
    def _stop(self, _):
        self._shutdown_agent()

    @rumps.clicked("Quit")
    def _quit(self, _):
        self._shutdown_agent()
        rumps.quit_application()

    # -- agent lifecycle (background thread) --

    def _run_agent(self):
        self._agent_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._agent_loop)
        self._agent = AraAgent(state_callback=self._on_state)
        try:
            self._agent_loop.run_until_complete(self._agent.run())
        except Exception as e:
            # run() already handles clean WebSocket closures silently, so
            # anything caught here is an actual unexpected error.
            print(f"Voice agent error: {type(e).__name__}: {e}")
        finally:
            self._on_state("idle")
            try:
                self._agent_loop.close()
            except Exception:
                pass
            self._agent_loop = None
            self._agent = None

    def _shutdown_agent(self):
        loop = self._agent_loop
        agent = self._agent
        if loop and agent and loop.is_running():
            asyncio.run_coroutine_threadsafe(agent.stop(), loop)
        self._on_state("idle")


def run_menu_bar():
    AraMenuBar().run()


if __name__ == "__main__":
    run_menu_bar()
