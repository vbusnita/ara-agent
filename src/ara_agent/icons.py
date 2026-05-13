"""Procedurally rendered Grok-blackhole menu bar icons.

Every frame is composed from glowing rings and halos drawn at 4× the target
resolution and downsampled with LANCZOS, so the result reads as smooth on
retina menu bars rather than aliased pixel art. Frames are cached to disk
the first time the app launches and reused thereafter.

Design notes
------------
- All four states share a "luminous ring around a dark core" silhouette so
  identity is consistent across transitions.
- The center stays transparent: on a dark menu bar that reads as the event
  horizon of a black hole; on a light bar the ring still reads as a sphere.
- Each state has its own hue so the user can tell the agent's mood at a
  glance without watching the animation play.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFilter

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # Pillow <9.1
    LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]


# Bump this when changing the look so stale frames get regenerated.
ICON_VERSION = "v4"

# Source PNG is @2x so retina displays render crisply at the display size.
# Drawn as a single tonal layer (pure white + varying alpha); the menu bar
# loads it as an NSImage template so macOS auto-tints to text color
# (white in dark mode, black in light mode — native menu bar behavior).
DISPLAY_PT = 24
SIZE = DISPLAY_PT * 2          # 48 px on disk
SUPERSAMPLE = 4
WORK = SIZE * SUPERSAMPLE       # render at 192 px, downsample to 48

# Single-tone palette: everything is white, only alpha varies.
W = (255, 255, 255)

CACHE_DIR = (
    Path.home() / "Library" / "Caches" / "ara-agent" / f"icons-{ICON_VERSION}"
)


# ---- primitive drawing helpers (all operate at WORK resolution) ----

def _ring(radius: float, thickness: float, color: Tuple[int, int, int, int],
          blur: float = 0.0) -> Image.Image:
    """A glowing ring of the given color, optionally blurred for halo effect."""
    img = Image.new("RGBA", (WORK, WORK), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = WORK / 2
    d.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        outline=color,
        width=max(1, int(round(thickness))),
    )
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    return img


def _disk(cx: float, cy: float, radius: float,
          color: Tuple[int, int, int, int], blur: float = 0.0) -> Image.Image:
    """A filled disk, optionally blurred to a soft blob."""
    img = Image.new("RGBA", (WORK, WORK), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=color,
    )
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    return img


def _stack(*layers: Image.Image) -> Image.Image:
    """Alpha-composite bottom-to-top."""
    base = layers[0]
    for layer in layers[1:]:
        base = Image.alpha_composite(base, layer)
    return base


# NOTE: previous versions added a dark "void" disk at the center for the
# blackhole feel on light backgrounds. In template mode the system tints
# every drawn pixel uniformly, so a void disk would *add* tinted mass at
# the center rather than darkening it. We rely on the transparent center
# showing the menu bar background instead — that's what reads as "void".


# ---- per-state frame generators (t in [0, 1) over a full cycle) ----

def _idle(t: float) -> Image.Image:
    """Quiet ring with a slow breath and a lensing fleck drifting around
    the rim. Lower overall intensity reads as 'at rest'."""
    breathe = 0.5 + 0.5 * math.sin(2 * math.pi * t)
    r = WORK * (0.28 + 0.012 * breathe)
    cx = cy = WORK / 2

    halo = _ring(r + WORK * 0.08, WORK * 0.10,
                 (*W, int(95 + 30 * breathe)),
                 blur=WORK * 0.08)
    edge = _ring(r, WORK * 0.07, (*W, 215))

    angle = 2 * math.pi * t
    fx = cx + r * math.cos(angle)
    fy = cy + r * math.sin(angle)
    fleck = _disk(fx, fy, WORK * 0.07, (*W, 230), blur=WORK * 0.045)

    return _stack(halo, edge, fleck)


def _listening(t: float) -> Image.Image:
    """Strong pulse — the ring breathes wider and the halo brightens
    noticeably, signaling active attention."""
    pulse = 0.5 + 0.5 * math.sin(2 * math.pi * t)
    r = WORK * (0.26 + 0.05 * pulse)

    halo = _ring(r + WORK * 0.10, WORK * 0.12,
                 (*W, int(135 + 90 * pulse)),
                 blur=WORK * 0.09)
    inner_halo = _ring(r + WORK * 0.04, WORK * 0.05,
                       (*W, int(150 + 70 * pulse)),
                       blur=WORK * 0.035)
    edge = _ring(r, WORK * 0.075, (*W, 250))
    return _stack(halo, inner_halo, edge)


def _thinking(t: float) -> Image.Image:
    """Medium ring with a bright particle orbiting it — the rotation is
    the strongest cue here since hue is gone."""
    r = WORK * 0.28
    cx = cy = WORK / 2

    ring_halo = _ring(r + WORK * 0.05, WORK * 0.06,
                      (*W, 110), blur=WORK * 0.045)
    ring = _ring(r, WORK * 0.06, (*W, 200))

    # Bright leading particle + 4-segment trailing comet tail.
    trail: List[Image.Image] = []
    for i in range(5):
        trail_t = t - i * 0.035
        a = 2 * math.pi * trail_t
        sx = cx + r * math.cos(a)
        sy = cy + r * math.sin(a)
        alpha = int(255 * (1 - i / 5) ** 1.3)
        spot_r = WORK * (0.075 - i * 0.008)
        trail.append(_disk(sx, sy, spot_r * 2.5,
                           (*W, alpha // 3), blur=WORK * 0.05))
        trail.append(_disk(sx, sy, spot_r, (*W, alpha)))

    return _stack(ring_halo, ring, *trail)


def _speaking(t: float) -> Image.Image:
    """Bright core with rings radiating outward — highest intensity of the
    four states, reads as energetic broadcast."""
    base_r = WORK * 0.15
    layers: List[Image.Image] = []

    # Three concentric radiating rings at staggered phases.
    for i in range(3):
        phase = (t + i / 3) % 1.0
        r = base_r + phase * (WORK * 0.27)
        alpha = int(255 * (1 - phase) ** 1.1)
        if alpha <= 6:
            continue
        layers.append(_ring(r, WORK * 0.07,
                            (*W, alpha // 2),
                            blur=WORK * 0.045))
        layers.append(_ring(r, WORK * 0.055,
                            (*W, alpha),
                            blur=WORK * 0.012))

    # Bright energetic core.
    core_glow = _ring(base_r, WORK * 0.10, (*W, 245), blur=WORK * 0.06)
    core_edge = _ring(base_r, WORK * 0.07, (*W, 255))
    layers.append(core_glow)
    layers.append(core_edge)

    return _stack(*layers)


STATE_GENERATORS: Dict[str, Tuple[Callable[[float], Image.Image], int]] = {
    "idle":      (_idle,      16),
    "listening": (_listening, 16),
    "thinking":  (_thinking,  24),
    "speaking":  (_speaking,  12),
}


def ensure_icons() -> Dict[str, List[str]]:
    """Generate (once) and return paths to every frame, grouped by state."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    first = CACHE_DIR / "idle_00.png"
    if not first.exists():
        print("✨ Rendering Ara icon frames (first launch, ~2s)...")

    out: Dict[str, List[str]] = {}
    for state, (fn, count) in STATE_GENERATORS.items():
        paths: List[str] = []
        for i in range(count):
            path = CACHE_DIR / f"{state}_{i:02d}.png"
            if not path.exists():
                large = fn(i / count)
                final = large.resize((SIZE, SIZE), LANCZOS)
                final.save(path, optimize=True)
            paths.append(str(path))
        out[state] = paths

    return out


if __name__ == "__main__":
    paths = ensure_icons()
    for state, ps in paths.items():
        print(f"{state}: {len(ps)} frames")
    print(f"Cache: {CACHE_DIR}")
