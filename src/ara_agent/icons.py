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
ICON_VERSION = "v6"

# Source PNGs are sized to look crisp at both UIs that consume them:
#   menu bar  — displayed at ~24 pt
#   overlay   — displayed at ~72 pt
# 144 px source gives retina-sharp output at the overlay's @2x resolution
# (72 pt × 2 = 144 px) and clean downsample at the menu bar's smaller
# size. Consumers set their own per-image display size via setSize_().
SIZE = 144
SUPERSAMPLE = 3
WORK = SIZE * SUPERSAMPLE       # render at 432 px, downsample to 144

# Single-tone palette: bright matter is white (alpha varies), shadow is
# near-black. The dark shadow is what sells the "event horizon" feel —
# without it the icon reads as a ring; with it, it reads as something
# the light is bending around.
W = (255, 255, 255)
SHADOW = (8, 10, 16)

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


def _arc(radius: float, thickness: float, start_deg: float, end_deg: float,
         color, blur: float = 0.0) -> Image.Image:
    """A glowing arc — used for asymmetric lensing brightening on one
    side of the photon ring, which is what gives real black-hole images
    their lopsided 'bright crescent' look (relativistic beaming).
    """
    img = Image.new("RGBA", (WORK, WORK), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = WORK / 2
    d.arc(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        start=start_deg, end=end_deg,
        fill=color,
        width=max(1, int(round(thickness))),
    )
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    return img


def _shadow_disk(radius: float, alpha: int) -> Image.Image:
    """A dark, soft-edged disk at the center — the event horizon's shadow.
    On dark backgrounds it blends invisibly; on light backgrounds it
    reads as the void that the bright matter is orbiting."""
    img = Image.new("RGBA", (WORK, WORK), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = WORK / 2
    d.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=(*SHADOW, alpha),
    )
    return img.filter(ImageFilter.GaussianBlur(radius=WORK * 0.035))


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
    """Event horizon at rest. Dark central shadow, a faint full photon
    ring, a brighter lensing arc that drifts slowly around the rim, and
    a layered halo (wide-soft outer + tight-bright inner) that breathes."""
    breathe = 0.5 + 0.5 * math.sin(2 * math.pi * t)
    r = WORK * (0.30 + 0.013 * breathe)
    lensing_deg = (360 * t) % 360  # one full revolution per cycle

    layers = []
    # Wide soft outer halo
    layers.append(_ring(r + WORK * 0.12, WORK * 0.10,
                        (*W, int(60 + 25 * breathe)),
                        blur=WORK * 0.10))
    # Tighter inner halo
    layers.append(_ring(r + WORK * 0.05, WORK * 0.06,
                        (*W, int(110 + 35 * breathe)),
                        blur=WORK * 0.05))
    # Event horizon shadow (the void)
    layers.append(_shadow_disk(r * 0.85, alpha=130))
    # Faint full photon ring (the "back" side, dimmer)
    layers.append(_ring(r, WORK * 0.025, (*W, 130)))
    # Bright lensing arc (the "front" side, brightened by gravitational
    # light bending — slowly rotates so even idle feels alive)
    layers.append(_arc(r, WORK * 0.04,
                       lensing_deg - 75, lensing_deg + 75,
                       (*W, 235),
                       blur=WORK * 0.01))
    return _stack(*layers)


def _listening(t: float) -> Image.Image:
    """Active attention. Same event-horizon construction as idle, but the
    ring pulses wider and brighter, the halo intensifies dramatically,
    and the lensing arc rotates faster."""
    pulse = 0.5 + 0.5 * math.sin(2 * math.pi * t)
    r = WORK * (0.28 + 0.04 * pulse)
    lensing_deg = (360 * t * 1.5) % 360

    layers = []
    # Outer wide halo — visibly pulsing
    layers.append(_ring(r + WORK * 0.14, WORK * 0.13,
                        (*W, int(110 + 110 * pulse)),
                        blur=WORK * 0.11))
    # Bright inner halo
    layers.append(_ring(r + WORK * 0.05, WORK * 0.07,
                        (*W, int(160 + 80 * pulse)),
                        blur=WORK * 0.05))
    # Shadow at center
    layers.append(_shadow_disk(r * 0.85, alpha=120))
    # Photon ring full circumference
    layers.append(_ring(r, WORK * 0.028, (*W, 175)))
    # Bright lensing arc, more prominent than idle
    layers.append(_arc(r, WORK * 0.05,
                       lensing_deg - 80, lensing_deg + 80,
                       (*W, 250),
                       blur=WORK * 0.012))
    return _stack(*layers)


def _thinking(t: float) -> Image.Image:
    """Particle orbits the photon ring with a comet trail. Static halo
    keeps focus on the rotational motion."""
    r = WORK * 0.30
    cx = cy = WORK / 2

    layers = []
    layers.append(_ring(r + WORK * 0.10, WORK * 0.09,
                        (*W, 75), blur=WORK * 0.08))
    layers.append(_ring(r + WORK * 0.04, WORK * 0.05,
                        (*W, 130), blur=WORK * 0.04))
    layers.append(_shadow_disk(r * 0.85, alpha=120))
    layers.append(_ring(r, WORK * 0.025, (*W, 165)))

    # Orbiting bright particle plus 5-segment trailing comet tail.
    for i in range(6):
        trail_t = t - i * 0.028
        a = 2 * math.pi * trail_t
        sx = cx + r * math.cos(a)
        sy = cy + r * math.sin(a)
        alpha = int(255 * (1 - i / 6) ** 1.3)
        spot_r = WORK * (0.07 - i * 0.008)
        layers.append(_disk(sx, sy, spot_r * 2.5,
                            (*W, alpha // 3), blur=WORK * 0.05))
        layers.append(_disk(sx, sy, spot_r, (*W, alpha)))
    return _stack(*layers)


def _speaking(t: float) -> Image.Image:
    """Bright core with concentric rings radiating outward. Highest
    intensity of the four states — reads as energetic broadcast."""
    base_r = WORK * 0.17
    layers: List[Image.Image] = []
    layers.append(_shadow_disk(base_r * 0.65, alpha=80))

    # Three concentric radiating rings at staggered phases.
    for i in range(3):
        phase = (t + i / 3) % 1.0
        r = base_r + phase * (WORK * 0.27)
        alpha = int(255 * (1 - phase) ** 1.0)
        if alpha < 6:
            continue
        layers.append(_ring(r, WORK * 0.08,
                            (*W, alpha // 2),
                            blur=WORK * 0.045))
        layers.append(_ring(r, WORK * 0.055,
                            (*W, alpha),
                            blur=WORK * 0.012))

    # Bright energetic core
    layers.append(_ring(base_r, WORK * 0.12, (*W, 245), blur=WORK * 0.06))
    layers.append(_ring(base_r, WORK * 0.07, (*W, 255)))
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
