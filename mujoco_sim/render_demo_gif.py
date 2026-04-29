#!/usr/bin/env python3
"""Render the README demo GIF.

Showcases the LQR's full balance + position regulation by running the real
control loop (`SegwaySimulation.step`) — LQR is on the whole time — and
injecting three escalating impulses via `apply_disturbance`. Each kick tips
the body 10°–30° before the controller drives it back upright AND back to
the starting position. Side-view camera; text overlays annotate each phase.

Run from `mujoco_sim/`:
    .venv/bin/python render_demo_gif.py
Output: ../docs/demo.gif
"""

import os
import sys

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

# Make this script work whether invoked from repo root or mujoco_sim/.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(THIS_DIR)
sys.path.insert(0, THIS_DIR)

from segway_sim import SegwaySimulation, SIM_DT  # noqa: E402


# ── Render config ─────────────────────────────────────────────────────────
# 400×300 + 20 fps + 32-color palette = ~1.3 MB. Bigger settings push past
# 3 MB which is too heavy to embed at the top of README.md.
WIDTH, HEIGHT = 400, 300
FPS = 20
DURATION_S = 11.0
RENDER_EVERY = max(1, int(round(1.0 / (FPS * SIM_DT))))   # sim steps per frame
OUTPUT_PATH = os.path.join("..", "docs", "demo.gif")
INITIAL_PITCH_DEG = 0.0   # start perfectly upright, hands-off, motionless


# ── Disturbance schedule ──────────────────────────────────────────────────
# (start_s, force_N, force_duration_s, label)
#
# LQR is on the *entire* run — no off-window trick. The kicks are strong
# enough that they overpower the controller for the 0.3 s force window
# (peak θ ≈ 10°/20°/27°), but the position-regulating LQR drives the
# segway back to the starting position within ~1 s. Forces tuned to keep
# the segway within ±1 m of origin so the camera doesn't need to track.
#
# 80 N is roughly the recovery limit with this K — bigger forces invert
# the body before the wheels can catch it.
DISTURBANCES = [
    (1.5,  30.0, 0.3, "weak push →"),
    (4.5,  50.0, 0.3, "medium push →"),
    (7.5, -80.0, 0.3, "← STRONG push"),
]


# ── Camera (fixed side view; segway returns to origin so no tracking) ─────
def _make_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.5
    cam.azimuth = 100
    cam.elevation = -12
    cam.lookat[:] = [0.0, 0.0, 0.18]
    return cam


# ── Font selection (graceful fallback if no TTF found on the host) ────────
def _load_font(size=18):
    candidates = [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _draw_overlay(img, t, theta_deg, x_pos, label, kick_active):
    """Composite timestamp + state + (optional) phase label onto a frame."""
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Top strip: time + telemetry
    font_small = _load_font(14)
    info = f"t = {t:5.2f} s   theta = {theta_deg:+5.2f} deg   x = {x_pos:+5.3f} m"
    draw.rectangle([(0, 0), (img.size[0], 24)], fill=(0, 0, 0, 140))
    draw.text((8, 4), info, fill=(255, 255, 255, 255), font=font_small)

    # Bottom centred banner
    if label:
        font_big = _load_font(22)
        bbox = draw.textbbox((0, 0), label, font=font_big)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (img.size[0] - tw) // 2
        y = img.size[1] - th - 24
        bg = (180, 40, 40, 200) if kick_active else (40, 40, 40, 180)
        draw.rectangle(
            [(x - 12, y - 8), (x + tw + 12, y + th + 8)],
            fill=bg,
        )
        draw.text((x, y - bbox[1]), label, fill=(255, 255, 255, 255), font=font_big)

    composed = Image.alpha_composite(img, overlay)
    return composed.convert("RGB")


def _current_phase(t):
    """Return (label, kick_banner_active) for the given sim time.

    Three phases per disturbance:
      - kick window:   the force is being applied; banner shows the kick
                       label in red.
      - recovering:    1 s after the force ends; banner shows
                       "LQR recovering" in calm grey.
      - balanced:      anything else.
    """
    for start, _, dur, kick_label in DISTURBANCES:
        if start <= t < start + dur:
            return kick_label, True
        if start + dur <= t < start + dur + 1.2:
            return "LQR recovering", False
    return "balanced", False


def main():
    sim = SegwaySimulation(use_ros2=False)
    sim.reset(pitch_deg=INITIAL_PITCH_DEG)

    renderer = mujoco.Renderer(sim.model, height=HEIGHT, width=WIDTH)
    cam = _make_camera()

    fired = [False] * len(DISTURBANCES)
    frames = []
    n_steps = int(DURATION_S / SIM_DT) + 1

    print(f"Rendering {DURATION_S}s @ {FPS} fps "
          f"(~{n_steps // RENDER_EVERY} frames @ {WIDTH}x{HEIGHT})")

    for i in range(n_steps):
        t = sim.data.time

        # Trigger any disturbance whose scheduled time has arrived.
        for k, (start, force, duration, _) in enumerate(DISTURBANCES):
            if not fired[k] and t >= start:
                sim.apply_disturbance(force_N=force, duration_s=duration)
                fired[k] = True

        # Plain `sim.step()` — LQR is on, position regulation included.
        sim.step()

        if i % RENDER_EVERY == 0:
            renderer.update_scene(sim.data, camera=cam)
            pixels = renderer.render()
            frame = Image.fromarray(pixels)

            theta_deg = float(np.degrees(sim.ext.get_theta(sim.data)))
            x_pos = float(sim.data.qpos[0])
            label, kick_active = _current_phase(t)
            frames.append(_draw_overlay(frame, t, theta_deg, x_pos, label, kick_active))

    sim.close()

    print(f"Captured {len(frames)} frames. Quantizing + writing GIF...")
    quantized = [
        f.quantize(colors=32, method=Image.Quantize.MEDIANCUT) for f in frames
    ]
    out = os.path.abspath(OUTPUT_PATH)
    quantized[0].save(
        out,
        save_all=True,
        append_images=quantized[1:],
        duration=int(round(1000 / FPS)),
        loop=0,
        optimize=True,
    )
    size_mb = os.path.getsize(out) / 1e6
    print(f"Wrote {out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
