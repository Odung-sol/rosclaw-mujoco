#!/usr/bin/env python3
"""Render the README demo GIF.

Showcases the LQR controller's balance + disturbance rejection by running
the *real* control loop (`SegwaySimulation.step`) and injecting two
external impulses at scheduled times via `apply_disturbance`. Side-view
camera so pitch is clearly visible. Text overlays annotate what's
happening at each moment.

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
# 400×300 + 20 fps + 32-color palette = ~1.5 MB. Bigger settings drift past
# 3 MB which is too heavy to embed at the top of README.md.
WIDTH, HEIGHT = 400, 300
FPS = 20
DURATION_S = 11.0
RENDER_EVERY = max(1, int(round(1.0 / (FPS * SIM_DT))))   # sim steps per frame
OUTPUT_PATH = os.path.join("..", "docs", "demo.gif")
# Start perfectly upright. The narrative is: "operator keeps trying to knock
# it over, the LQR keeps catching it" — so the segway has to begin in a
# stable, hands-off posture before the first kick.
INITIAL_PITCH_DEG = 0.0


# ── Disturbance schedule ──────────────────────────────────────────────────
# Each entry: (start_s, force_N, force_duration_s, lqr_off_extra_s, label).
#
# `lqr_off_extra_s` is the *extra* time the LQR stays disabled after the
# external force ends — the segway keeps falling under gravity for that
# window, so the body visibly tips before the controller is allowed to
# react. This is what creates the "uh-oh, then save" moment.
#
# The forces escalate to tell a story: weak push → medium → strong reverse,
# each successively closer to the recovery limit. Empirically tuned to
# produce ~20°, ~22°, ~30° peak tilts; bigger numbers in the same
# off-window send the body past inversion.
DISTURBANCES = [
    (1.0,  40.0, 0.3, 0.2, "weak push →"),
    (4.0,  60.0, 0.3, 0.2, "medium push →"),
    (7.5, -80.0, 0.3, 0.2, "← STRONG push"),
]


# ── Camera (side view; tracks the body's x-position so the segway never
#    drifts out of frame after a kick — kicks generate >0.5 m of drift even
#    though the LQR holds θ near 0.) ────────────────────────────────────
def _make_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.2          # pulled back so a few metres of drift fit
    cam.azimuth = 100
    cam.elevation = -12
    cam.lookat[:] = [0.0, 0.0, 0.18]
    return cam


def _update_camera(cam, sim):
    """Track the segway's x-position. The kicks send it 10+ m off origin;
    if the camera can't keep up, half the GIF is the segway in the corner.

    Single proportional gain (0.25 per frame at 25 fps) — plenty of catch-up
    speed to keep the segway near the frame centre even at peak velocity.
    The floor pattern shifts a lot as a side-effect, which inflates GIF
    size; that's offset by lower color count + dimensions in the encoder.
    """
    target_x = float(sim.data.qpos[0])
    cam.lookat[0] += 0.25 * (target_x - cam.lookat[0])


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
    """Composite timestamp + state + (optional) kick label onto a frame."""
    img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Top strip: time + telemetry
    font_small = _load_font(14)
    info = f"t = {t:5.2f} s   theta = {theta_deg:+5.2f} deg   x = {x_pos:+5.3f} m"
    draw.rectangle([(0, 0), (img.size[0], 24)], fill=(0, 0, 0, 140))
    draw.text((8, 4), info, fill=(255, 255, 255, 255), font=font_small)

    # Centered banner
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
    """Return (label, kick_banner_active, lqr_off) for the given sim time.

    Phases:
      - kick window:   external force is on, LQR is off, banner shows the
                       kick label (red).
      - falling delay: force has ended but LQR still off; segway tips
                       under gravity. Banner stays red.
      - recovering:    LQR back on, fighting the residual tilt. Banner
                       turns calm grey.
      - balanced:      between scenarios.
    """
    for start, _, dur, off_extra, kick_label in DISTURBANCES:
        kick_end = start + dur
        lqr_on_at = kick_end + off_extra
        if start <= t < kick_end:
            return kick_label, True, True
        if kick_end <= t < lqr_on_at:
            return kick_label + "  (falling…)", True, True
        if lqr_on_at <= t < lqr_on_at + 1.2:
            return "LQR catches it", False, False
    return "balanced", False, False


def _demo_step(sim, lqr_off):
    """Demo-only step that bypasses the |θ| > 30° fail latch and (optionally)
    suppresses wheel torque so the body visibly tips.

    The production `SegwaySimulation.step()` flips `failed=True` and zeros
    the torques once theta exceeds 30° — a safety latch so unit tests can
    detect a runaway. For a demo we want the controller to keep trying.

    `lqr_off=True` zeros the wheel torque for this tick. The render loop
    holds it true during the kick window AND for an extra `lqr_off_extra_s`
    after the force ends; the body falls under gravity in that delay,
    producing the "uh-oh" moment. Once the delay expires, lqr_off=False
    and the controller catches it.
    """
    state = sim.ext.get_state(sim.data)
    if lqr_off:
        tL, tR = 0.0, 0.0
    else:
        tL, tR = sim.lqr.compute_torque(state)
    sim.data.ctrl[sim.L_act] = float(tL)
    sim.data.ctrl[sim.R_act] = float(tR)
    sim._apply_pending_disturbance()
    mujoco.mj_step(sim.model, sim.data)
    return state, tL, tR


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
        for k, (start, force, duration, _, _) in enumerate(DISTURBANCES):
            if not fired[k] and t >= start:
                sim.apply_disturbance(force_N=force, duration_s=duration)
                fired[k] = True

        # LQR is OFF during the kick AND the falling-delay window after it.
        lqr_off = any(
            start <= t < start + duration + off_extra
            for start, _, duration, off_extra, _ in DISTURBANCES
        )
        _demo_step(sim, lqr_off=lqr_off)

        if i % RENDER_EVERY == 0:
            _update_camera(cam, sim)  # smooth-tracks x so a kick doesn't slide segway off-frame
            renderer.update_scene(sim.data, camera=cam)
            pixels = renderer.render()
            frame = Image.fromarray(pixels)

            theta_deg = float(np.degrees(sim.ext.get_theta(sim.data)))
            x_pos = float(sim.data.qpos[0])
            label, banner_active, _ = _current_phase(t)
            frames.append(_draw_overlay(frame, t, theta_deg, x_pos, label, banner_active))

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
