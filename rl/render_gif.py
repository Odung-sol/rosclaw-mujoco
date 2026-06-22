#!/usr/bin/env python3
"""Render an RL-vs-LQR balancing comparison GIF (offscreen MuJoCo).

Run (repo root, MuJoCo env):  python -m rl.render_gif
Output: docs/rl_vs_lqr.gif — ~5 s, side-by-side (RL policy | LQR), both released
from a +2° forward tilt. RL barely moves; the LQR lurches before recovering.

Reuses the trained policy in rl/models/ and the same SegwaySimulation + camera
setup as mujoco_sim/render_demo_gif.py.
"""

import os
import sys
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "mujoco_sim")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from segway_sim import SegwaySimulation, SIM_DT  # noqa: E402
from rl.policy_adapter import RLPolicy  # noqa: E402

MODELS = _REPO_ROOT / "rl" / "models"
DOCS = _REPO_ROOT / "docs"
PANEL_W, PANEL_H = 300, 300
GAP = 4
FPS = 20
DURATION_S = 5.0
INITIAL_PITCH_DEG = 2.0
RENDER_EVERY = max(1, round(1.0 / (FPS * SIM_DT)))  # mj steps per rendered frame
COLORS = 32


def _make_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.5
    cam.azimuth = 100
    cam.elevation = -12
    cam.lookat[:] = [0.0, 0.0, 0.18]
    return cam


def _font(size):
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _label_panel(pixels, title, theta_deg, fell):
    img = Image.fromarray(pixels).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([(0, 0), (img.size[0], 26)], fill=(30, 30, 30))
    d.text((8, 5), title, fill=(255, 255, 255), font=_font(16))
    color = (235, 90, 60) if fell else (235, 235, 235)
    txt = f"theta = {theta_deg:+5.1f} deg" + ("   FELL" if fell else "")
    d.rectangle([(0, img.size[1] - 26), (img.size[0], img.size[1])], fill=(30, 30, 30))
    d.text((8, img.size[1] - 22), txt, fill=color, font=_font(15))
    return img


def main():
    DOCS.mkdir(exist_ok=True)
    # segway.xml loads via a cwd-relative path (like render_demo_gif.py). DOCS
    # and MODELS are absolute, so the chdir doesn't affect I/O paths.
    os.chdir(_REPO_ROOT / "mujoco_sim")
    rl_policy = RLPolicy.from_files(MODELS / "ppo_segway.zip", MODELS / "vecnormalize.pkl")
    sim_rl = SegwaySimulation(use_ros2=False, controller=rl_policy)
    sim_lqr = SegwaySimulation(use_ros2=False)  # default controller = SegwayLQR
    sim_rl.reset(pitch_deg=INITIAL_PITCH_DEG)
    sim_lqr.reset(pitch_deg=INITIAL_PITCH_DEG)

    r_rl = mujoco.Renderer(sim_rl.model, height=PANEL_H, width=PANEL_W)
    r_lqr = mujoco.Renderer(sim_lqr.model, height=PANEL_H, width=PANEL_W)
    cam = _make_camera()

    n_steps = int(DURATION_S / SIM_DT) + 1
    frames = []
    fell_rl = False
    fell_lqr = False
    print(f"Rendering {DURATION_S}s @ {FPS} fps "
          f"(~{n_steps // RENDER_EVERY} frames, {PANEL_W * 2 + GAP}x{PANEL_H})")

    for i in range(n_steps):
        sim_rl.step()
        sim_lqr.step()
        fell_rl = fell_rl or sim_rl.failed
        fell_lqr = fell_lqr or sim_lqr.failed
        if i % RENDER_EVERY == 0:
            r_rl.update_scene(sim_rl.data, camera=cam)
            r_lqr.update_scene(sim_lqr.data, camera=cam)
            th_rl = float(np.degrees(sim_rl.ext.get_theta(sim_rl.data)))
            th_lqr = float(np.degrees(sim_lqr.ext.get_theta(sim_lqr.data)))
            left = _label_panel(r_rl.render(), "RL (PPO)", th_rl, fell_rl)
            right = _label_panel(r_lqr.render(), "LQR", th_lqr, fell_lqr)
            combined = Image.new("RGB", (PANEL_W * 2 + GAP, PANEL_H), (255, 255, 255))
            combined.paste(left, (0, 0))
            combined.paste(right, (PANEL_W + GAP, 0))
            frames.append(combined)

    sim_rl.close()
    sim_lqr.close()

    print(f"Captured {len(frames)} frames. Quantizing + writing GIF...")
    quant = [f.quantize(colors=COLORS, method=Image.Quantize.MEDIANCUT) for f in frames]
    out = DOCS / "rl_vs_lqr.gif"
    quant[0].save(out, save_all=True, append_images=quant[1:],
                  duration=int(round(1000 / FPS)), loop=0, optimize=True)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
