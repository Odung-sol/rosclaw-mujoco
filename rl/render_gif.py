#!/usr/bin/env python3
"""Render a 3-way balancing comparison GIF (offscreen MuJoCo).

Run (repo root, MuJoCo env):  python -m rl.render_gif
Output: docs/rl_vs_lqr.gif — ~5 s, three panels released from a +2° tilt:
    RL (PPO) | LQR (fixed gains) | LQR (CARE-tuned)
The fixed-gain LQR (tuned for impulse-from-upright) overshoots from a tilt,
while the RL policy and a properly-tuned CARE LQR both stay nearly upright.

Reuses the trained policy in rl/models/ and the same camera as
mujoco_sim/render_demo_gif.py.
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
from rl.lqr_tuning import make_care_lqr  # noqa: E402

MODELS = _REPO_ROOT / "rl" / "models"
DOCS = _REPO_ROOT / "docs"
PANEL_W, PANEL_H = 280, 300
GAP = 4
FPS = 20
DURATION_S = 5.0
INITIAL_PITCH_DEG = 2.0
RENDER_EVERY = max(1, round(1.0 / (FPS * SIM_DT)))
COLORS = 48


def _make_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.5
    cam.azimuth = 100
    cam.elevation = -12
    cam.lookat[:] = [0.0, 0.0, 0.18]
    return cam


def _font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/SFNSMono.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _label_panel(pixels, title, theta_deg, fell):
    img = Image.fromarray(pixels).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([(0, 0), (img.size[0], 24)], fill=(30, 30, 30))
    d.text((6, 4), title, fill=(255, 255, 255), font=_font(14))
    color = (235, 90, 60) if fell else (235, 235, 235)
    txt = f"theta = {theta_deg:+5.1f} deg" + ("  FELL" if fell else "")
    d.rectangle([(0, img.size[1] - 24), (img.size[0], img.size[1])], fill=(30, 30, 30))
    d.text((6, img.size[1] - 20), txt, fill=color, font=_font(13))
    return img


def main():
    DOCS.mkdir(exist_ok=True)
    os.chdir(_REPO_ROOT / "mujoco_sim")  # segway.xml loads via a cwd-relative path

    sim_rl = SegwaySimulation(
        use_ros2=False,
        controller=RLPolicy.from_files(MODELS / "ppo_segway.zip", MODELS / "vecnormalize.pkl"),
    )
    sim_fixed = SegwaySimulation(use_ros2=False)                       # default fixed-gain LQR
    sim_care = SegwaySimulation(use_ros2=False, controller=make_care_lqr(sim_rl.model))
    panels = [("RL (PPO)", sim_rl), ("LQR (fixed)", sim_fixed), ("LQR (CARE-tuned)", sim_care)]
    for _, s in panels:
        s.reset(pitch_deg=INITIAL_PITCH_DEG)

    renderers = [mujoco.Renderer(s.model, height=PANEL_H, width=PANEL_W) for _, s in panels]
    cam = _make_camera()
    fell = [False, False, False]

    n_steps = int(DURATION_S / SIM_DT) + 1
    frames = []
    print(f"Rendering {DURATION_S}s @ {FPS} fps "
          f"(~{n_steps // RENDER_EVERY} frames, {PANEL_W * 3 + GAP * 2}x{PANEL_H})")

    for i in range(n_steps):
        for k, (_, s) in enumerate(panels):
            s.step()
            fell[k] = fell[k] or s.failed
        if i % RENDER_EVERY == 0:
            combined = Image.new("RGB", (PANEL_W * 3 + GAP * 2, PANEL_H), (255, 255, 255))
            for k, (title, s) in enumerate(panels):
                renderers[k].update_scene(s.data, camera=cam)
                th = float(np.degrees(s.ext.get_theta(s.data)))
                panel = _label_panel(renderers[k].render(), title, th, fell[k])
                combined.paste(panel, (k * (PANEL_W + GAP), 0))
            frames.append(combined)

    for _, s in panels:
        s.close()

    print(f"Captured {len(frames)} frames. Quantizing + writing GIF...")
    quant = [f.quantize(colors=COLORS, method=Image.Quantize.MEDIANCUT) for f in frames]
    out = DOCS / "rl_vs_lqr.gif"
    quant[0].save(out, save_all=True, append_images=quant[1:],
                  duration=int(round(1000 / FPS)), loop=0, optimize=True)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
