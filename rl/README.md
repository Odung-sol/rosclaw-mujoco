# RL (PPO) balancing controller

A reinforcement-learning alternative to the LQR for balancing the Segway. It
trains a PPO policy against the **same** MuJoCo physics (`mujoco_sim/segway.xml`)
and **same** state pipeline (`state_extractor`) as the LQR, so the policy is a
drop-in alternative and a fair head-to-head comparison.

Runs entirely **macOS-native** (like `mujoco_sim/`) — no Docker, no ROS2.

## Install (into the MuJoCo env)

The RL deps are isolated from CI/Docker. Install them into the same Python
environment that runs the MuJoCo sim:

```bash
pip install -r requirements-rl.txt   # gymnasium, stable-baselines3, torch
```

## Train

```bash
python -m rl.train --timesteps 300000
```

Saves to `rl/models/`: `ppo_segway.zip` (policy), `vecnormalize.pkl` (obs-norm
stats), and `training_curve.png` / `.csv` (episode reward vs timesteps). These
artifacts are gitignored — regenerate by retraining.

## Evaluate (RL vs LQR)

```bash
python -m rl.evaluate --model rl/models/ppo_segway.zip
```

Rolls both controllers through the same env from matched initial pitches and
prints a metrics table (peak |θ|, settling time, torque RMS, position drift,
survival). Also prints the **phi_dot sign-convention verification** (confirms
the corrected `(L − R)/2` getter is in use so the comparison isn't biased).

## Pieces

| File | What |
|---|---|
| `segway_env.py` | `SegwayBalanceEnv(gym.Env)` wrapping `SegwaySimulation` |
| `reward.py` | 4-term quadratic balancing reward + alive bonus |
| `metrics.py` | per-episode summary metrics (pure, CI-tested) |
| `train.py` | PPO training + training-curve plot |
| `evaluate.py` | RL-vs-LQR comparison + phi_dot verification |
| `policy_adapter.py` | run the trained policy in the live sim via `--rl` |
| `lqr_tuning.py` | CARE-tuned LQR gain (fair baseline; physics read from the model) |
| `benchmark.py` | objective metrics + region-of-attraction sweeps (`python -m rl.benchmark`) |
| `plot_results.py` / `render_gif.py` | result figures + the comparison GIF |

## Objective comparison

`python -m rl.benchmark` scores each controller (RL / fixed LQR / CARE-tuned LQR)
by the metrics the self-balancing-robot literature reports: settling time,
peak/overshoot, control-effort RMS, position drift, and the two robustness
limits — **max recoverable tilt** (region of attraction) and **max recoverable
impulse**. Takeaway: a CARE-tuned LQR is the stronger controller on nearly every
metric; RL has by far the smallest region of attraction (recovers from only a 2°
tilt / 10 N push vs the LQR's ≥30° / ≥160 N). See the README's RL section for the
table and figures.

See `docs/superpowers/specs/2026-06-22-rl-ppo-balancing-design.md` for the design.
