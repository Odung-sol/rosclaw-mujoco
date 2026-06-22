# RL (PPO) Balancing Controller for Segway — Design Spec

- **Date:** 2026-06-22
- **Status:** Approved by user (pending spec review)
- **Scope decision:** Train a PPO policy + run it as an alternative *local-sim*
  controller. **No Docker / ROS2 changes** in this iteration.

---

## 1. Goal

Add a reinforcement-learning (PPO) controller that balances the same Segway
modeled in `mujoco_sim/segway.xml`, as a drop-in alternative to the existing
LQR. The trained policy must be runnable in the local MuJoCo sim and directly
comparable to the LQR baseline.

The LLM/ROS2 pipeline is untouched. RL lives entirely on the macOS-native side,
mirroring how `mujoco_sim/` already runs MuJoCo natively.

## 2. Decisions (locked with user)

1. **Framework:** Gymnasium + Stable-Baselines3 (SB3) PPO.
2. **Observation = local-LQR state:** `[θ, θ̇, φ, φ̇]` (same vector as
   `mujoco_sim/lqr_controller.py`'s `SegwayLQR`). Guarantees apples-to-apples
   comparison and lets the policy be swapped in behind the same interface.
3. **Action:** continuous scalar in `Box(-1, 1)`, scaled to **per-wheel torque**
   `tau_each = clip(action · max_torque, ±max_torque)` (±20 N·m) and applied
   equally to both wheels. This matches the actuator `ctrlrange` and the
   per-wheel authority `SegwayLQR` has after its own ±20 clip — full
   control-authority parity. *(Corrected during Commit 1 implementation: the
   original "scale to total `Tw` then split `/2`" wording would have capped RL
   at ±10 per wheel — half of LQR's authority — making the comparison unfair.)*
4. **Reward:** 4 terms — penalize `θ²`, `θ̇²`, `φ²`, and normalized control
   effort `τ²`, plus an alive bonus. (See §4.3.)
5. **Commit count:** 4 commits (§8).
6. **First training goal = pure balancing only.** Disturbance robustness is
   explicitly **out of scope** for this iteration (added later, once balancing
   is solid).
7. **Commit 2 must verify** whether the known historical `phi`/`phi_dot`
   sign-convention issue affects the RL-vs-LQR comparison (§4.5).

## 3. Non-goals (this iteration)

- No disturbance forces during training reward/reset (pure balancing).
- No ROS2 node, no `docker-compose` service, no bridge changes.
- No change to the ROS2 LQR node or the Gemini NLP path.
- No new physics parameters or MJCF edits (reuse `segway.xml` as-is).
- Trained model binaries are **not** committed to git (see §6).

## 4. Architecture

### 4.1 Component overview

```
                 rl/segway_env.py
   ┌───────────────────────────────────────────┐
   │ SegwayBalanceEnv(gymnasium.Env)            │
   │   obs  = [θ, θ̇, φ, φ̇]   (via state_extractor)
   │   act  = Box(-1,1) → ±20 N·m → tau_each    │
   │   reward = rl/reward.py                     │
   │   wraps ▼                                   │
   │     mujoco_sim/segway_sim.py                │
   │       SegwaySimulation(use_ros2=False)      │
   │       .reset(pitch_deg) / set_torque_and_step()
   │       .ext (SegwayStateExtractor)           │
   │       segway.xml  (single physics source)   │
   └───────────────────────────────────────────┘
        │ train.py (PPO.learn)        │ evaluate.py (RL vs SegwayLQR)
        ▼                             ▼
   rl/models/ppo_segway.zip      metrics: θ_max, settling, τ_RMS

   rl/policy_adapter.py  RLPolicy.compute_torque(state) ── same interface as
                                                           SegwayLQR
        │
        ▼
   mujoco_sim/segway_sim.py  `--rl <path>` selects RLPolicy vs SegwayLQR
```

### 4.2 `SegwayBalanceEnv` (`rl/segway_env.py`)

A `gymnasium.Env` that **wraps a `SegwaySimulation(use_ros2=False)` instance**.
It does *not* call the sim's existing `step()` (which runs the LQR); it sets the
torque itself and advances physics.

- **`__init__(self, frame_skip=5, max_episode_steps=2000, reset_pitch_deg=3.0, seed=None)`**
  - Constructs `SegwaySimulation(use_ros2=False)`.
  - `observation_space = Box(-inf, +inf, shape=(4,))` for `[θ, θ̇, φ, φ̇]`.
  - `action_space = Box(-1.0, 1.0, shape=(1,))`.
  - `frame_skip=5` → control at ~100 Hz (5 × 0.002 s mj_steps per env step),
    matching the ROS2 LQR's documented ~100 Hz rate for a fair comparison.
- **`reset(seed, options)`** → calls `sim.reset(pitch_deg=...)` and returns
  `(obs, info)`. Pitch is sampled `U(−reset_pitch_deg, +reset_pitch_deg)` from
  the env RNG, **unless** `options={"pitch_deg": x}` is given, which forces a
  specific pitch — used by `evaluate.py` for deterministic, episode-matched
  resets (§4.5) and by termination tests.
- **`step(action)`**:
  1. `tau_each = clip(action[0] · max_torque, ±max_torque)` (per-wheel), applied
     to both wheels.
  2. Repeat `frame_skip` times: `sim.set_torque_and_step(tau_each, tau_each)`.
  3. Read `obs = sim.ext.get_state(sim.data)`.
  4. `reward = compute_reward(obs, action)` — passes the **raw normalized
     `action`** (the `Box(-1,1)` array), not the scaled torque; see §4.3.
  5. `terminated = abs(θ) > deg2rad(30)` (reuse existing fail threshold).
  6. `truncated = step_count >= max_episode_steps`.
  7. Return `(obs, reward, terminated, truncated, info)`.
- **Disturbance hook available but unused:** because the env wraps
  `SegwaySimulation`, `apply_disturbance()` is reachable for the future
  robustness iteration. It is **not** invoked during training in this iteration.

**Seam into existing code (additive, Commit 1):** add
`SegwaySimulation.set_torque_and_step(tau_L, tau_R)` to `mujoco_sim/segway_sim.py`.
It sets `data.ctrl[L_act/R_act]`, calls `_apply_pending_disturbance()`, then
`mujoco.mj_step`. This is the single low-level primitive the env uses, so the
env never pokes private attributes or duplicates the step mechanics. **Existing
`step()` / `step_ros2()` behavior is unchanged** in Commit 1 (the method is
purely additive); they are refactored to route through the controller seam in
Commit 3.

### 4.3 Reward (`rl/reward.py`)

Kept in its own module so it is unit-testable without MuJoCo. `compute_reward`
receives the **raw normalized `action`** (the same `Box(-1,1)` array `step()`
was given), *not* the scaled torque `Tw`/`tau_each` — `w_τ` is calibrated to the
`[-1, 1]` range.

```
compute_reward(obs, action):
    θ, θ_dot, φ, φ_dot = obs
    τ_norm = action[0]                         # already in [-1, 1]
    cost = w_θ*θ² + w_θdot*θ_dot² + w_φ*φ² + w_τ*τ_norm²
    return alive_bonus - cost
```

- Default weights (tunable hyperparameters, declared as module constants):
  `w_θ=1.0`, `w_θdot=0.05`, `w_φ=0.1`, `w_τ=0.01`, `alive_bonus=1.0`.
- Rationale: dominant term is upright (`θ`); `φ` term lightly discourages drift
  (parity with the LQR's position-regulating gains); control term discourages
  chattering. Final values are tuned empirically during Commit 2; the spec fixes
  the *form*, not the exact constants.

### 4.4 Training (`rl/train.py`)

- CLI entrypoint: `python rl/train.py [--timesteps N] [--seed S] [--out PATH]`.
- Wraps the env in SB3 `VecNormalize` (normalizes obs; recommended for PPO on
  unbounded state), trains `PPO("MlpPolicy", ...)`.
- Saves policy to `rl/models/ppo_segway.zip` and the `VecNormalize` stats
  alongside (`rl/models/vecnormalize.pkl`) so evaluation/inference reproduce the
  same observation scaling.
- **Inference must freeze normalization:** `evaluate.py` and `RLPolicy` load
  `VecNormalize` with `training=False` and `norm_reward=False` so obs scaling is
  fixed and rewards aren't re-normalized at inference (the most common SB3 eval
  bug).
- Prints/saves a short training summary (final mean episode reward/length).

### 4.5 Evaluation (`rl/evaluate.py`) + phi_dot verification

- CLI: `python rl/evaluate.py [--model PATH] [--episodes N]`.
- Runs N episodes of the **trained policy** and N of the **`SegwayLQR`
  baseline** from identical initial conditions (same seeds / initial pitch), in
  the same `SegwaySimulation`, and reports for each: peak `|θ|`, settling time
  to `|θ|<0.5°`, `τ_RMS`, final `|φ|` drift, and survival. Both paths must seed
  `sim.reset(pitch_deg=...)` with the **same drawn pitch value per episode** —
  the gym env RNG and the direct-`step()` baseline consume randomness
  differently, so equal RNG seeds alone don't guarantee equal initial
  conditions.
- **Baseline choice:** compare against the **local `SegwayLQR`** (state
  `[θ,θ̇,φ,φ̇]`), *not* the ROS2 `SegwayLQRController` (state `[θ,θ̇,x,ẋ]`).
  RL and the local LQR then share the **identical** observation pipeline
  (`state_extractor.get_state`), so the position representation cannot bias the
  comparison.

- **phi_dot verification task (required, per user):**
  - Background: `state_extractor.get_phi`/`get_phi_dot` historically used
    `(L + R)/2`, which collapsed to 0 for pure forward/backward motion and
    silently zeroed the LQR's position state. Fixed 2026-04-29 to `(L − R)/2`;
    `SegwayLQR`'s position gains were re-tuned to `(+5, +3)` to match. The ROS2
    node uses a *different* position representation (`x`, not `φ`).
  - Verification to perform in Commit 2 and record in the eval output / spec:
    1. Assert in an eval rollout that `φ̇` is non-zero and correctly signed
       during sustained forward motion (proves the old `(L+R)/2 ≡ 0` bug is not
       silently present in the comparison path).
    2. Confirm RL and `SegwayLQR` read `φ̇` through the same corrected
       `(L − R)/2` getter — i.e. the convention is shared, so it cannot skew the
       head-to-head metrics.
    3. Document the local-LQR (`φ`) vs ROS2-LQR (`x`) divergence so no one
       accidentally benchmarks RL against the ROS2 node's different state.
  - Outcome: either "no effect on comparison (shared corrected pipeline),
    documented" or a concrete follow-up if an effect is found.

### 4.6 Policy adapter + `--rl` seam (`rl/policy_adapter.py`, Commit 3)

- **`RLPolicy`** mirrors `SegwayLQR`'s public interface:
  `compute_torque(state) -> (tau_L, tau_R)`.
  - Loads the SB3 model + `VecNormalize` stats; in `compute_torque`, builds the
    obs from `state`, applies the saved normalization, `model.predict(obs,
    deterministic=True)`, scales action to torque, clips to `±torque_limit`,
    returns `(tau_each, tau_each)`.
  - **Lazy import of SB3** inside `__init__` so the module imports without
    torch/SB3 installed — this lets the interface-parity unit test run on CI
    with a mocked predict (CI has no torch).
- **`segway_sim.py` controller seam:** in `__init__`, select
  `self.controller = SegwayLQR(...)` (default) or `RLPolicy(path)` when
  `--rl <path>` is passed. Refactor `step()` to call
  `self.controller.compute_torque(state)` (replacing the direct `self.lqr`
  call). `--rl` is parallel to the existing `--ros2` flag. **LQR path behavior
  must remain identical** — verified by re-running the existing
  `test_disturbance_recovery.py` (which drives `step()`).

## 5. Folder structure

```
rl/                              # NEW — macOS-native (peer of mujoco_sim/)
├── __init__.py
├── README.md                   # train / eval / --rl usage + dep install note
├── segway_env.py               # SegwayBalanceEnv(gym.Env)
├── reward.py                   # compute_reward() — MuJoCo-free, unit-tested
├── train.py                    # PPO training → rl/models/ppo_segway.zip
├── evaluate.py                 # rollout metrics, RL vs SegwayLQR + phi_dot check
├── policy_adapter.py           # RLPolicy.compute_torque() — SegwayLQR interface
└── models/
    └── .gitkeep                # .zip / .pkl artifacts are gitignored

requirements-rl.txt             # NEW — gymnasium==, stable-baselines3==, torch==

tests/
├── test_reward.py              # NEW — reward contract, pure NumPy (runs on CI)
├── test_segway_env.py          # NEW — env contract + set_torque_and_step (mujoco/gym importorskip)
└── test_rl_policy_adapter.py   # NEW — adapter == SegwayLQR interface (SB3 mocked)
```

Modified existing files:
- `mujoco_sim/segway_sim.py` — `set_torque_and_step()` (C1), `--rl` seam (C3)
- `.gitignore` — ignore `rl/models/*.zip` and `*.pkl` (C2)
- `CLAUDE.md`, `README.md` — docs (C4)

## 6. Dependencies & CI

- New deps go **only** in `requirements-rl.txt`, pinned `==`
  (`gymnasium`, `stable-baselines3`, `torch`). They are **not** added to
  `requirements.txt` / `-dev.txt` / `-ros2.txt`, so CI and the Docker image are
  unchanged (no torch in CI/Docker).
- CI behavior:
  - `test_segway_env.py` → `pytest.importorskip("mujoco")` **and**
    `importorskip("gymnasium")` → **skips on CI**, runs locally on macOS.
  - `test_rl_policy_adapter.py` → SB3 lazy-imported + mocked predict → **runs on
    CI** (pure interface check, no torch needed). If it still needs SB3 symbols,
    it `importorskip`s them.
  - `ruff check rl/` must pass (line-length 110, E/F/W) — add `rl/` to the lint
    invocation in §7 of CLAUDE.md and (optionally) the CI lint step.
- Trained model binaries (`.zip`, `.pkl`) are large → gitignored. A fresh clone
  reproduces them via `python rl/train.py`; `rl/README.md` documents this.

## 7. Testing strategy

- **TDD per commit:** write the test first, watch it fail, implement, watch it
  pass. Each commit leaves `ruff check` + `pytest tests/ -v` green.
- `test_segway_env.py`: observation/action space shapes & bounds; `reset`
  returns a length-4 obs in-space; `step` returns the 5-tuple; `terminated`
  flips when `|θ|>30°`; reward sign sanity (upright+no-torque > tilted); reward
  module tested directly (MuJoCo-free).
- `test_rl_policy_adapter.py`: `compute_torque(state)` returns
  `(tau_L, tau_R)`; respects `±torque_limit` clip; returns equal L/R; signature
  parity with `SegwayLQR` — all with a mocked `predict`, no torch.
- Existing `test_disturbance_recovery.py` re-run after the Commit 3 refactor to
  prove the LQR path is unchanged.

## 8. Commit plan (4 commits)

Each commit is independently reviewable and keeps CI green. After each, **stop
for user review** (per user request).

| # | Commit message | Adds | Existing-pipeline impact |
|---|---|---|---|
| 1 | `feat(rl): Gymnasium balance env wrapping SegwaySimulation` | `requirements-rl.txt`, `rl/{__init__,segway_env,reward}.py`, `SegwaySimulation.set_torque_and_step()` (additive), `tests/test_segway_env.py` | None — additive method only |
| 2 | `feat(rl): PPO training + evaluation against LQR baseline` | `rl/train.py`, `rl/evaluate.py` (+ phi_dot verification), `rl/README.md`, `.gitignore` update | None |
| 3 | `feat(rl): run trained PPO policy in local sim via --rl flag` | `rl/policy_adapter.py`, `segway_sim.py` controller seam + `--rl`, `tests/test_rl_policy_adapter.py` | `step()` small refactor (LQR behavior identical, guarded by existing tests) |
| 4 | `docs(rl): document RL workflow + update CLAUDE.md` | `CLAUDE.md` (§2 repo map, §3 commands, §5 matrix, §6 decisions, §7 lint scope), `README.md` RL section | None |

## 9. Invariants respected (CLAUDE.md §4)

- **No 4th physics copy:** env reuses `segway.xml` via `SegwaySimulation`.
- **No unpinned deps:** all new deps `==` in `requirements-rl.txt`.
- **LLM not in control loop:** unaffected (RL is local, no LLM).
- **rosbridge/127.0.0.1, ROS2 services, Gemini path:** untouched.
- **No `.bak`/backup cruft:** none introduced.
- **Always mock `rclpy` in tests:** new tests don't import ROS2; existing
  conftest mocks remain valid (they're registered globally at conftest import
  time, so they apply to the RL tests too — harmless, since RL tests don't touch
  ROS2).

## 10. Risks & open questions

- **Reward tuning:** the 4 weights and `frame_skip` may need iteration in
  Commit 2 to get reliable balancing; the spec fixes the *form*, constants are
  empirical. Mitigation: short training runs + the eval harness drive tuning.
- **Sample efficiency / training time:** PPO on a 500 Hz sim can be slow.
  `frame_skip=5` (100 Hz control) reduces horizon length. Acceptable for an
  offline, run-locally workflow.
- **CI test isolation:** the adapter test must truly avoid importing torch on
  CI. Mitigation: lazy import + mock; verified by running the adapter test in an
  environment without torch.
- **`set_torque_and_step` vs duplicating step logic:** chosen the additive
  primitive over the env reaching into `sim.data`/private methods; reviewer
  should confirm this is the right seam.
