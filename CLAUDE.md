# CLAUDE.md — rosclaw-mujoco

> Read this file FIRST. Everything below is load-bearing for a cold AI session
> to understand what this repo is, how it runs, and what not to break.

## 1. Overview

ROS2 Segway (inverted-pendulum) balancing simulator with natural-language
control via the OpenClaw plugin. Three-hop pipeline:

```
MuJoCo sim (macOS native)  ─ws─►  rosbridge (Docker)  ─►  LQR + Gemini NLP (Docker)
        ▲                                                       │
        └──────────────── cmd_torque / cmd_reference ◄──────────┘
```

Key external dep: **Google AI Studio** (Gemini 2.x) for natural-language → ROS2
command parsing. LLM is NOT in the control loop — it only translates user
intent into structured commands on `/segway/cmd_reference`.

## 2. Repo map

| Path | What | Where it runs |
|---|---|---|
| `mujoco_sim/` | MuJoCo sim, state extractor, WebSocket bridge, standalone LQR | macOS native (MuJoCo) |
| `ros2_ws/src/segway_bridge/` | rosbridge wrapper node (relays topics) | Docker `ros2_bridge` |
| `ros2_ws/src/segway_controller/` | `lqr_controller_node.py`, `gemini_nlp_node.py`, `discovery_node.py`, `params.yaml` | Docker `lqr_controller` / `gemini_nlp` / `rosclaw_discovery` |
| `extensions/openclaw-plugin/` | OpenClaw TS plugin — wraps rosbridge as a tool for the AI host | Node.js (OpenClaw host) |
| `packages/rosbridge-client/` | Shared TS WebSocket client library | Node.js workspace |
| `docker/Dockerfile.ros2` | `ros:humble-ros-base` + rosbridge + pinned pip deps | all four Docker services |
| `docker-compose.yml` | **4 services**: `ros2_bridge`, `lqr_controller`, `gemini_nlp`, `rosclaw_discovery` | Docker |
| `tests/` | pytest suite — ROS2 mocked via `conftest.py` (no rclpy / std_msgs install needed) | CI + local |

Docker image builds **linux/arm64** only (Apple Silicon).

## 3. Key commands

```bash
# Start full ROS2 stack (4 services)
docker compose up -d

# Smoke-test bridge without MuJoCo
python mujoco_sim/segway_bridge.py

# Full simulation (MuJoCo + ROS2)
python mujoco_sim/segway_sim.py

# Run tests locally
pytest tests/ -v

# Lint
ruff check ros2_ws/ tests/ mujoco_sim/
```

Required env (see `.env.example`): `GOOGLE_API_KEY`. Optional:
`ROSBRIDGE_URL` (default `ws://127.0.0.1:9090`), `ROS_DOMAIN_ID` (default 0).

## 4. Invariants — Never do / Always do

### Never

- **Never commit `.env`** — it holds a live `GOOGLE_API_KEY`. `.gitignore`
  covers `.env` + `.env.*` + `*.pem` + `*.key` + `credentials*` + `secrets*`
  + `*-sa.json` + `.secrets/`. `.env.example` is the allowed one.
- **Never expose port 9090 on `0.0.0.0`.** rosbridge has no auth; it accepts
  `update_gains` / `disable` / arbitrary topic publishes from any peer.
  `docker-compose.yml` binds `127.0.0.1:9090:9090` — keep it that way, or
  front it with an auth proxy.
- **Never revive `google-generativeai`.** The repo migrated to
  `google-genai` (`from google import genai`) in commit `7776481`. Old SDK
  has a different API and is stale.
- **Never change physics params in one place only.** Three files must stay
  in sync: `mujoco_sim/segway.xml` (MJCF) ↔ `ros2_ws/src/segway_controller/
  lqr_controller_node.py` defaults ↔ `ros2_ws/src/segway_controller/
  params.yaml`. Canonical values: `body_mass=37.65`, `body_length=0.14`,
  `body_inertia=5.42`, `wheel_mass=0.85`, `wheel_radius=0.1`.
- **Never unpin deps.** Everything in `requirements*.txt` is `==`; Docker
  installs from `requirements-ros2.txt`. Dependabot bumps them. Floating
  `>=` reintroduced in a PR = reject.
- **Never put LLM calls in the control loop.** Gemini only parses user
  intent on `/segway/nlp_input` → `/segway/cmd_reference`. The LQR runs at
  ~100 Hz and must never depend on the network.
- **Never commit `.bak` / `.broken` / `.before_restore` / `_WORKING` /
  `_old` files.** The backup-attic was purged once; don't rebuild it.

### Always

- **Always use `os.environ.get(...)` for host/port defaults** in Python,
  matching the TS plugin pattern. `ROSBRIDGE_URL` is the only one today.
- **Always mock `rclpy` + `std_msgs` in tests** (see `tests/conftest.py`).
  CI has no ROS2 install.
- **Always validate JSON payloads from rosbridge.** `update_gains`
  validates `Q_diag` (length-4, finite, positive) and `R_val`
  (finite, positive) before touching `self.Q` / `self.R_lqr`. New
  commands that accept array payloads must do the same.
- **Always use `std_msgs/String` JSON payloads** between ROS2 services —
  that is the wire format the TS plugin and bridge expect.

## 5. Change-type → file matrix

| Intent | Touch these |
|---|---|
| Add a new NL command (e.g. "spin_left") | `gemini_nlp_node.py` system prompt + `VALID_COMMANDS` + `lqr_controller_node.py:_on_reference` + a test in `test_lqr_controller_node.py` |
| Add a new sim disturbance / external force | `segway_sim.py:apply_disturbance()` already covers single-axis push at body top; for new shapes, mirror the `pending_disturbance` state machine and add a test in `test_disturbance_recovery.py`. Bridge channel is `/segway/disturbance` (`{"force": float, "duration": float}`). |
| Tune LQR aggressiveness | `params.yaml` `Q_diag` / `R_val` — NOT the node defaults (the defaults are the safety net for missing params file) |
| Change physics | all three files in §4 Never #4 — same commit |
| Add a ROS2 node | new file under `ros2_ws/src/segway_controller/` + new service in `docker-compose.yml` + update §2 Repo map + update §4 if service count changes |
| Add a new env var | `.env.example` + docstring in the reading code + `docker-compose.yml` `environment:` if needed + this file §3 |
| Bump a Python dep | `requirements.txt` (or `-dev`/`-ros2` depending on where it runs) — pinned `==` always |

## 6. Recently-decided (don't re-litigate)

- **Python 3.10** fixed in `pyproject.toml` and CI. Do not bump without
  validating the Docker base image. (2026-03-24)
- **`google-genai` not `google-generativeai`.** New SDK, uses `genai.Client`.
  Commit `7776481`. (2026-03-27)
- **LQR K computed via `scipy.linalg.solve_continuous_are` with a
  hard-coded MATLAB K fallback** on solver failure. Both live in
  `lqr_controller_node.py:_compute_lqr_gain`. Don't remove the fallback.
  (2026-03-20)
- **Dual WebSocket connections** in `segway_bridge.py` (one for publish,
  one for subscribe) — chosen over a single connection because rosbridge's
  advertise + subscribe on the same socket had race conditions at the
  `setup_topics` phase. Don't collapse back to one. (2026-03-18)
- **Korean prompts for Gemini.** The system prompt in `gemini_nlp_node.py`
  is Korean on purpose — user-facing NL input is Korean. English prompts
  worked worse in evals. (2026-03-26)
- **rosbridge bound to `127.0.0.1`.** See §4 Never #2. (2026-04-16)
- **Physics params source of truth = MJCF (`segway.xml`).** Node defaults
  and `params.yaml` must match. (2026-04-16)
- **External disturbance API (Issue #4).** `SegwaySimulation.apply_disturbance(
  force_N, duration_s)` writes a one-shot horizontal force at body-local
  point `BODY_TOP_LOCAL = (0, 0, 0.34)` via `mj_applyFT` into
  `qfrc_applied`. The point matches the segway.xml collision-box top —
  bumping the box geometry without bumping `BODY_TOP_LOCAL` silently moves
  the application point. Bridge channel is `/segway/disturbance` with the
  same JSON-in-String wire format as the rest of the topics. Acceptance:
  1 N × 0.3 s → peak |θ| < 5°, recovers to <0.5° within 2 s. (2026-04-29)

## 7. Verification before merging

```bash
ruff check ros2_ws/ tests/ mujoco_sim/
pytest tests/ -v
# Docker build smoke:
docker compose build
```

CI (`.github/workflows/ci.yml`) runs all three on every push / PR to
`main`. TS packages in `extensions/` and `packages/` also get typechecked
via `npx tsc --noEmit` in the `typescript` matrix job.

## 8. Pointers

- README.md — user-facing setup + feature list
- `docs/ARCHITECTURE.md` — full topic catalogue, component map, and
  control / NL / disturbance flow diagrams. **Read after this file** for
  any change that crosses process boundaries.
- `docs/SECURITY.md` — threat model, port 9090 exposure rules, key
  rotation runbook
- `mujoco_sim/PROJECT_SUMMARY.md` — historical Korean design doc; parts
  are stale (pre-ROS2). Keep for derivations, don't treat as current.
- `.env.example` — canonical env-var list
- `ros2_ws/src/segway_controller/params.yaml` — LQR weights + physics
- `pyproject.toml` — ruff + pytest config
