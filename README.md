# rosclaw-mujoco

> ROS2 Segway Balancing Simulator with NLP-based Natural Language Control

[![CI](https://github.com/Odung-sol/rosclaw-mujoco/actions/workflows/ci.yml/badge.svg)](https://github.com/Odung-sol/rosclaw-mujoco/actions/workflows/ci.yml)

A two-wheeled inverted pendulum (Segway) balanced by an LQR controller in MuJoCo, with a Gemini-powered NLP pipeline that converts natural language commands into robot control signals over ROS2.

<p align="center">
  <img src="docs/demo.gif" alt="Segway LQR rejecting three escalating disturbances" width="480">
</p>

<p align="center">
  <em>Three escalating pushes, three recoveries. The LQR controller catches the segway after each kick.</em>
</p>

> **Implementer's guide:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) is the
> full topic catalogue, component map, and control / NL / disturbance flow
> diagrams. [`CLAUDE.md`](CLAUDE.md) lists the load-bearing invariants you
> shouldn't break. [`docs/SECURITY.md`](docs/SECURITY.md) covers the threat
> model and key-rotation runbook.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  User                                                        │
│  "move forward 1.5 meters smoothly" / "reduce vibration"    │
└──────────────┬──────────────────────┬────────────────────────┘
               │ CLI (nlp_cli_node)   │ OpenClaw (extensions/)
               ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│  Docker Container (ROS2 Humble / linux/arm64)                │
│                                                              │
│  ┌─────────────────────────┐   ┌───────────────────────────┐ │
│  │  Gemini NLP Node        │   │  rosbridge_websocket :9090│ │
│  │  /segway/nlp_input ──►  │   └────────┬──────────────────┘ │
│  │  Gemini API ──► JSON    │            │ DDS                │
│  │  ──► /segway/cmd_reference           │                    │
│  └──────────┬──────────────┘            │                    │
│             │                           │                    │
│  ┌──────────▼──────────────┐  ┌────────▼──────────────────┐ │
│  │  LQR Controller Node   │  │  ROSClaw Discovery Node   │ │
│  │  - LQR gain scheduling │  │  - capability report      │ │
│  │  - on-the-fly tuning   │  │  - auto-discovery         │ │
│  │  - CARE solver + rollback  └───────────────────────────┘ │
│  └──────────┬──────────────┘                                 │
└─────────────┼────────────────────────────────────────────────┘
              │ WebSocket (/segway/cmd_torque)
┌─────────────▼────────────────────────────────────────────────┐
│  macOS Native                                                │
│  ┌─────────────────┐    ┌──────────────────────┐             │
│  │ segway_bridge.py │◄──►│  MuJoCo Simulator   │             │
│  │ (WS client)      │    │  segway_sim.py      │             │
│  └─────────────────┘    │  + STL meshes        │             │
│                          └──────────────────────┘             │
└──────────────────────────────────────────────────────────────┘
```

## How It Works

This project demonstrates an LLM acting as a **high-level planner** for a real-time control system:

1. **Natural Language Input** — The user issues commands in plain language via terminal or chatbot.
2. **Intent Parsing (Gemini NLP Node)** — A ROS2-native node calls the Gemini API to parse the user's intent into a structured JSON control command.
3. **ROS2 Middleware** — The parsed command is published to `/segway/cmd_reference` over DDS, staying entirely within the local ROS2 graph.
4. **LQR Gain Scheduling** — The `lqr_controller_node` updates target states or re-tunes Q/R weights on the fly.
5. **MuJoCo Simulation** — Computed wheel torques are sent to the physics engine via the WebSocket bridge, and the Segway reacts in real time.

## Features

- **LQR Balancing** — Optimal control based on a linearized inverted pendulum model (CARE solver with hard-coded MATLAB K fallback)
- **Position Regulation** — The controller returns the segway to its starting position after a disturbance, not just upright
- **External Disturbance API** — `SegwaySimulation.apply_disturbance(force_N, duration_s)` injects an impulse at the body's top point; or push the same payload over the `/segway/disturbance` ROS2 topic. Acceptance bound: 1 N × 0.3 s recovers to within 0.5° / 0.1 m in 2 s
- **Gemini NLP Node** — Natural language to JSON command conversion via `google-genai`, with rate limiting and schema validation. **Not in the control loop** — runs at intent-translation latency
- **MuJoCo Physics** — Full rigid-body simulation with real STL meshes (linux/arm64 Docker, macOS-native MuJoCo viewer)
- **ROSClaw + OpenClaw** — AI agent interface for natural language control (move, stop, tune gains, etc.)
- **CI/CD Pipeline** — GitHub Actions: ruff lint + **62 unit tests** + TypeScript typecheck + Docker arm64 build (Buildx GHA-cached)
- **WebSocket Bridge** — Stable macOS-to-Docker ROS2 communication, dual-connection (publish + subscribe) for race-free advertise / subscribe

## Project Structure

```
rosclaw-mujoco/
├── docker-compose.yml               # ROS2 stack (4 services)
├── docker/
│   └── Dockerfile.ros2              # ros:humble + rosbridge + pinned pip deps
│
├── docs/
│   ├── ARCHITECTURE.md              # Full topic catalogue + flow diagrams
│   ├── SECURITY.md                  # Threat model + key rotation runbook
│   └── demo.gif                     # README banner
│
├── mujoco_sim/                      # MuJoCo simulation (macOS native)
│   ├── segway_sim.py                # Main simulator with viewer + apply_disturbance API
│   ├── segway.xml                   # MJCF model definition
│   ├── lqr_controller.py            # Standalone LQR controller
│   ├── state_extractor.py           # Quaternion-based state extraction
│   ├── segway_bridge.py             # ROS2 WebSocket bridge client (dual connection)
│   ├── render_demo_gif.py           # Offscreen rendering for demo GIF
│   ├── plot_client.py               # Real-time plotting
│   └── meshes/                      # STL mesh files
│
├── ros2_ws/src/segway_controller/   # ROS2 nodes
│   ├── lqr_controller_node.py       # LQR control node (CARE + MATLAB fallback)
│   ├── gemini_nlp_node.py           # Gemini NLP parsing node
│   ├── nlp_cli_node.py              # Terminal input publisher
│   ├── discovery_node.py            # ROSClaw auto-discovery
│   └── params.yaml                  # Physical params + LQR weights
│
├── tests/                           # pytest suite (62 total)
│   ├── conftest.py                  # ROS2/Gemini mock fixtures
│   ├── test_gemini_nlp_node.py      # NLP node tests (21)
│   ├── test_lqr_controller_node.py  # LQR node tests (17)
│   ├── test_disturbance_recovery.py # Disturbance API + acceptance (6)
│   └── test_bridge_disturbance.py   # Bridge JSON validation (18)
│
├── extensions/openclaw-plugin/      # OpenClaw plugin (TypeScript)
│   └── src/index.ts                 # 7 tools (move, stop, tune, etc.)
│
├── packages/rosbridge-client/       # rosbridge WebSocket client lib
│   └── src/index.ts
│
├── requirements.txt                 # macOS sim deps (pinned ==)
├── requirements-dev.txt             # pytest + ruff (pinned ==)
├── requirements-ros2.txt            # Docker runtime deps (pinned ==)
├── .env.example                     # Canonical env-var list
└── CLAUDE.md                        # Load-bearing invariants for AI agents
```

## Installation

### Prerequisites (macOS)

```bash
# Python 3.10–3.12 (scipy doesn't ship wheels for 3.13+ yet, so 3.14 will fail
# at install time). Pinned in pyproject.toml; verify with `python3 --version`.
brew install python@3.11        # or python@3.10 / python@3.12

# Recommended: a project-local venv so MuJoCo + scipy live next to the repo.
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt

# Docker Desktop (Apple Silicon)
# https://www.docker.com/products/docker-desktop
```

### Docker ROS2 Stack

```bash
# Copy the env example and fill in your Gemini API key
cp .env.example .env
# Edit .env to set GOOGLE_API_KEY=<your key>

# First run (builds images, takes 3–5 min on first arm64 build, < 1 min after)
docker compose up -d

# Verify all four services are up
docker compose ps
# Expected: segway_ros2 (healthy), segway_lqr, segway_discovery, segway_nlp
```

## Quick Start

### 1. Communication Test (no MuJoCo required)

```bash
python mujoco_sim/segway_bridge.py
```

Expected output:
```
[Bridge] Connected to ws://127.0.0.1:9090
[Bridge] Topics ready.
  step      theta          x       torque
------------------------------------------
     0     +0.0500    +0.0000    +0.0000
    50     +0.0215    +0.0001    -2.1453
  [OK] Balanced for 30s!
```

### 2. MuJoCo Simulation

```bash
python mujoco_sim/segway_sim.py
```

### 3. Monitor ROS2 Topics

```bash
docker exec segway_ros2 bash -c \
  "source /opt/ros/humble/setup.bash && ros2 topic list"
```

### 4. Natural Language Control (optional)

```bash
# Set your API key (get one at https://aistudio.google.com/app/apikey)
export GOOGLE_API_KEY="your-key-here"

# Restart the NLP container to pick up the key
docker compose up -d gemini_nlp

# Send commands via CLI
python ros2_ws/src/segway_controller/nlp_cli_node.py
# Type: "move forward 1 meter" → Gemini parses → LQR executes
```

### 5. OpenClaw Integration (optional)

```bash
brew install node
npm install -g pnpm
cd extensions/openclaw-plugin
pnpm install && pnpm build
```

## ROS2 Topics

| Topic | Direction | Hz | Description |
|---|---|---|---|
| `/segway/state` | MuJoCo → ROS2 | 100 | Robot state (theta, x, velocity) |
| `/segway/cmd_torque` | ROS2 → MuJoCo | 100 | Wheel torque commands |
| `/segway/cmd_reference` | NLP / OpenClaw → LQR | on-demand | JSON control commands |
| `/segway/disturbance` | Operator → MuJoCo | on-demand | External impulse for testing recovery |
| `/segway/nlp_input` | User → Gemini NLP | on-demand | Natural language text input |
| `/segway/controller/status` | ROS2 → All | 10 | Controller status |
| `/rosclaw/capabilities` | Discovery → All | 1 | Robot capability report |

All topics are `std_msgs/String` carrying UTF-8 JSON. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §3 for full payload schemas.

### State Message Format

```json
{
  "timestamp": 1709500000.123,
  "theta": 0.015,
  "theta_dot": -0.003,
  "x": 0.5,
  "x_dot": 0.02,
  "wheel_angle": 12.5,
  "wheel_vel": 1.2
}
```

### Reference Commands

| Command | Parameters | Description |
|---------|-----------|-------------|
| `move_to` | `x` (m) | Move to target position |
| `set_velocity` | `velocity` (m/s) | Set target velocity |
| `enable` | — | Start the controller |
| `disable` | — | Emergency stop |
| `update_gains` | `Q_diag`, `R_val` | Update LQR weights |
| `reset` | — | Reset to initial state |

## NLP Command Examples

```
"move forward 1 meter"         → {"command": "set_velocity", "velocity": 0.5}
"go back slowly"               → {"command": "set_velocity", "velocity": -0.2}
"stop"                         → {"command": "set_velocity", "velocity": 0.0}
"start balancing"              → {"command": "enable"}
"emergency stop"               → {"command": "disable"}
```

## Disturbance Recovery

Push an impulse at the body's top point and watch the LQR recover.

### From a Python script (or test)

```python
sim = SegwaySimulation(use_ros2=False)
sim.reset(pitch_deg=0.0)
sim.apply_disturbance(force_N=1.0, duration_s=0.3)   # 1 N forward kick × 0.3 s
for _ in range(2500):
    sim.step()
# Expected: peak |theta| < 5 deg, |theta| < 0.5 deg at +2 s, x_drift < 0.1 m
```

### Over the ROS2 wire

```bash
docker compose exec ros2_bridge bash -c \
  'ros2 topic pub --once /segway/disturbance std_msgs/msg/String \
   "{data: \"{\\\"force\\\": 1.0, \\\"duration\\\": 0.3}\"}"'
```

The bridge's listener validates the payload (rejects NaN / Inf / non-positive duration) and forwards it into `apply_disturbance()` on the simulator.

The README demo GIF runs three escalating kicks (30 N → 50 N → −80 N) with the LQR active throughout — see `mujoco_sim/render_demo_gif.py` to reproduce.

## Testing

```bash
# Run unit tests (Gemini API is mocked — no billing)
pip install -r requirements-dev.txt
pytest tests/ -v

# Lint
ruff check .
```

CI runs automatically on every push and PR:
- **lint-and-test** — ruff + pytest (62 tests)
- **typescript** — `tsc --noEmit` matrix over `extensions/openclaw-plugin` and `packages/rosbridge-client`
- **docker-build** — linux/arm64 image build verification with Buildx GHA cache

## LQR Gain Tuning

```
state = [theta, theta_dot, x, x_dot]
u = -K @ state
```

| Weight | Effect |
|--------|--------|
| `Q[0]` (theta) ↑ | Faster uprighting |
| `Q[1]` (theta_dot) ↑ | Vibration damping |
| `Q[2]` (x) ↑ | Stronger position tracking |
| `Q[3]` (x_dot) ↑ | Velocity stability |
| `R` ↑ | Conservative control (smaller torques) |

## Troubleshooting

| Issue | Solution |
|---|---|
| `platform (linux/amd64) does not match` | Verify `platform: linux/arm64` in `docker-compose.yml` |
| `ros2: command not found` (macOS) | ROS2 runs inside Docker only |
| Topics not showing | Wait at least 2s after publisher connects |
| WebSocket drops | Check `ping_interval=10` setting |
| Segway falls over | Increase `Q_diag[0]` in `params.yaml` |
| Segway oscillates | Decrease `Q_diag`, increase `R_val` |

## References

- [ROSClaw](https://github.com/PlaiPin/rosclaw) — OpenClaw-ROS2 integration
- [ROS2 Humble](https://docs.ros.org/en/humble/)
- [rosbridge_suite](https://github.com/RobotWebTools/rosbridge_suite)
- [MuJoCo](https://mujoco.org/)
- [Google Gemini API](https://ai.google.dev/)

## License

Apache-2.0 — See [LICENSE](LICENSE)
