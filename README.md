# rosclaw-mujoco

> ROS2 Segway Balancing Simulator with NLP-based Natural Language Control

[![CI](https://github.com/Odung-sol/rosclaw-mujoco/actions/workflows/ci.yml/badge.svg)](https://github.com/Odung-sol/rosclaw-mujoco/actions/workflows/ci.yml)

A two-wheeled inverted pendulum (Segway) balanced by an LQR controller in MuJoCo, with a Gemini-powered NLP pipeline that converts natural language commands into robot control signals over ROS2.

<p align="center">
  <img src="docs/demo.gif" alt="Segway LQR Balancing Demo" width="480">
</p>

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

- **LQR Balancing** — Optimal control based on a linearized inverted pendulum model (CARE solver with rollback)
- **Gemini NLP Node** — Natural language to JSON command conversion with rate limiting and schema validation
- **MuJoCo Physics** — Full rigid-body simulation with real STL meshes
- **ROSClaw + OpenClaw** — AI agent interface for natural language control (move, stop, tune gains, etc.)
- **CI/CD Pipeline** — GitHub Actions: ruff lint + 38 unit tests + Docker arm64 build verification
- **WebSocket Bridge** — Stable macOS-to-Docker ROS2 communication
- **Docker First** — Native Apple Silicon (M-series) support

## Project Structure

```
rosclaw-mujoco/
├── docker-compose.yml               # ROS2 stack (4 services)
├── docker/
│   └── Dockerfile.ros2              # ros:humble + rosbridge + scipy
│
├── mujoco_sim/                      # MuJoCo simulation (macOS native)
│   ├── segway_sim.py                # Main simulator with viewer
│   ├── segway.xml                   # MJCF model definition
│   ├── lqr_controller.py            # Standalone LQR controller
│   ├── state_extractor.py           # Quaternion-based state extraction
│   ├── segway_bridge.py             # ROS2 WebSocket bridge client
│   ├── render_demo_gif.py           # Offscreen rendering for demo GIF
│   ├── plot_client.py               # Real-time plotting
│   └── meshes/                      # STL mesh files
│
├── ros2_ws/src/segway_controller/   # ROS2 nodes
│   ├── lqr_controller_node.py       # LQR control node
│   ├── gemini_nlp_node.py           # Gemini NLP parsing node
│   ├── nlp_cli_node.py              # Terminal input publisher
│   ├── discovery_node.py            # ROSClaw auto-discovery
│   └── params.yaml                  # Physical params + LQR weights
│
├── tests/                           # Unit tests (38 total)
│   ├── conftest.py                  # ROS2/Gemini mock fixtures
│   ├── test_gemini_nlp_node.py      # NLP node tests (21)
│   └── test_lqr_controller_node.py  # LQR node tests (17)
│
├── extensions/openclaw-plugin/      # OpenClaw plugin (TypeScript)
│   └── src/index.ts                 # 7 tools (move, stop, tune, etc.)
│
└── packages/rosbridge-client/       # rosbridge WebSocket client lib
    └── src/index.ts
```

## Installation

### Prerequisites (macOS)

```bash
# Python 3.10+
brew install python3

# MuJoCo
pip install mujoco numpy scipy

# Docker Desktop (Apple Silicon)
# https://www.docker.com/products/docker-desktop
```

### Docker ROS2 Stack

```bash
# First run (builds images, takes 3-5 min)
docker compose up -d

# Verify all services are up
docker compose ps
# Expected: segway_ros2, segway_lqr, segway_discovery, segway_nlp
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
| `/segway/nlp_input` | User → Gemini NLP | on-demand | Natural language text input |
| `/segway/cmd_reference` | NLP / OpenClaw → LQR | on-demand | JSON control commands |
| `/segway/controller/status` | ROS2 → All | 10 | Controller status |
| `/rosclaw/capabilities` | Discovery → All | 1 | Robot capability report |

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

## Testing

```bash
# Run unit tests (Gemini API is mocked — no billing)
pip install -r requirements-dev.txt
pytest tests/ -v

# Lint
ruff check .
```

CI runs automatically on every push and PR:
- **lint-and-test** — ruff + pytest (38 tests)
- **docker-build** — arm64 image build verification

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
