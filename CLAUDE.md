# CLAUDE.md — rosclaw-mujoco

## Overview
ROS2 Segway balancing simulator + OpenClaw AI natural language control.
MuJoCo simulation (macOS) ↔ rosbridge WebSocket ↔ ROS2 Docker (LQR + ROSClaw).

## Key Commands
```bash
# Start ROS2 stack
docker compose up -d

# Test bridge (no MuJoCo needed)
python mujoco_sim/segway_bridge.py

# Full simulation
python mujoco_sim/segway_sim.py
```

## Architecture
- `mujoco_sim/` — MuJoCo simulator + LQR + bridge (macOS native)
- `ros2_ws/` — ROS2 LQR node + ROSClaw discovery (Docker)
- `extensions/` — OpenClaw plugin for natural language control
- `docker-compose.yml` — 3 services: rosbridge, lqr_controller, rosclaw_discovery

## Code Conventions
- Python: snake_case, JSON payloads via std_msgs/String
- TypeScript: strict ESM, ws library
- ROS2 Humble on linux/arm64 (Apple Silicon Docker)
