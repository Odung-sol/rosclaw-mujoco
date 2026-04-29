# ARCHITECTURE — rosclaw-mujoco

Companion to `CLAUDE.md`. CLAUDE.md tells a cold AI session **what not to
break**; this file tells the same agent **how the system actually flows
together** so they can reason about new features without re-reading every
node.

---

## 1. The 30-second mental model

Three processes, two protocols:

```
┌──────────────────────────┐    WebSocket (rosbridge JSON v2)    ┌──────────────────────────┐
│  MuJoCo simulator        │  ◄─────────────────────────────►   │  rosbridge_websocket     │
│  (macOS native)          │     ws://127.0.0.1:9090            │  (Docker, port-forwarded │
│                          │                                    │   to loopback only)      │
│  segway_sim.py           │                                    └────────────┬─────────────┘
│   ├─ MuJoCo physics      │                                                 │ DDS / ROS2
│   ├─ SegwayLQR (local)   │                                                 │
│   └─ SegwayROSBridge ────┘                                                 │
└──────────────────────────┘                                                 ▼
                                                            ┌────────────────────────────┐
                                                            │ ROS2 nodes (Docker)        │
                                                            │  ├─ lqr_controller_node    │
                                                            │  ├─ gemini_nlp_node        │
                                                            │  ├─ rosclaw_discovery_node │
                                                            │  └─ nlp_cli_node (CLI)     │
                                                            └────────────────────────────┘

                                                            ▲
                                                            │ same WebSocket
                                                            │
                                              ┌─────────────┴──────────────┐
                                              │  OpenClaw plugin (Node.js) │
                                              │  extensions/openclaw-plugin │
                                              └────────────────────────────┘
```

**Key invariant:** the LQR control loop is closed *across the WebSocket* —
state goes out from MuJoCo, torque comes back from the Docker LQR. The
loop runs at ~100 Hz; the WebSocket adds a few ms of latency, which is
why the physics integration timestep is 2 ms (`SIM_DT = 0.002`) and the
LQR was tuned with that delay in mind.

For ad-hoc / unit-test scenarios there is also a *local* LQR in
`mujoco_sim/lqr_controller.py` that runs in-process — this is what the
demo GIF and the disturbance recovery tests use. It gets the same K
gain matrix the Docker LQR computes via CARE, but with no network in
the loop.

---

## 2. Components

| Component | Process | Source | Role |
|---|---|---|---|
| **MuJoCo simulator** | macOS native | `mujoco_sim/segway_sim.py` | Physics (MJCF model in `segway.xml`), state extraction, optional MuJoCo viewer, optional bridge connection |
| **State extractor** | macOS native | `mujoco_sim/state_extractor.py` | Pulls `[θ, θ̇, φ, φ̇]` out of `MjData` for the LQR. **Convention warning:** `get_phi` averages wheel joint angles as `(L − R) / 2` because the L joint hinges about −y and the R joint about +y (mirror axes in segway.xml) — averaging them with `(L + R) / 2` would zero out for forward motion. CLAUDE.md §6 records the bug fix. |
| **Local LQR** | macOS native | `mujoco_sim/lqr_controller.py` | Hard-coded MATLAB-tuned K. Used for headless tests and the demo GIF. Matches the gain the Docker CARE solver produces for the default `Q_diag`. |
| **WebSocket bridge** | macOS native | `mujoco_sim/segway_bridge.py` | Two `websocket.WebSocket` connections (one for publish, one for subscribe — see CLAUDE.md §6 for why a single socket race-conditioned). Runs a daemon listener thread that fans out incoming messages by topic. |
| **rosbridge_suite** | Docker (`ros2_bridge` service) | `ros-humble-rosbridge-suite` | Translates rosbridge JSON v2 ↔ DDS. Bound to `127.0.0.1:9090` (CLAUDE.md §4 Never #2). |
| **LQR ROS2 node** | Docker (`lqr_controller`) | `ros2_ws/src/segway_controller/lqr_controller_node.py` | Subscribes `/segway/state`, computes torque via K, publishes `/segway/cmd_torque`. Also handles `/segway/cmd_reference` (start/stop, retune Q/R). Publishes `/segway/controller/status` at 10 Hz. |
| **Gemini NLP node** | Docker (`gemini_nlp`) | `ros2_ws/src/segway_controller/gemini_nlp_node.py` | Subscribes `/segway/nlp_input` (raw Korean text), calls Gemini 2.x via `google-genai` SDK, publishes structured commands on `/segway/cmd_reference`. **Not in the control loop.** |
| **NLP CLI** | Docker (`gemini_nlp` shell) | `ros2_ws/src/segway_controller/nlp_cli_node.py` | Read line from stdin, publish on `/segway/nlp_input`. For terminal-driven demos. |
| **Discovery node** | Docker (`rosclaw_discovery`) | `ros2_ws/src/segway_controller/discovery_node.py` | Publishes a JSON capabilities document on `/rosclaw/capabilities` at 1 Hz so OpenClaw hosts can discover what topics this robot exposes. |
| **OpenClaw plugin** | Node.js (host) | `extensions/openclaw-plugin/src/index.ts` | Wraps the rosbridge as a tool the LLM can use. Subscribes status + state, advertises `/segway/cmd_reference`. |

---

## 3. Topic catalogue (single source of truth)

Every topic is `std_msgs/msg/String` carrying a UTF-8 JSON payload — that
wire format is a hard invariant (CLAUDE.md §4 Always #4) so the
rosbridge JSON encoding stays uniform on every hop.

| Topic | Payload schema | Direction | Producer | Consumers |
|---|---|---|---|---|
| `/segway/state` | `{timestamp, theta, theta_dot, x, x_dot, wheel_angle, wheel_vel}` | sim → controllers | `segway_bridge.publish_state` | `lqr_controller_node`, `openclaw-plugin` |
| `/segway/cmd_torque` | `{torque}` (single float, N·m, applied to both wheels) | LQR → sim | `lqr_controller_node._on_state` | `segway_bridge` listener (sets `latest_torque`) |
| `/segway/cmd_reference` | `{command: "stop"\|"forward"\|"backward"\|"reset"\|"update_gains", ...}` — `update_gains` carries `Q_diag` (length 4) and/or `R_val` | high-level → LQR | `gemini_nlp_node`, `openclaw-plugin`, manual `ros2 topic pub` | `lqr_controller_node._on_reference` |
| `/segway/disturbance` | `{force: float (N), duration: float (s)}` | operator → sim | manual `ros2 topic pub` (or future NLP) | `segway_bridge` listener → `SegwaySimulation.apply_disturbance` |
| `/segway/nlp_input` | `{text: "한국어 문장"}` (or raw plain text) | user → NLP | `nlp_cli_node`, OpenClaw | `gemini_nlp_node._on_nlp_input` |
| `/segway/controller/status` | `{enabled, K, Q_diag, R_val, last_state_dt}` | LQR → observers | `lqr_controller_node._publish_status` (10 Hz timer) | `openclaw-plugin`, dashboards |
| `/rosclaw/capabilities` | full discovery doc — see `discovery_node.py:CAPABILITIES` | discovery → hosts | `discovery_node._publish_capabilities` (1 Hz) | OpenClaw hosts |

### Validation

Anything that mutates LQR state validates first — see
`lqr_controller_node._on_reference`'s `Q_diag` / `R_val` checks
(length-4 finite positive) and `segway_bridge._parse_disturbance_payload`
(rejects bool / NaN / Inf / non-positive duration). Any *new* topic that
takes structured input must do the same; bad payloads should be dropped
silently (logging is a free DoS lever).

---

## 4. Closed-loop control flow

The high-frequency path that has to keep the segway upright:

```
┌──────────────────────────────────────────────────────────────────┐
│                        every SIM_DT = 2 ms                       │
│                                                                  │
│   MjData ──► state_extractor.get_state() ──► [θ, θ̇, φ, φ̇]       │
│                                                  │               │
│                                                  ▼               │
│                                       publish /segway/state      │
│                                                  │               │
│                                                  ▼  ws_pub       │
│                                       rosbridge → DDS → lqr      │
│                                                                  │
│                                       lqr._on_state              │
│                                       torque = K @ state - r·ref │
│                                                  │               │
│                                                  ▼               │
│                                       publish /segway/cmd_torque │
│                                                  │               │
│                                                  ▼  ws_sub       │
│                                       bridge.latest_torque updated│
│                                                                  │
│   data.ctrl[L_act] = data.ctrl[R_act] = latest_torque            │
│   mujoco.mj_step()                                               │
└──────────────────────────────────────────────────────────────────┘
```

In `--ros2` mode `step_ros2()` does the round-trip every tick. In the
default mode `step()` skips the network and calls
`SegwayLQR.compute_torque` in-process — same K, ~100x faster.

---

## 5. High-level command flow (NL → LQR)

The path that translates natural language into setpoint changes:

```
operator types Korean text on a CLI or in OpenClaw
        │
        ▼  /segway/nlp_input  ({"text": "..."})
gemini_nlp_node._on_nlp_input
        │  google-genai client.generate_content(...)
        │  prompt asks Gemini for a JSON command
        ▼
        {"command": "forward", "speed": 0.5}     ← validated against VALID_COMMANDS
        │
        ▼  /segway/cmd_reference
lqr_controller_node._on_reference
        │  ├─ stop / forward / backward / reset → set self.ref state
        │  └─ update_gains → re-solve CARE for the new Q/R, swap K atomically
        ▼
LQR uses the new reference / K for subsequent /segway/state messages
```

The Gemini call adds ~1–3 s of latency, which is fine for an
intent-translation layer but would catastrophically break a feedback
loop — that's why CLAUDE.md §4 forbids putting it in the control path.

---

## 6. External disturbance flow (Issue #4)

For *testing* recovery against an external kick, e.g. simulating a hand
shoving the segway:

```
operator publishes JSON                   { "force": 1.0, "duration": 0.3 }
        │  rostopic pub /segway/disturbance ...
        ▼
rosbridge → segway_bridge listener
        │  _parse_disturbance_payload validates, stores in latest_disturbance
        ▼
sim step (--ros2 mode):
  bridge.pop_disturbance() → returns dict once, then None    ← single-shot
  sim.apply_disturbance(force_N, duration_s)
        │
        ▼
each tick while data.time < end_time:
  mj_applyFT(force at BODY_TOP_LOCAL=(0,0,0.34), torque=0)   → qfrc_applied
  mj_step
        │
        ▼
once data.time ≥ end_time: pending_disturbance = None
```

Acceptance bound (codified in `tests/test_disturbance_recovery.py`):
1 N × 0.3 s impulse → peak |θ| < 5°, |θ| < 0.5° within 2 s, x_drift
< 0.1 m. Validated against the hard-coded MATLAB K.

---

## 7. Entry points

| Goal | Command | Notes |
|---|---|---|
| Run full ROS2 stack | `docker compose up -d` | 4 services. `ros2_bridge` healthchecks; the others depend on it. Logs: `docker compose logs -f`. |
| Sim alone (local LQR, MuJoCo viewer) | `mjpython mujoco_sim/segway_sim.py` | macOS Cocoa requires `mjpython` for the GUI; plain `python` works for `--headless`. |
| Sim with Docker LQR | `mjpython mujoco_sim/segway_sim.py --ros2` | Requires `docker compose up -d` first. |
| Bridge round-trip smoke test | `python mujoco_sim/segway_bridge.py` | No MuJoCo, just toy physics. Verifies the WebSocket plumbing. |
| Issue an NL command | `docker compose exec gemini_nlp python3 /root/ros2_ws/src/segway_controller/nlp_cli_node.py` | Or any `ros2 topic pub /segway/nlp_input ...`. |
| Send a kick | `docker compose exec ros2_bridge bash -c 'ros2 topic pub --once /segway/disturbance std_msgs/msg/String "{data: \"{\\\"force\\\": 1.0, \\\"duration\\\": 0.3}\"}"'` | Body-top push for `duration` seconds. |
| Render the README demo GIF | `mjpython mujoco_sim/render_demo_gif.py` | Headless renderer; output goes to `docs/demo.gif`. |
| Tests + lint | `pytest tests/ -v` and `ruff check ros2_ws/ tests/ mujoco_sim/` | Same as CI. |

---

## 8. Where to look next

- **Full state-vector / dynamics derivation** — `mujoco_sim/PROJECT_SUMMARY.md`. Korean, partly stale (pre-ROS2), but the equations of motion and the LQR derivation are still the canonical write-up.
- **MJCF model** — `mujoco_sim/segway.xml`. Keep in sync with `params.yaml` and `lqr_controller_node` defaults (CLAUDE.md §4 Never #4).
- **CI pipeline** — `.github/workflows/ci.yml`. Lint + pytest + TS matrix + Buildx-cached arm64 Docker build.
- **Security model** — `docs/SECURITY.md`. Threat model + key-rotation runbook. Read before exposing port 9090 outside loopback.
