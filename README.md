# rosclaw-mujoco

> ROS2 Segway Balancing Simulator + LLM-based Natural Language Control

[![CI](https://github.com/Odung-sol/rosclaw-mujoco/actions/workflows/ci.yml/badge.svg)](https://github.com/Odung-sol/rosclaw-mujoco/actions/workflows/ci.yml)

MuJoCo 시뮬레이터의 Segway(Inverted Pendulum)를 LQR 컨트롤러로 밸런싱하고,
**Gemini API 기반 자연어 처리 노드**와 [ROSClaw](https://github.com/PlaiPin/rosclaw)를 통해
자연어 명령으로 로봇을 제어할 수 있는 시스템입니다.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  User                                                        │
│  "앞으로 1.5미터 부드럽게 이동해" / "진동 줄여줘"                  │
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

## Use Case Scenario (LLM-based LQR Control)

본 프로젝트는 단순한 하드코딩된 명령을 넘어, **LLM(대형 언어 모델)이 제어 시스템의 최상위 계획자(High-level Planner) 역할**을 수행하는 구조를 갖추고 있습니다.

1. **Natural Language Input** — 사용자가 터미널이나 챗봇을 통해 일상 언어로 명령합니다.
   > "앞으로 1.5미터 부드럽게 이동해", "로봇 진동이 너무 심한데 튜닝 좀 해줘"
2. **LLM Intent Parsing (`gemini_nlp_node`)** — ROS2 네이티브 노드에 탑재된 Gemini API가 사용자의 의도를 분석하여, JSON 형태의 제어 명령으로 변환합니다.
3. **ROS2 Middleware** — 변환된 데이터는 외부 서버를 거치지 않고 `/segway/cmd_reference` 토픽을 통해 발행됩니다.
4. **LQR Gain Scheduling** — `lqr_controller_node`가 목표 상태를 업데이트하거나, LQR 제어기의 Q/R 가중치를 실시간으로 재조정합니다 (On-the-fly Tuning).
5. **MuJoCo Simulation** — 계산된 최적 휠 토크가 WebSocket 브리지를 통해 물리 엔진으로 전달되어, Segway가 즉각 반응합니다.

## Features

- **LQR Balancing** — 선형화된 Segway 모델 기반 최적 제어 (CARE solver + rollback)
- **Gemini NLP Node** — 자연어 → JSON 명령 변환 (rate limiting, 스키마 검증)
- **MuJoCo Simulation** — 실제 STL 메쉬를 사용한 물리 시뮬레이션
- **ROSClaw + OpenClaw** — AI 에이전트로 자연어 제어 (이동, 정지, 게인 조정 등)
- **CI/CD Pipeline** — GitHub Actions (ruff lint + 38 unit tests + Docker arm64 build)
- **WebSocket Bridge** — macOS ↔ Docker ROS2 안정적 통신
- **Docker First** — Apple Silicon (M4) 네이티브 지원

## Project Structure

```
rosclaw-mujoco/
├── README.md
├── CLAUDE.md                        # AI 코딩 에이전트용 컨텍스트
├── LICENSE                          # Apache-2.0
├── requirements.txt                 # Python 의존성
├── .gitignore
├── docker-compose.yml               # ROS2 스택 (3개 서비스)
│
├── docker/
│   └── Dockerfile.ros2              # ros:humble + rosbridge + scipy
│
├── mujoco_sim/                      # ← 기존 시뮬레이션 코드
│   ├── segway_sim.py                # MuJoCo 시뮬레이터 (메인)
│   ├── segway.xml                   # MuJoCo MJCF 모델
│   ├── lqr_controller.py            # LQR 컨트롤러 (standalone)
│   ├── state_extractor.py           # 상태 추출기
│   ├── plot_client.py               # 실시간 그래프
│   ├── torque_test.py               # 토크 테스트
│   ├── segway_bridge.py             # ★ NEW: ROS2 WebSocket 브리지
│   └── meshes/
│       ├── body.stl
│       ├── L_wheel.stl
│       └── R_wheel.stl
│
├── ros2_ws/src/                     # ★ NEW: ROS2 노드
│   └── segway_controller/
│       ├── lqr_controller_node.py   # ROS2 LQR 제어 노드
│       ├── gemini_nlp_node.py       # ★ Gemini API 자연어 파싱 노드
│       ├── nlp_cli_node.py          # ★ 터미널 입력 발행 노드
│       ├── discovery_node.py        # ROSClaw 자동탐색 노드
│       └── params.yaml              # 물리 파라미터 + LQR 가중치
│
├── tests/                           # ★ NEW: 단위 테스트 (38개)
│   ├── conftest.py                  # ROS2/Gemini mock fixtures
│   ├── test_gemini_nlp_node.py      # NLP 노드 테스트 (21개)
│   └── test_lqr_controller_node.py  # LQR 노드 테스트 (17개)
│
├── .github/workflows/ci.yml        # ★ NEW: CI/CD 파이프라인
│
├── extensions/                      # ★ NEW: OpenClaw 플러그인
│   └── openclaw-plugin/
│       ├── src/index.ts             # 7개 tool (move, stop, tune 등)
│       ├── package.json
│       └── tsconfig.json
│
├── packages/                        # ★ NEW: TypeScript 라이브러리
│   └── rosbridge-client/
│       ├── src/index.ts             # rosbridge WebSocket 클라이언트
│       └── package.json
│
└── rosclaw_scripts/                 # 기존 ROSClaw 스크립트
```

**★ NEW** = 이번에 추가된 파일 (기존 파일은 그대로 유지)

## Installation

### 1. Prerequisites (macOS)

```bash
# Python 3.10+
brew install python3

# Docker Desktop (Apple Silicon 버전)
# https://www.docker.com/products/docker-desktop
```

### 2. Python 패키지

```bash
cd rosclaw-mujoco
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Docker ROS2 스택

```bash
# 첫 실행 (이미지 빌드 포함, 3-5분)
docker compose up -d

# 확인
docker compose ps
# segway_ros2, segway_lqr, segway_discovery 3개 모두 Up
```

## Quick Start

### Step 1: 통신 테스트 (MuJoCo 없이)

```bash
python mujoco_sim/segway_bridge.py
```

예상 출력:
```
[Bridge] Connected to ws://127.0.0.1:9090
[Bridge] Topics ready.
  step      theta          x       torque
------------------------------------------
     0     +0.0500    +0.0000    +0.0000
    50     +0.0215    +0.0001    -2.1453
  [OK] Balanced for 30s!
```

### Step 2: MuJoCo 시뮬레이션

```bash
python mujoco_sim/segway_sim.py
```

### Step 3: ROS2 토픽 모니터링

```bash
docker exec segway_ros2 bash -c \
  "source /opt/ros/humble/setup.bash && ros2 topic list"

docker exec segway_ros2 bash -c \
  "source /opt/ros/humble/setup.bash && ros2 topic echo /segway/state"
```

### Step 4: Gemini 자연어 제어 (선택)

```bash
# API 키 설정 (https://aistudio.google.com/app/apikey 에서 발급)
export GOOGLE_API_KEY="your-key-here"

# CLI로 자연어 명령 입력
python ros2_ws/src/segway_controller/nlp_cli_node.py
# → "앞으로 1미터 이동해" 입력 → Gemini가 JSON 파싱 → LQR 제어기로 전달
```

### Step 5: OpenClaw 연동 (선택)

```bash
brew install node
npm install -g pnpm
cd extensions/openclaw-plugin
pnpm install && pnpm build
```

## ROS2 Topics

| Topic | Direction | Hz | Description |
|---|---|---|---|
| `/segway/state` | MuJoCo → ROS2 | 100 | 로봇 상태 (theta, x, velocity) |
| `/segway/cmd_torque` | ROS2 → MuJoCo | 100 | 휠 토크 명령 |
| `/segway/nlp_input` | User → Gemini NLP | on-demand | 자연어 텍스트 입력 |
| `/segway/cmd_reference` | Gemini NLP / OpenClaw → LQR | on-demand | JSON 제어 명령 |
| `/segway/controller/status` | ROS2 → All | 10 | 컨트롤러 상태 |
| `/rosclaw/capabilities` | Discovery → All | 1 | 로봇 능력 정보 (자동탐색) |

### State Message

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

| command | parameters | 설명 |
|---------|-----------|------|
| `move_to` | `x` (m) | 목표 위치로 이동 |
| `set_velocity` | `velocity` (m/s) | 목표 속도 설정 |
| `enable` | — | 컨트롤러 시작 |
| `disable` | — | 긴급 정지 |
| `update_gains` | `Q_diag`, `R_val` | LQR 게인 변경 |
| `reset` | — | 초기화 |

## OpenClaw 자연어 예시

```
"segway 상태 알려줘"              → segway_status
"앞으로 1미터 이동"               → segway_move(x=1.0)
"밸런싱 시작"                    → segway_enable
"긴급 정지"                      → segway_stop
"LQR 게인 Q=[200,20,2,10]으로"   → segway_tune(Q_diag=[200,20,2,10])
```

## Testing & CI/CD

```bash
# 단위 테스트 실행 (Gemini API mock — 과금 없음)
pip install -r requirements-dev.txt
pytest tests/ -v

# Lint 검사
ruff check .
```

GitHub Actions가 모든 push/PR에 대해 자동으로 실행합니다:
- **lint-and-test** — ruff 코드 스타일 + pytest 38개 테스트
- **docker-build** — Docker arm64 이미지 빌드 검증

## Troubleshooting

| Issue | Solution |
|---|---|
| `platform (linux/amd64) does not match` | `docker-compose.yml`에 `platform: linux/arm64` 확인 |
| `ros2: command not found` (macOS) | ROS2는 Docker 내부에서만 실행됨 |
| `librmw_cyclonedds_cpp.so not found` | `RMW_IMPLEMENTATION` 환경변수 제거 |
| Topic이 안 보임 | publisher 연결 후 최소 2초 대기 필요 |
| WebSocket 끊김 | `ping_interval=10` 설정 확인 |
| Segway가 넘어짐 | `params.yaml`에서 `Q_diag` 첫 번째 값 증가 |
| Segway 발진 (oscillation) | `Q_diag` 감소, `R_val` 증가 |
| MuJoCo viewer 안 열림 | `segway_bridge.py`로 통신 먼저 테스트 |

## LQR 게인 튜닝

```
state = [θ, θ̇, x, ẋ]
u = -K @ state
```

- `Q_diag[0]` (θ) ↑ = 더 빠르게 세움
- `Q_diag[1]` (θ̇) ↑ = 진동 억제
- `Q_diag[2]` (x) ↑ = 위치 추적 강화
- `Q_diag[3]` (ẋ) ↑ = 속도 안정성
- `R_val` ↑ = 보수적 제어 (작은 토크)

## References

- [ROSClaw](https://github.com/PlaiPin/rosclaw) — OpenClaw ↔ ROS2 integration
- [ROS2 Humble](https://docs.ros.org/en/humble/)
- [rosbridge_suite](https://github.com/RobotWebTools/rosbridge_suite)
- [MuJoCo](https://mujoco.org/)

## License

Apache-2.0 — See [LICENSE](LICENSE)
