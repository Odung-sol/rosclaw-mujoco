/**
 * ROSClaw Segway Plugin for OpenClaw
 *
 * OpenClaw AI 에이전트가 자연어로 Segway를 제어.
 * OpenClaw → 이 플러그인 → rosbridge WebSocket → ROS2
 */

import WebSocket from "ws";

const ROSBRIDGE_URL = process.env.ROSBRIDGE_URL || "ws://127.0.0.1:9090";
const CMD_TOPIC = "/segway/cmd_reference";
const STATUS_TOPIC = "/segway/controller/status";
const STATE_TOPIC = "/segway/state";
const MSG_TYPE = "std_msgs/msg/String";

let ws: WebSocket | null = null;
let latestStatus: Record<string, unknown> = {};
let latestState: Record<string, unknown> = {};

function ensureConnection(): WebSocket {
  if (ws && ws.readyState === WebSocket.OPEN) return ws;

  ws = new WebSocket(ROSBRIDGE_URL);

  ws.on("open", () => {
    console.log("[ROSClaw] Connected to rosbridge");
    ws!.send(JSON.stringify({ op: "subscribe", topic: STATUS_TOPIC, type: MSG_TYPE }));
    ws!.send(JSON.stringify({ op: "subscribe", topic: STATE_TOPIC, type: MSG_TYPE, throttle_rate: 500 }));
    ws!.send(JSON.stringify({ op: "advertise", topic: CMD_TOPIC, type: MSG_TYPE }));
  });

  ws.on("message", (raw: string) => {
    try {
      const msg = JSON.parse(raw);
      if (msg.topic === STATUS_TOPIC) latestStatus = JSON.parse(msg.msg.data);
      else if (msg.topic === STATE_TOPIC) latestState = JSON.parse(msg.msg.data);
    } catch { /* ignore */ }
  });

  ws.on("error", (err: Error) => console.error("[ROSClaw]", err.message));

  return ws;
}

function publishCommand(command: Record<string, unknown>): void {
  const conn = ensureConnection();
  conn.send(JSON.stringify({
    op: "publish", topic: CMD_TOPIC,
    msg: { data: JSON.stringify(command) },
  }));
}

export const tools = [
  {
    name: "segway_status",
    description: "Segway 상태 조회 (밸런스, 위치, 컨트롤러 정보)",
    parameters: {},
    execute: async () => {
      ensureConnection();
      await new Promise((r) => setTimeout(r, 300));
      return {
        controller: latestStatus,
        state: {
          theta_deg: ((latestState.theta as number) || 0) * 180 / Math.PI,
          position_m: latestState.x || 0,
          velocity_ms: latestState.x_dot || 0,
        },
      };
    },
  },
  {
    name: "segway_move",
    description: "Segway를 목표 위치로 이동 (미터 단위)",
    parameters: { x: { type: "number", description: "목표 위치 (m)" } },
    execute: async ({ x }: { x: number }) => {
      publishCommand({ command: "move_to", x });
      return { success: true, message: `x=${x}m 으로 이동 중` };
    },
  },
  {
    name: "segway_velocity",
    description: "Segway 목표 속도 설정 (m/s)",
    parameters: { velocity: { type: "number", description: "목표 속도 (m/s)" } },
    execute: async ({ velocity }: { velocity: number }) => {
      publishCommand({ command: "set_velocity", velocity });
      return { success: true, message: `속도 ${velocity}m/s 설정` };
    },
  },
  {
    name: "segway_enable",
    description: "밸런싱 컨트롤러 활성화",
    parameters: {},
    execute: async () => {
      publishCommand({ command: "enable" });
      return { success: true, message: "컨트롤러 활성화됨" };
    },
  },
  {
    name: "segway_stop",
    description: "긴급 정지 — 토크 0, 컨트롤러 비활성화",
    parameters: {},
    execute: async () => {
      publishCommand({ command: "disable" });
      return { success: true, message: "긴급 정지 실행됨" };
    },
  },
  {
    name: "segway_tune",
    description: "LQR 게인 조정. Q=[theta, theta_dot, x, x_dot] 가중치",
    parameters: {
      Q_diag: { type: "array", description: "상태 가중치 [theta, theta_dot, x, x_dot]", items: { type: "number" } },
      R_val: { type: "number", description: "제어 effort 가중치" },
    },
    execute: async ({ Q_diag, R_val }: { Q_diag?: number[]; R_val?: number }) => {
      const cmd: Record<string, unknown> = { command: "update_gains" };
      if (Q_diag) cmd.Q_diag = Q_diag;
      if (R_val) cmd.R_val = R_val;
      publishCommand(cmd);
      return { success: true, message: "LQR 게인 업데이트됨" };
    },
  },
  {
    name: "segway_reset",
    description: "컨트롤러 초기 상태로 리셋",
    parameters: {},
    execute: async () => {
      publishCommand({ command: "reset" });
      return { success: true, message: "리셋 완료" };
    },
  },
];

export default {
  name: "rosclaw-segway",
  displayName: "ROSClaw Segway Controller",
  description: "Control a self-balancing Segway robot via ROS2",
  version: "1.0.0",
  tools,
};
