/**
 * @rosclaw/rosbridge-client
 * Lightweight TypeScript rosbridge WebSocket client.
 */

import WebSocket from "ws";

export interface RosbridgeConfig {
  url: string;
  reconnect?: boolean;
  reconnectInterval?: number;
}

export interface RosMessage {
  op: string;
  topic?: string;
  type?: string;
  msg?: Record<string, unknown>;
  [key: string]: unknown;
}

type MessageHandler = (msg: RosMessage) => void;

export class RosbridgeClient {
  private ws: WebSocket | null = null;
  private config: Required<RosbridgeConfig>;
  private handlers: Map<string, MessageHandler[]> = new Map();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(config: RosbridgeConfig) {
    this.config = {
      url: config.url,
      reconnect: config.reconnect ?? true,
      reconnectInterval: config.reconnectInterval ?? 3000,
    };
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.config.url);
      this.ws.on("open", () => { resolve(); });
      this.ws.on("message", (raw: string) => {
        try {
          const msg: RosMessage = JSON.parse(raw);
          const topic = msg.topic;
          if (topic && this.handlers.has(topic)) {
            this.handlers.get(topic)!.forEach((h) => h(msg));
          }
        } catch { /* skip */ }
      });
      this.ws.on("close", () => {
        if (this.config.reconnect) {
          this.reconnectTimer = setTimeout(() => this.connect(), this.config.reconnectInterval);
        }
      });
      this.ws.on("error", (err: Error) => {
        if (this.ws?.readyState !== WebSocket.OPEN) reject(err);
      });
    });
  }

  advertise(topic: string, type: string): void {
    this.send({ op: "advertise", topic, type });
  }

  publish(topic: string, msg: Record<string, unknown>): void {
    this.send({ op: "publish", topic, msg });
  }

  subscribe(topic: string, type: string, handler: MessageHandler, throttleRate = 0): void {
    if (!this.handlers.has(topic)) {
      this.handlers.set(topic, []);
      this.send({ op: "subscribe", topic, type, throttle_rate: throttleRate });
    }
    this.handlers.get(topic)!.push(handler);
  }

  unsubscribe(topic: string): void {
    this.handlers.delete(topic);
    this.send({ op: "unsubscribe", topic });
  }

  private send(msg: RosMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  close(): void {
    this.config.reconnect = false;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
  }
}

export default RosbridgeClient;
