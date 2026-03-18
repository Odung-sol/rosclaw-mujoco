#!/usr/bin/env python3
"""
plot_client.py (macOS) - UDP로 오는 Segway telemetry를 실시간 플롯
- recv는 별도 스레드
- plot은 30Hz 고정 갱신
- theta/phi는 unwrap + 정렬 + 중복시간 제거로 계단(점프) 최소화
"""

import socket, json, threading, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# ===== 설정 =====
UDP_PORT = 9091
MAXLEN   = 4000
PLOT_HZ  = 30

# ===== 버퍼 =====
t_buf     = deque(maxlen=MAXLEN)
theta_buf = deque(maxlen=MAXLEN)
phi_buf   = deque(maxlen=MAXLEN)
tau_buf   = deque(maxlen=MAXLEN)

_lock = threading.Lock()
_recv = 0

def udp_thread():
    global _recv
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
    sock.bind(("127.0.0.1", UDP_PORT))
    sock.settimeout(1.0)
    print(f"[plot_client] UDP listen 127.0.0.1:{UDP_PORT}")

    while True:
        try:
            raw, _ = sock.recvfrom(8192)
        except socket.timeout:
            continue
        except OSError as e:
            print("[plot_client] socket error:", e)
            break

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue

        # 필드 이름은 네 segway_sim.py가 보내는 키 기준
        t     = float(msg.get("time", 0.0))
        theta = float(msg.get("theta", 0.0))
        phi   = float(msg.get("phi", 0.0))
        tau   = float(msg.get("tau_L", 0.0))   # 일단 L만 플롯

        with _lock:
            t_buf.append(t)
            theta_buf.append(theta)
            phi_buf.append(phi)
            tau_buf.append(tau)
            _recv += 1

threading.Thread(target=udp_thread, daemon=True).start()

# ===== matplotlib =====
plt.rcParams["toolbar"] = "toolbar2"

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
ax[0].set_ylabel("theta (rad)")
ax[1].set_ylabel("phi (rad)")
ax[2].set_ylabel("tau_L")
ax[2].set_xlabel("time (s)")

for a in ax:
    a.grid(True, alpha=0.3)

line1, = ax[0].plot([], [], lw=1.2)
line2, = ax[1].plot([], [], lw=1.2)
line3, = ax[2].plot([], [], lw=1.2)

txt = ax[2].text(0.01, 0.05, "", transform=ax[2].transAxes, fontsize=9)

def _prepare_arrays():
    """버퍼 스냅샷 → 정렬/중복제거/unwrap 적용"""
    with _lock:
        if len(t_buf) < 3:
            return None
        t     = np.asarray(t_buf, dtype=float)
        theta = np.asarray(theta_buf, dtype=float)
        phi   = np.asarray(phi_buf, dtype=float)
        tau   = np.asarray(tau_buf, dtype=float)
        recv  = _recv

    # 1) 시간 기준 정렬 (UDP는 순서가 가끔 꼬일 수 있음)
    idx = np.argsort(t)
    t, theta, phi, tau = t[idx], theta[idx], phi[idx], tau[idx]

    # 2) 같은 time이 여러 개면 마지막 것만 남김 (중복 타임스탬프 제거)
    #    (같은 t가 반복되면 플롯이 더 계단처럼 보임)
    uniq_t, uniq_idx = np.unique(t, return_index=True)
    # np.unique는 첫 번째 인덱스만 주므로, "마지막"을 쓰고 싶으면 아래 방식:
    # 여기선 간단히 첫 번째로 처리해도 개선됨. 더 확실히 하려면 reverse unique 구현 가능.
    t = uniq_t
    theta = theta[uniq_idx]
    phi = phi[uniq_idx]
    tau = tau[uniq_idx]

    # 3) unwrap: 라디안 각도가 -pi~pi 점프하는 걸 연속으로 펼침
    theta_u = np.unwrap(theta)
    phi_u   = np.unwrap(phi)

    return t, theta_u, phi_u, tau, recv

def update(_frame):
    pack = _prepare_arrays()
    if pack is None:
        return line1, line2, line3

    t, theta_u, phi_u, tau, recv = pack

    line1.set_data(t, theta_u)
    line2.set_data(t, phi_u)
    line3.set_data(t, tau)

    # autoscale (매 프레임 relim/autoscale 하면 튀어보일 수 있어서 최소만)
    for a in ax:
        a.relim()
        a.autoscale_view()

    txt.set_text(f"recv={recv}  buf={len(t)}/{MAXLEN}  t={t[-1]:.3f}s")
    return line1, line2, line3

ani = animation.FuncAnimation(
    fig, update,
    interval=int(1000 / PLOT_HZ),
    blit=False,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()
