#!/usr/bin/env python3
"""
plot_client.py (macOS) - UDP로 오는 Segway telemetry를 실시간 플롯
- recv는 별도 스레드
- plot은 30Hz 고정 갱신
- 슬라이딩 윈도우(최근 WINDOW_S 초만 표시) → 계단 현상 제거
"""

import socket, json, threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# ===== 설정 =====
UDP_PORT = 9091
MAXLEN   = 6000          # 버퍼 크기 (넉넉하게)
PLOT_HZ  = 30            # 플롯 갱신 주파수
WINDOW_S = 10.0          # 화면에 표시할 시간 범위(초) - PlotJuggler처럼 스크롤

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

        t     = float(msg.get("time", 0.0))
        theta = float(msg.get("theta", 0.0))
        phi   = float(msg.get("phi", 0.0))
        tau   = float(msg.get("tau_L", 0.0))

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
ax[0].set_ylabel("theta (deg)")
ax[1].set_ylabel("phi (deg)")
ax[2].set_ylabel("tau_L (Nm)")
ax[2].set_xlabel("time (s)")

for a in ax:
    a.grid(True, alpha=0.3)

line1, = ax[0].plot([], [], lw=1.2, color='tab:blue')
line2, = ax[1].plot([], [], lw=1.2, color='tab:orange')
line3, = ax[2].plot([], [], lw=1.2, color='tab:green')

txt = ax[2].text(0.01, 0.05, "", transform=ax[2].transAxes, fontsize=9)

def _prepare_arrays():
    """버퍼 스냅샷 → 슬라이딩 윈도우(최근 WINDOW_S초) 추출"""
    with _lock:
        if len(t_buf) < 3:
            return None
        t     = np.asarray(t_buf, dtype=float)
        theta = np.asarray(theta_buf, dtype=float)
        phi   = np.asarray(phi_buf, dtype=float)
        tau   = np.asarray(tau_buf, dtype=float)
        recv  = _recv

    # 1) 시간 기준 정렬
    idx = np.argsort(t)
    t, theta, phi, tau = t[idx], theta[idx], phi[idx], tau[idx]

    # 2) 중복 타임스탬프 제거 (마지막 값 유지)
    #    np.unique는 첫 번째를 반환하므로 reversed unique로 마지막 값 선택
    _, first_idx = np.unique(t, return_index=True)
    # 역방향으로 unique → 마지막 인덱스 구하기
    _, last_idx_rev = np.unique(t[::-1], return_index=True)
    last_idx = len(t) - 1 - last_idx_rev
    last_idx = np.sort(last_idx)
    t     = t[last_idx]
    theta = theta[last_idx]
    phi   = phi[last_idx]
    tau   = tau[last_idx]

    # 3) 슬라이딩 윈도우: 최근 WINDOW_S 초만 추출
    t_end   = t[-1]
    t_start = t_end - WINDOW_S
    mask    = t >= t_start
    if mask.sum() < 2:
        return None
    t     = t[mask]
    theta = theta[mask]
    phi   = phi[mask]
    tau   = tau[mask]

    # 4) unwrap: ±π 점프 제거
    #    theta = 실제 pitch 공식 → unwrap 적용하면 부드러운 연속 곡선
    #    phi   = 바퀴 누적 회전 → unwrap 필수
    theta_u = np.unwrap(theta)
    phi_u   = np.unwrap(phi)

    # 5) radian → degree 변환
    theta_deg = np.degrees(theta_u)
    phi_deg   = np.degrees(phi_u)

    return t, theta_deg, phi_deg, tau, recv

def update(_frame):
    pack = _prepare_arrays()
    if pack is None:
        return line1, line2, line3

    t, theta_u, phi_u, tau, recv = pack

    line1.set_data(t, theta_u)
    line2.set_data(t, phi_u)
    line3.set_data(t, tau)

    # x축: 슬라이딩 윈도우 고정 범위
    x_end   = t[-1]
    x_start = x_end - WINDOW_S
    for a in ax:
        a.set_xlim(x_start, x_end)

    # y축: 데이터에 맞게 자동 조절 (여백 10% 추가)
    for a, data in zip(ax, [theta_u, phi_u, tau]):
        if len(data) > 0:
            lo, hi = data.min(), data.max()
            margin = max((hi - lo) * 0.1, 0.01)
            a.set_ylim(lo - margin, hi + margin)

    txt.set_text(f"recv={recv}  buf={len(t)}/{MAXLEN}  t={t[-1]:.3f}s  window={WINDOW_S:.0f}s")
    return line1, line2, line3

ani = animation.FuncAnimation(
    fig, update,
    interval=int(1000 / PLOT_HZ),
    blit=False,
    cache_frame_data=False
)

plt.tight_layout()
plt.show()
