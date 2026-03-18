import numpy as np
import socket
import json
from collections import deque

import matplotlib
matplotlib.use("MacOSX")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

UDP_IP = "127.0.0.1"
UDP_PORT = 9091

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

t_buf = deque(maxlen=500)
theta_buf = deque(maxlen=500)
phi_buf = deque(maxlen=500)
tau_buf = deque(maxlen=500)

fig, ax = plt.subplots(3,1,sharex=True)

line1, = ax[0].plot([],[])
line2, = ax[1].plot([],[])
line3, = ax[2].plot([],[])

ax[0].set_ylabel("theta")
ax[1].set_ylabel("phi")
ax[2].set_ylabel("torque")
ax[2].set_xlabel("time")


def update(frame):

    try:
        while True:

            data,_ = sock.recvfrom(4096)

            msg = json.loads(data.decode())

            t = msg["time"]
            theta = msg["theta"]
            phi = msg["phi"]
            tau = msg["tau_L"]

            t_buf.append(t)
            theta_buf.append(theta)
            phi_buf.append(phi)
            tau_buf.append(tau)

    except BlockingIOError:
        pass

    if len(t_buf) > 2:
        theta_plot = np.unwrap(np.array(theta_buf))
        phi_plot = np.unwrap(np.array(phi_buf))

        line1.set_data(t_buf,theta_buf)
        line2.set_data(t_buf,phi_buf)
        line3.set_data(t_buf,tau_buf)

        ax[0].relim()
        ax[0].autoscale_view()

        ax[1].relim()
        ax[1].autoscale_view()

        ax[2].relim()
        ax[2].autoscale_view()

    return line1,line2,line3


ani = FuncAnimation(fig,update,interval=30)

plt.show()
