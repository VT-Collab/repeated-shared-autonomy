import socket
import time
import numpy as np
import pickle
import pygame
import sys
import random

import torch
import torch.nn as nn
from train_cae import CAE
from train_classifier import Net
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear GPU
torch.cuda.empty_cache()


# Storing some important states
# HOME = np.asarray([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
HOME = np.asarray([0.022643, -0.789077, -0.000277, -2.358605, -0.005446, 1.573151, -0.708887])
# HOME = np.asarray([0.066701, -0.61756, 0.041429, -1.712728, -0.003661, 1.092369, -0.671616])
Q_MAX = [2.8, 1.7, 2.8, -0.75, 2.8, 3.7, 2.8]
Q_MIN = [-2.8, -1.7, -2.8, -3.0, -2.8, 0.0, -2.8]
NOTEPAD = np.asarray([0.344895, 0.796234, 0.227432, -1.514207, -0.195541, 2.286966, 1.421964])
TAPE = np.asarray([-0.205312, 0.142973, -0.581547, -2.623161, 0.138313, 2.741362, -0.122495])

class Joystick(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.deadband = 0.1
        self.timeband = 0.5
        self.lastpress = time.time()

    def input(self):
        pygame.event.get()
        curr_time = time.time()
        dx = self.gamepad.get_axis(0)
        dy = -self.gamepad.get_axis(1)
        dz = -self.gamepad.get_axis(4)
        if abs(dx) < self.deadband:
            dx = 0.0
        if abs(dy) < self.deadband:
            dy = 0.0
        if abs(dz) < self.deadband:
            dz = 0.0
        A_pressed = self.gamepad.get_button(0) and (curr_time - self.lastpress > self.timeband)
        B_pressed = self.gamepad.get_button(1) and (curr_time - self.lastpress > self.timeband)
        START_pressed = self.gamepad.get_button(7) and (curr_time - self.lastpress > self.timeband)
        if A_pressed or B_pressed or START_pressed:
            self.lastpress = curr_time
        return [dx, dy, dz], A_pressed, B_pressed, START_pressed


def connect2robot(PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('127.0.0.1', PORT))
    s.listen()
    conn, addr = s.accept()
    s.setblocking(False)
    return conn

def send2robot(conn, qdot, limit=1.0):
    qdot = np.asarray(qdot)
    scale = np.linalg.norm(qdot)
    if scale > limit:
        qdot = np.asarray([qdot[i] * limit/scale for i in range(7)])
    send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
    send_msg = "s," + send_msg + ","
    conn.send(send_msg.encode())

def listen2robot(conn):
    state_length = 7 + 7 + 7 + 42
    message = str(conn.recv(2048))[2:-2]
    state_str = list(message.split(","))
    for idx in range(len(state_str)):
        if state_str[idx] == "s":
            state_str = state_str[idx+1:idx+1+state_length]
            break
    try:
        state_vector = [float(item) for item in state_str]
    except ValueError:
        return None
    if len(state_vector) is not state_length:
        return None
    state_vector = np.asarray(state_vector)
    state = {}
    state["q"] = state_vector[0:7]
    state["dq"] = state_vector[7:14]
    state["tau"] = state_vector[14:21]
    state["J"] = state_vector[21:].reshape((7,6)).T
    return state

def readState(conn):
    while True:
        state = listen2robot(conn)
        if state is not None:
            break
    return state

def joint2pose(q):
    def RotX(q):
        return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
    def RotZ(q):
        return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    def TransX(q, x, y, z):
        return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
    def TransZ(q, x, y, z):
        return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    H1 = TransZ(q[0], 0, 0, 0.333)
    H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
    H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
    H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
    H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
    H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
    H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
    H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.2105)
    H = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
    return H[:,3][:3]

def xdot2qdot(xdot, state):
    J_pinv = np.linalg.pinv(state["J"])
    return J_pinv @ np.asarray(xdot)

def go2home(conn):
    home = HOME
    home[0] += np.random.uniform(-np.pi/8, np.pi/8)
    home[1] += np.random.uniform(-np.pi/16, np.pi/16)
    # home[2] += np.random.uniform(-np.pi/16, np.pi/16) 

    total_time = 35.0;
    start_time = time.time()
    state = readState(conn)
    current_state = np.asarray(state["q"].tolist())

    # Determine distance between current location and home
    dist = np.linalg.norm(current_state - home)
    curr_time = time.time()
    action_time = time.time()
    elapsed_time = curr_time - start_time

    # If distance is over threshold then find traj home
    while dist > 0.02 and elapsed_time < total_time:
        current_state = np.asarray(state["q"].tolist())

        action_interval = curr_time - action_time
        if action_interval > 0.005:
            # Get human action
            qdot = home - current_state
            qdot = np.clip(qdot, -0.4, 0.4)
            send2robot(conn, qdot.tolist())
            action_time = time.time()

        state = readState(conn)
        dist = np.linalg.norm(current_state - home)
        curr_time = time.time()
        elapsed_time = curr_time - start_time

    # Send completion status
    if dist <= 0.02:
        return True
    elif elapsed_time >= total_time:
        return False


def main():
    usernumber = sys.argv[1]
    filename = sys.argv[2]
    savename = "user/user" + str(usernumber) + "/demos/" + str(filename) + ".pkl"
    print('[*] Connecting to low-level controller...')
    PORT = 8080
    conn = connect2robot(PORT)
    interface = Joystick()

    print('[*] Initializing recording...')
    demonstration = []
    record = False
    translation_mode = True
    start_time = time.time()
    scaling_trans = 0.1
    scaling_rot =0.2
    steptime = 0.1

    print('[*] Main loop...')
    go2home(conn)
    state = readState(conn)
    print('[*] Waiting for start...')
    start_q = np.asarray(state["q"])
    start_pose = joint2pose(start_q)

    while True:
        state = readState(conn)
        s = state["q"].tolist()
        pose = joint2pose(state["q"])

        u, start, mode, stop = interface.input()
        if stop:
            pickle.dump( demonstration, open( savename, "wb" ) )
            print(demonstration)
            print("[*] Done!")
            print("[*] I recorded this many datapoints: ", len(demonstration))
            print("[*] Saved file at: ", savename)
            return True
        if start and not record:
            record = True
            start_time = time.time()
            print('[*] Recording the demonstration...')

        curr_time = time.time()
        if record and curr_time - start_time >= steptime:
            demonstration.append(start_q.tolist() + s)
            start_time = curr_time

        # if mode:
        #     translation_mode = not translation_mode
        xdot = [0]*6

        if translation_mode:
            xdot[0] = scaling_trans * u[0]
            xdot[1] = scaling_trans * u[1]
            xdot[2] = scaling_trans * u[2]
        # else:
        #     xdot[3] = scaling_rot * u[0]
        #     xdot[4] = scaling_rot * u[1]
        #     xdot[5] = scaling_rot * u[2]

        qdot = xdot2qdot(xdot, state)
        send2robot(conn, qdot)


if __name__ == "__main__":
    main()