import socket
import time
import numpy as np
import pickle
import pygame
import sys
import random

# Storing some important states
HOME = np.asarray([0.022643, -0.789077, -0.000277, -2.358605, -0.005446, 1.573151, -0.708887])
GOAL_EE = np.asarray([0.50, 0.02665257, 0.25038403])
SIGMA = np.identity(3) * 0.001 
Q_MAX = [2.8, 1.7, 2.8, -0.75, 2.8, 3.7, 2.8]
Q_MIN = [-2.8, -1.7, -2.8, -3.0, -2.8, 0.0, -2.8]

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
    return J_pinv @ xdot

def go2home(conn):
    home = HOME
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


def generate_demo(filename, conn, interface):
        savename = "demos/" + str(filename) + ".pkl"
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
        start_q = np.asarray(state["q"])
        start_pose = joint2pose(start_q)
        goal = np.random.multivariate_normal(GOAL_EE, SIGMA)
        stationary_points = 0
        while True:
            state = readState(conn)
            s = state["q"].tolist()
            pose = joint2pose(state["q"])
            dist = np.linalg.norm(goal - pose)
            u, start, mode, stop = interface.input()
            if stop or stationary_points > 10:
                pickle.dump( demonstration, open( savename, "wb" ) )
                print(demonstration)
                print("[*] Done!")
                print("[*] I recorded this many datapoints: ", len(demonstration))
                print("[*] Saved file at: ", savename)
                return True

            xdot = np.zeros(6)
            xdot[:3] =  np.clip((goal - pose), -0.1, 0.1)
            curr_time = time.time()
            if curr_time - start_time >= steptime:
                if dist < 0.02:
                    stationary_points += 1
                demonstration.append([start_pose.tolist() + pose.tolist(), xdot[:3].tolist()])
                start_time = curr_time
            qdot = xdot2qdot(xdot, state)
            send2robot(conn, qdot)

def main():
    print('[*] Connecting to low-level controller...')
    PORT = 8080
    conn = connect2robot(PORT)
    interface = Joystick()
    for filename in range(1,25):
        generate_demo(filename, conn, interface)


if __name__ == "__main__":
    main()