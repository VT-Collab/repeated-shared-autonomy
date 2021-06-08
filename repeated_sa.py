import socket
import time
import numpy as np
import pickle
import pygame
import sys
import random

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear GPU
torch.cuda.empty_cache()


# Storing some important states
HOME = np.asarray([-0.000453, -0.7853, 0.000176, -2.355998, -0.000627, 1.572099, 0.7859])
SOUPCAN = np.asarray([-0.334918, 0.973279, -0.157398, -1.357066, 0.152549, 2.317455, 0.258333])
NOTEPAD = np.asarray([0.331809, 1.039726, 0.258258, -1.44227, -0.311749, 2.444667, 1.4943])

# Important states in cartesian
NOTEPAD_C = np.asarray([-0.73348988, 0.2997922, -1.01174036, 0.6871058, 0.33944648, -0.51690848])
SOUPCAN_C = np.asarray([-0.5010549, -0.26220472, -1.05470799, -0.31927663, 0.10334563, -0.92563016])
HOME_C = np.asarray([-9.14728418e-02, 4.52350730e-05, -7.32618876e-01, 2.27485177e-04,
 -2.17104407e-03, -7.84353974e-01])


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
    s.bind(('172.16.0.3', PORT))
    s.listen()
    conn, addr = s.accept()
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

def xdot2qdot(xdot, state):
    J_pinv = np.linalg.pinv(state["J"])
    return J_pinv @ np.asarray(xdot)

def joints2cartesian(q, state):
    J = np.asarray(state["J"])
    return J @ np.asarray(q)

def sendRobotHome(conn):
    total_time = 10.0;
    start_time = time.time()
    state = readState(conn)
    current_state = np.asarray(state["q"].tolist())

    # Determine distance between current location and home
    dist = np.linalg.norm(current_state - HOME)
    
    curr_time = time.time()
    elapsed_time = curr_time - start_time
    
    # If distance is over threshold then find traj home
    while dist > 0.002 and elapsed_time < total_time:
        delta = HOME - current_state
        delta = np.clip(delta, -0.1, 0.1)
        send2robot(conn, delta.tolist())
        state = readState(conn)
        current_state = np.asarray(state["q"].tolist())
    # Send completion status
    if dist <= 0.02:
        return True
    elif elapsed_time >= total_time:
        return False
def go2goal(conn, goal):
    total_time = 10.0;
    start_time = time.time()
    state = readState(conn)
    current_state = np.asarray(state["q"].tolist())
    start_q = state["q"].tolist()
    # Determine distance between current location and home
    dist = np.linalg.norm(current_state - goal)
    
    curr_time = time.time()
    elapsed_time = curr_time - start_time
    dataset = []
    # If distance is over threshold then find traj home
    while dist > 0.02 and elapsed_time < total_time:
        delta = goal - current_state
        delta = np.clip(delta, -0.1, 0.1)
        send2robot(conn, delta.tolist())
        state = readState(conn)
        current_state = np.asarray(state["q"].tolist())
        dist = np.linalg.norm(current_state - goal)
        dataset.append(start_q + state["q"].tolist())
        # print(dist)
    # Send completion status
    if dist <= 0.02:
        return True, dataset
    elif elapsed_time >= total_time:
        return False, dataset


def main():

    save_loc = "data/demos/"

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
    scaling_rot = 0.2
    steptime = 0.1
    soupcan_count = 9
    notepad_count = 10

    print('[*] Main loop...')
    while True:

        state = readState(conn)
        s = state["q"].tolist()

        u, start, mode, stop = interface.input()
        if stop:
            # pickle.dump( demonstration, open( filename, "wb" ) )
            print(demonstration)
            sendRobotHome(conn)
            print("[*] Done!")
            return True
        if start:
            while True:
                start_time = time.time()
                print('[*] Going to goal')
                choice  = random.choice([0,1])
                if choice == 0:
                    _, dataset = go2goal(conn, NOTEPAD)
                    filename = save_loc + "n_" + str(notepad_count) + ".pkl"
                    notepad_count += 1
                else:
                    _, dataset = go2goal(conn, SOUPCAN)
                    filename = save_loc + "c_" + str(soupcan_count) + ".pkl"
                    soupcan_count += 1
                
                pickle.dump(dataset, open(filename, "wb"))
                print('[*] Saved data at ', filename)
                time.sleep(1)
                go2goal(conn, HOME)
                print('[*] Returned home')
                time.sleep(2)

        # curr_time = time.time()
        # if record and curr_time - start_time >= steptime:
        #     demonstration.append(s)
        #     start_time = curr_time

        # if mode:
        #     translation_mode = not translation_mode
        # xdot = [0]*6

        # if translation_mode:
        #     xdot[0] = scaling_trans * u[0]
        #     xdot[1] = scaling_trans * u[1]
        #     xdot[2] = scaling_trans * u[2]
        # else:
        #     xdot[3] = scaling_rot * u[0]
        #     xdot[4] = scaling_rot * u[1]
        #     xdot[5] = scaling_rot * u[2]

        # qdot = xdot2qdot(xdot, state)
        # send2robot(conn, qdot)


if __name__ == "__main__":
    main()
