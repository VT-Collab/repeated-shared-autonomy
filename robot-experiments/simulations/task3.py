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
HOME = np.asarray([-0.000453, -0.7853, 0.000176, -2.355998, -0.000627, 1.572099, 0.7859])
SOUPCAN = [np.asarray([-0.334918, 0.973279, -0.157398, -1.357066, 0.152549, 2.317455, 0.258333])]
NOTEPAD = [np.asarray([0.331809, 1.039726, 0.258258, -1.44227, -0.311749, 2.444667, 1.4943])]
CUP = [np.asarray([0.031225, 0.951345, 0.053155, -1.176091, -0.052566, 2.173325, 0.874767])]
TAPE = [np.asarray([-0.016421, 0.210632, 0.031353, -2.594983, 0.020064, 2.816127, 0.877912])]


# SHELF_1 = np.asarray([-0.374638, 0.486482, -0.362611, -0.738007, 0.159652, 1.205456, 0.134028])
# SHELF_2 = np.asarray([-0.283282, 0.30204, -0.36736, -1.20683, 0.093675, 1.489563, 0.139279])
# SHELF_3 = np.asarray([-0.064875, 0.073641, -0.259815, -1.517264, 0.004217, 1.574452, 0.451075])

# SHELF = [SHELF_1, SHELF_2, SHELF_3]

SHELF_1 = np.asarray([0.178267, -0.293579, -0.31612, -1.222855, -0.154832, 1.008354, 0.442823])
SHELF_2 = np.asarray([-0.061376, 0.179526, -0.734143, -1.037778, 0.095751, 1.264654, -0.125589])
SHELF_3 = np.asarray([0.059273, -0.224507, -0.623964, -1.862647, -0.180825, 1.758632, 0.081797])
SHELF_4 = np.asarray([0.06259, -0.521078, -0.3806, -2.141666, -0.268599, 1.71679, 0.374477])

SHELF = [SHELF_1, SHELF_2, SHELF_3, SHELF_4]

class Model(object):

    def __init__(self, classifier_name, cae_name):
        self.class_net = Net()
        self.cae_net = CAE()
        
        model_dict = torch.load(classifier_name, map_location='cpu')
        self.class_net.load_state_dict(model_dict)
        
        model_dict = torch.load(cae_name, map_location='cpu')
        self.cae_net.load_state_dict(model_dict)

        self.class_net.eval
        self.cae_net.eval

    def classify(self, c):
        labels = self.class_net.classifier(torch.FloatTensor(c))
        confidence = F.softmax(labels, dim=0)
        return confidence.data[0].numpy()

    def encoder(self, c):
        z_mean_tensor = self.cae_net.encoder(torch.FloatTensor(c))
        return z_mean_tensor.tolist()

    def decoder(self, z, s):
        z_tensor = torch.FloatTensor(z + s)
        a_predicted = self.cae_net.decoder(z_tensor)
        return a_predicted.data.numpy()

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

def send2gripper(conn):
    send_msg = "s"
    conn.send(send_msg.encode())

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
    home = np.copy(HOME)
    home[0] += np.random.uniform(-np.pi/8, np.pi/8)
    home[1] += np.random.uniform(-np.pi/16, np.pi/16)
    home[2] += np.random.uniform(-np.pi/16, np.pi/16) 

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

def go2goal(conn, goals):
    state = readState(conn)
    current_state = np.asarray(state["q"].tolist())
    start_q = state["q"].tolist()
    # Determine distance between current location and home

    dataset = []
    first = True
    for goal in goals:
        # If distance is over threshold then find traj home
        dist = np.linalg.norm(current_state - goal)
        start_time = time.time()
        curr_time = time.time()
        data_time = time.time()
        noise_time = time.time()
        action_time = time.time()
        elapsed_time = curr_time - start_time
        total_time = 35.0;
        if first:
            min_dist = 0.2
            first = False
        else:
            min_dist = 0.02
            
        while dist > min_dist and elapsed_time < total_time:
            state = readState(conn)
            s = state["q"].tolist()
            current_state = np.asarray(s)
            curr_time = time.time()
            elapsed_time = curr_time - start_time
            data_interval = curr_time - data_time
            noise_interval = curr_time - noise_time
            action_interval = curr_time - action_time
            scale = 0.2
            # Get human action
            qdot_h = (goal - current_state)
            qdot_h = np.clip(qdot_h, -0.5, 0.5)
            qdot = qdot_h
            # if scale < 1.0:
            #     scale += 0.2

            send2robot(conn, qdot.tolist())
            dist = np.linalg.norm(current_state - goal)
            action_time = time.time()

            if data_interval > 0.1:            
                data_time = time.time()
                dataset.append(start_q + state["q"].tolist())

            x_pos = joint2pose(s)
            if x_pos[2] < 0.1:
                dist = 0

    # Send completion status
    return dataset

def main():

    save_loc = "data/demos/"
    print('[*] Connecting to low-level controller...')
    PORT = 8080
    # PORT_gripper = 8081
    conn = connect2robot(PORT)
    # conn_gripper = connect2robot(PORT_gripper)
    interface = Joystick()

    print('[*] Initializing recording...')
    demonstration = []
    record = False
    translation_mode = True
    start_time = time.time()
    scaling_trans = 0.1
    scaling_rot = 0.2
    steptime = 0.1
    soupcan_count = 0
    notepad_count = 0
    cup_count = 0
    tape_count = 0
    shelf_count = 0

    print('[*] Main loop...')
    go2home(conn)
    while True:

        state = readState(conn)
        s = state["q"].tolist()
        choice = 1
        u, start, mode, stop = interface.input()
        if start:
            while True:
                start_time = time.time()
                # choice  = random.choice([0, 1])
                # if choice == 0:
                #     print('[*] My goal is measuring tape')
                #     dataset = go2goal(conn, TAPE, model)
                #     filename = save_loc + "t_" + str(tape_count) + ".pkl"
                #     tape_count += 1
                if choice == 1:
                    print('[*] My goal is shelf')
                    dataset = go2goal(conn, SHELF)
                    filename = save_loc + "s_" + str(shelf_count) + ".pkl"
                    shelf_count += 1
                # elif choice == 2:
                #     print('[*] My goal is notepad')
                #     dataset = go2goal(conn, NOTEPAD, model)
                #     filename = save_loc + "n_" + str(notepad_count) + ".pkl"
                #     notepad_count += 1
                pickle.dump(dataset, open(filename, "wb"))
                print('[*] Saved a demonstration of this size: ',len(dataset))
                print('[*] Saved data at ', filename)
                time.sleep(1.0)
                go2home(conn)
                print('[*] Returned home')
                time.sleep(2.0)
                # choice += 1
                # if choice == 3:
                #     choice = 0
                # current_run += 1

if __name__ == "__main__":
    main()
