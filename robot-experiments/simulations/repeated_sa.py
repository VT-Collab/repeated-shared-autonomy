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

# GLASS_1 = np.asarray([-1.71012, -0.323783, 2.805134, -2.512991, -1.942274, 1.779556, 0.465056])
# GLASS_2 = np.asarray([-1.849151, -0.446339, 2.7138, -2.319702, -2.111138, 1.932963, 0.470417])
# GLASS_3 = np.asarray([-1.898036, -0.520775, 2.70938, -2.21471, -2.123111, 1.993295, 0.436397])
# GLASS_4 = np.asarray([-1.95431, -0.587327, 2.696936, -2.101126, -2.150754, 2.059856, 0.429128])
# GLASS_5 = np.asarray([-2.076943, -0.740182, 2.68599, -1.810833, -2.1805, 2.198324, 0.415956])
# GLASS_6 = np.asarray([-2.040561, -0.261006, 2.595029, -1.796662, -2.05092, 2.444951, -0.107962])

GLASS_1 = np.asarray([0.270788, 0.514913, 0.045394, -1.486561, -1.448856, 1.396283, -0.328713])
GLASS_2 = np.asarray([0.265837, 0.810107, 0.049898, -1.51005, -1.418245, 1.4389, -0.013954])
GLASS_3 = np.asarray([0.009951, 0.909226, -0.085316, -1.405642, -1.575763, 1.761664, -0.04302])
GLASS_4 = np.asarray([-0.164737, 1.006999, -0.150719, -1.236234, -1.64325, 1.956706, -0.091362])
GLASS_5 = np.asarray([-0.182065, 0.630679, -0.134357, -1.240687, -1.556667, 1.984427, -0.497718])

# skip 2 and 4??
GLASS = [GLASS_2, GLASS_4, GLASS_5]#, GLASS_6]#, GLASS_7, GLASS_8, GLASS_9]
# SOUPCAN_ROT = np.asarray([0.067057, 0.793774, -0.374163, -1.673316, -1.744919, 2.099874, 1.936402])
# # Location of soupcan and notepad
# OBJ = [-0.2988283336162567, -0.28685495257377625\
#                         , 0.18556469678878784, -0.16989314556121826]

# # Important states in cartesian
# NOTEPAD_C = np.asarray([-0.73348988, 0.2997922, -1.01174036, 0.6871058, 0.33944648, -0.51690848])
# SOUPCAN_C = np.asarray([-0.5010549, -0.26220472, -1.05470799, -0.31927663, 0.10334563, -0.92563016])
# HOME_C = np.asarray([-9.14728418e-02, 4.52350730e-05, -7.32618876e-01, 2.27485177e-04,
#  -2.17104407e-03, -7.84353974e-01])

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

def go2goal(conn, goals, model):
    state = readState(conn)
    current_state = np.asarray(state["q"].tolist())
    start_q = state["q"].tolist()
    # Determine distance between current location and home

    noise = np.random.normal(0, 0.1, 7)
    dataset = []

    if len(goals) > 1:
        first_point = True
    else:
        first_point = False
    scale = 0.2
    rand_action = np.asarray([0.0]*6)
    noise_level = 0.
    idx = 0.
    for goal in goals:
        # If distance is over threshold then find traj home
        dist = np.linalg.norm(current_state - goal)
        start_time = time.time()
        curr_time = time.time()
        data_time = time.time()
        noise_time = time.time()
        action_time = time.time()
        elapsed_time = curr_time - start_time
        time_step = 20

        if first_point:
            total_time = 27.0
        else:
            total_time = 35.
            first_point = False
        while dist > 0.02 and elapsed_time < total_time:
            state = readState(conn)
            s = state["q"].tolist()
            current_state = np.asarray(s)
            curr_time = time.time()
            elapsed_time = curr_time - start_time
            data_interval = curr_time - data_time
            noise_interval = curr_time - noise_time
            action_interval = curr_time - action_time
            confidence = 0
            if action_interval > 0.005:
                # Get human action
                qdot_h = (goal - current_state)
                qdot_h = np.clip(qdot_h, -0.1, 0.1) + np.clip(noise, -0.075, 0.075)

                z = model.encoder(HOME.tolist() + state["q"].tolist())
                z_true = [-1]
                # print('[*] My current z is: ', z)
                qdot_r = model.decoder(z, s)

                confidence = model.classify(HOME.tolist() + state["q"].tolist())
                

                if elapsed_time < 4.5:
                    qdot = scale * qdot_h
                    if scale < 1.0:
                        scale += 0.2

                    # print('[*] Using human input only')
                else:
                    # confidence = 1.0
                    qdot = confidence * qdot_r + (1 - confidence) * qdot_h
                    # print('[*] Using z:', z)
                    # qdot = qdot_h

                send2robot(conn, qdot.tolist())
                dist = np.linalg.norm(current_state - goal)
                action_time = time.time()

            if data_interval > 0.1:            
                data_time = time.time()
                dataset.append(start_q + state["q"].tolist())

            if noise_interval > 0.5:
                rand_action[0] = noise_level * 2*(np.random.random()-0.5)
                rand_action[1] = noise_level * 2*(np.random.random()-0.5)
                noise = xdot2qdot(rand_action, state)
                noise_time = time.time()
                print("[*] This is my confidence: ", confidence)
                # print(noise)
            # x_pos = joint2pose(s)
            # if x_pos[2] < 0.1:
            #     dist = 0

    # Send completion status
    if dist <= 0.02:
        return True, dataset
    else:
        return False, dataset


def main():

    save_loc = "data/demos/"
    model = Model("models/classifier_glass_new", "models/cae_glass_new_2")
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
    glass_count = 0

    print('[*] Main loop...')
    # go2home(conn)
    while True:

        state = readState(conn)
        s = state["q"].tolist()

        u, start, mode, stop = interface.input()
        if start:
            while True:
                start_time = time.time()
                choice  = random.choice([0, 1, 2, 3])
                choice = 4
                if choice == 0:
                    print('[*] My goal is notepad')
                    _, dataset = go2goal(conn, NOTEPAD, model)
                    filename = save_loc + "n_" + str(notepad_count) + ".pkl"
                    notepad_count += 1
                elif choice == 1:
                    print('[*] My goal is soupcan')
                    _, dataset = go2goal(conn, SOUPCAN, model)
                    filename = save_loc + "c_" + str(soupcan_count) + ".pkl"
                    soupcan_count += 1
                elif choice == 2:
                    print('[*] My goal is coffee cup')
                    _, dataset = go2goal(conn, CUP, model)
                    filename = save_loc + "u_" + str(cup_count) + ".pkl"
                    cup_count += 1
                elif choice == 3:
                    print('[*] My goal is measuring tape')
                    _, dataset = go2goal(conn, TAPE, model)
                    filename = save_loc + "t_" + str(tape_count) + ".pkl"
                    tape_count += 1
                elif choice == 4:
                    print('[*] My goal is wine glass')
                    _, dataset = go2goal(conn, GLASS, model)
                    filename = save_loc + "g_" + str(glass_count) + ".pkl"
                    glass_count += 1
                # pickle.dump(dataset, open(filename, "wb"))
                # print('[*] Saved a demonstration of this size: ',len(dataset))
                # print('[*] Saved data at ', filename)
                time.sleep(1)
                go2home(conn)
                print('[*] Returned home')
                time.sleep(2)

if __name__ == "__main__":
    main()
