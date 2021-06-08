import socket
import time
import numpy as np
import pickle
import pygame
import sys
import random

import torch
import torch.nn as nn
from train_cae_dropout import CAE
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear GPU
torch.cuda.empty_cache()


# Storing some important states
HOME = np.asarray([-0.000453, -0.7853, 0.000176, -2.355998, -0.000627, 1.572099, 0.7859])
SOUPCAN = [np.asarray([-0.305573, 0.70963, -0.183403, -1.500632, 0.13223, 2.214831, 0.270835])]
NOTEPAD = [np.asarray([0.344895, 0.796234, 0.227432, -1.514207, -0.195541, 2.286966, 1.421964])]
# CUP = [np.asarray([-0.169027, 0.119563, -0.662496, -2.628833, 0.120918, 2.729636, -0.146264])]
TAPE = [np.asarray([-0.016421, 0.210632, 0.031353, -2.594983, 0.020064, 2.816127, 0.877912])]
CUP = [np.asarray([-0.013822, 0.852029, 0.058359, -1.310726, -0.054624, 2.162042, 0.835634])]

class Model(object):

    def __init__(self, cae_name):
        self.cae_net = CAE()

        model_dict = torch.load(cae_name, map_location='cpu')
        self.cae_net.load_state_dict(model_dict)

        self.cae_net.eval

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

    noise = np.random.normal(0, 0.1, 7)
    dataset = []

    if len(goals) > 1:
        first_point = True
    else:
        first_point = False
    scale = 0.2
    rand_action = np.asarray([0.0]*6)
    noise_level = 0.0
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
            total_time = 60.
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
            if action_interval > 0.00:
                # Get human action
                qdot_h = (goal - current_state)

                a_robot = np.zeros((100, 7))
                for i in range(100):
                    z = model.encoder(start_q + state["q"].tolist())
                    a_robot[i, :] = model.decoder(z, s)

                qdot_r = np.mean(a_robot, axis=0)
                # print("[*] This is the mean action ", qdot_r)
                std = np.std(a_robot, axis=0) * 100
                confidence = np.abs(1 - np.mean(std))
                # print("[*] This is the std ",std)
                # confidence = model.classify(start_q + state["q"].tolist())
                

                if elapsed_time < 2.:
                    qdot_h = np.clip(qdot_h, -0.1, 0.1) + np.clip(noise, -0.075, 0.075)
                    qdot = scale * qdot_h
                    if scale < 1.0:
                        scale += 0.2
                    # print("[*] Using human input only ")
                else:
                    confidence = 1.0
                    # qdot_h = 0.01 * qdot_h
                    qdot_h = np.clip(qdot_h, -0.1, 0.1) + np.clip(noise, -0.075, 0.075)
                    qdot = confidence * qdot_r + (1 - confidence) * qdot_h
                    # print(max(qdot_h))
                    # qdot = qdot_h

                send2robot(conn, qdot.tolist())
                dist = np.linalg.norm(current_state - goal)
                action_time = time.time()

            if data_interval > 0.1 and elapsed_time >= 2.:            
                data_time = time.time()
                data = [elapsed_time] + [state["q"].tolist()] + [qdot_h.tolist()] + [qdot_r.tolist()] + [confidence]
                dataset.append(data)

            if noise_interval > 0.5:
                rand_action[0] = noise_level * 2*(np.random.random()-0.5)
                rand_action[1] = noise_level * 2*(np.random.random()-0.5)
                noise = xdot2qdot(rand_action, state)
                noise_time = time.time()
                print("[*] This is my confidence: ", confidence)
                # print("[*] a_r - a_h: ", np.linalg.norm(qdot_h - qdot_r))

            x_pos = joint2pose(s)
            if x_pos[2] < 0.2:
                dist = 0
    
    # Send completion status
    return dataset

def main():

    demos_num = 5
    save_loc = "data/test_runs/"
    cae_model = "models/task_1_cae_dropout_" + str(demos_num)
    model = Model(cae_model)
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
    soupcan_count = 0
    notepad_count = 0
    cup_count = 0
    tape_count = 0

    print('[*] Main loop...')
    total_run = 20
    current_run = 0
    while True:

        state = readState(conn)
        s = state["q"].tolist()

        u, start, mode, stop = interface.input()
        if start:
            choice = 0
            while current_run < total_run:
                start_time = time.time()
                # choice  = random.choice([0, 1, 2, 3])
                if choice == 0:
                    print('[*] My goal is notepad')
                    dataset = go2goal(conn, NOTEPAD, model)
                    filename = save_loc + "n_dropout_" + str(demos_num) + "_" + str(notepad_count) + ".pkl"
                    notepad_count += 1
                elif choice == 1:
                    print('[*] My goal is soupcan')
                    dataset = go2goal(conn, SOUPCAN, model)
                    filename = save_loc + "c_dropout_" + str(demos_num) + "_" + str(soupcan_count) + ".pkl"
                    soupcan_count += 1
                elif choice == 2:
                    print('[*] My goal is coffee cup')
                    dataset = go2goal(conn, CUP, model)
                    filename = save_loc + "u_dropout_" + str(demos_num) + "_" + str(cup_count) + ".pkl"
                    cup_count += 1
                elif choice == 3:
                    print('[*] My goal is measuring tape')
                    dataset = go2goal(conn, TAPE, model)
                    filename = save_loc + "t_dropout_" + str(demos_num) + "_" + str(tape_count) + ".pkl"
                    tape_count += 1
                pickle.dump(dataset, open(filename, "wb"))
                print('[*] Saved a test run of this size: ',len(dataset))
                print('[*] Saved data at ', filename)
                time.sleep(1.0)
                go2home(conn)
                print('[*] Returned home')
                time.sleep(2.0)
                choice += 1
                if choice == 4:
                    choice = 0
                current_run += 1
        

if __name__ == "__main__":
    main()
