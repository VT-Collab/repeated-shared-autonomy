import socket
import time
import numpy as np
import pickle
import pygame
import sys

import torch
import torch.nn as nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
END1 = [-0.003122, 0.310032, 0.012054, -2.495443, -0.008477, 2.804512, 0.801903] #End 1 
END2 = [0.00, 1.092539, 0.011473, -1.028987, -0.008504, 2.12294, 0.7976] #End 2

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


def gen_data():
    
    # env_goals = tasks
    env_goals = sys.argv[1]
    goals = []
    radius = 0.6
    # num_goals = int(env_goals)
    n_waypoints = 75
    flag = True
    # rand_i = np.linspace(0.0, 1.0,num_goals+1)
    # savename = "goals/goals" + str(num_goals) + ".pkl"

    filename = "demos/robot/" + str(env_goals) + ".pkl"


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

    print('[*] Main loop...')
    start_state = readState(conn)
    start_state = start_state["q"].tolist()
    while True:

        state = readState(conn)
        # print(state)
        s = state["q"].tolist()
        # print(s)

        u, start, mode, stop = interface.input()
        if stop or len(demonstration) > 160:
            pickle.dump( demonstration, open( filename, "wb" ) )
            print(demonstration)
            print("[*] Done!")
            print("[*] I recorded this many datapoints: ", len(demonstration))
            return True
        if start and not record:
            record = True
            start_time = time.time()
            print('[*] Recording the demonstration...')

        curr_time = time.time()
        if record and curr_time - start_time >= steptime:
            demonstration.append(s + start_state)
            start_time = curr_time

        if mode:
            translation_mode = not translation_mode
        xdot = [0]*6

        if translation_mode:
            # xdot[:3] = (np.asarray(END1) - np.asarray(s))*0.1
            # xdot[:3] = np.clip(xdot[:3], -0.05, 0.05)
            xdot[0] = scaling_trans * u[0]
            xdot[1] = scaling_trans * u[1]
            xdot[2] = scaling_trans * u[2]
        else:
            xdot[3] = scaling_rot * u[0]
            xdot[4] = scaling_rot * u[1]
            xdot[5] = scaling_rot * u[2]

        # qdot = xdot2qdot(xdot, state)
        qdot = [0]*7
        if np.linalg.norm(np.asarray(END1) - np.asarray(s)) > 0.05 and flag and record:
            vel = np.asarray(END1) - np.asarray(s)
            qdot =  np.random.multivariate_normal(vel, np.identity(7) * 0.005)*0.5
            qdot = np.clip(qdot, -0.5, 0.5)
        elif record:
            vel = np.asarray(END2) - np.asarray(s)
            qdot =np.random.multivariate_normal(vel, np.identity(7) * 0.005)*0.5
            qdot = np.clip(qdot, -0.5, 0.5)
            flag = False

        send2robot(conn, qdot)

def main():
    # num_tasks = 1
    gen_data()





if __name__ == "__main__":
    main()
