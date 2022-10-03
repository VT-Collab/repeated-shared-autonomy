import socket
import time
import numpy as np
import pickle
import pygame
import sys
import copy

# HOME = np.asarray([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
HOME = np.asarray([0.022643, -0.789077, -0.000277, -2.358605, -0.005446, 1.573151, -0.708887])
Q_MAX = [2.8, 1.7, 2.8, -0.75, 2.8, 3.7, 2.8]
Q_MIN = [-2.8, -1.7, -2.8, -3.0, -2.8, 0.0, -2.8]
# NOTEPAD = np.asarray([-0.022897, 1.024858, 0.069253, -1.080779, -0.003219, 2.066114, 0.802576])
NOTEPAD = np.asarray([0.328076, 0.657091, 0.280051, -1.781078, -0.234177, 2.411355, 1.514409])
TAPE = np.asarray([0.498362, 0.366738, 0.72834, -2.583318, -0.697905, 2.808314, 2.645841])

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
    s.bind(('localhost', PORT))
    s.listen()
    conn, addr = s.accept()
    return conn

def send2robot(conn, state, qdot, limit=1.0):
    qdot = np.asarray(qdot)
    scale = np.linalg.norm(qdot)
    if scale > limit:
        qdot *= limit/scale
    qdot = jointlimits(state, qdot)
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

def jointlimits(state, qdot):
    for idx in range(7):
        if state["q"][idx] > Q_MAX[idx] and qdot[idx] > 0:
            qdot[idx] = 0.0
        elif state["q"][idx] < Q_MIN[idx] and qdot[idx] < 0:
            qdot[idx] = 0.0
    return qdot

def go2home(conn):
    home = np.copy(HOME)
    # home[0] += np.random.uniform(-np.pi/8, np.pi/8)
    # home[1] += np.random.uniform(-np.pi/16, np.pi/16)
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
        # Get human action
        qdot = home - current_state
        qdot = np.clip(qdot, -0.4, 0.4)
        send2robot(conn, state, qdot.tolist())
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

def getRobotAction(x_pos, goal_pose, state):
    xdot = np.zeros(6)
    # for i in range(3):
    xdot[:3] = goal_pose - x_pos
    # qdot_r = xdot2qdot(xdot, state)
    # qdot_r = np.clip(qdot_r, -0.1, 0.1)
    # return qdot_r
    xdot = np.clip(xdot, -0.07, 0.07)
    return xdot

def main():

    usernumber = sys.argv[1]
    filename = sys.argv[2]
    demonstration = []
    steptime = 0.1
    BETA = 1.
    PORT_robot = 8080
    scaling = 0.1

    print('[*] Connecting to low-level controller...')
    conn = connect2robot(PORT_robot)
    interface = Joystick()
    foldername = "user/user" + usernumber + "/" + "policy_blending/"

    record = False
    start_time = None
    last_time = None
    go2home(conn)
    print('[*] Waiting for start...')
    state = readState(conn)
    prior = [0.5, 0.5]
    belief = copy.deepcopy(prior)
    start_q = np.asarray(state["q"])
    start_pose = joint2pose(start_q)

    goals = [joint2pose(TAPE), joint2pose(NOTEPAD)]

    assist = False
    assist_start = 2.
    while True:

        state = readState(conn)
        u, start, mode, stop = interface.input()

        if stop:
            pickle.dump( demonstration, open( foldername + filename + ".pkl", "wb" ) )
            print(demonstration)
            print("[*] Done!")
            print("[*] I recorded this many datapoints: ", len(demonstration))
            return True
        
        if start and not record:
            record = True
            start_time = time.time()
            assist_time = time.time()
            print('[*] Recording the demonstration...')
            


        xdot_h = np.zeros(6)
        xdot_h[:3] = scaling * np.asarray(u)

        x_pos = joint2pose(state["q"])

        for i in range(len(goals)):
            dist2start = np.linalg.norm(x_pos - start_pose)
            dist2goal = np.linalg.norm(x_pos - goals[i])
            start2goal = np.linalg.norm(start_pose - goals[i])
            belief[i] = (np.exp(-BETA * (dist2start + dist2goal)) \
                            / np.exp(-BETA * start2goal)) * prior[i]

        belief = belief / np.sum(belief)
        alpha = max(belief)
        idx = np.argmax(belief)
        xdot_r = getRobotAction(x_pos, goals[idx], state)
        # print(qdot_r)
        # print(belief[0])

        if alpha > 0.75:
            alpha = 0.75
        
        curr_time = time.time()
        if record and curr_time - assist_time >= assist_start and not assist:    
            print("[*] Assistance started...")
            assist = True

        if assist:
            xdot = alpha * xdot_r + (1 - alpha) * xdot_h 
        else:
            xdot = xdot_h

        if x_pos[2] < 0.1 and xdot[2] < 0:
            xdot[2] = 0  

        qdot = xdot2qdot(xdot, state)
        qdot = qdot.tolist()
        
        if record and curr_time - start_time >= steptime:
            s = state["q"].tolist()
            elapsed_time = curr_time - assist_time
            qdot_h = xdot2qdot(xdot_h, state).tolist()
            qdot_r = xdot2qdot(xdot_r, state).tolist()
            demonstration.append([elapsed_time] + [s] + [qdot_h] + [qdot_r] + [float(alpha)])
            start_time = curr_time

        send2robot(conn, state, qdot)


if __name__ == "__main__":
    main()
