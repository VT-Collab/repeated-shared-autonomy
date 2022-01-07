import socket
import time
import numpy as np
import pickle
import pygame
import sys
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from train_cae import CAE
from train_classifier import Net
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear GPU
torch.cuda.empty_cache()


# Storing some important states
HOME = np.asarray([0.022643, -0.789077, -0.000277, -2.358605, -0.005446, 1.573151, -0.708887])
GOAL_D = np.asarray([0.50, 0.02665257, 0.25038403])
SIGMA_D = np.identity(3) * 0.0001

GOAL_H = np.asarray([0.50, 0.02665257, 0.25038403])
Q_MAX = [2.8, 1.7, 2.8, -0.75, 2.8, 3.7, 2.8]
Q_MIN = [-2.8, -1.7, -2.8, -3.0, -2.8, 0.0, -2.8]

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

def send2robot(conn, state, qdot, limit=1.0):
    qdot = np.asarray(qdot)
    scale = np.linalg.norm(qdot)
    if scale > limit:
        qdot *= limit/scale
    qdot = jointlimits(state, qdot)
    send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
    send_msg = "s," + send_msg + ","
    conn.send(send_msg.encode())

def jointlimits(state, qdot):
    for idx in range(7):
        if state["q"][idx] > Q_MAX[idx] and qdot[idx] > 0:
            qdot[idx] = 0.0
        elif state["q"][idx] < Q_MIN[idx] and qdot[idx] < 0:
            qdot[idx] = 0.0
    return qdot

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
    home = np.copy(HOME)
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
            qdot = np.clip(qdot, -0.3, 0.3)
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

def run(conn, interface, gx):
    filename = sys.argv[1]
    tasks = sys.argv[2]
    demos_savename = "demos/" + str(filename) + ".pkl"
    data_savename = "runs/" + str(filename) + ".pkl"
    cae_model = 'models/' + '0_cae_' + str(tasks)
    class_model = 'models/' + '0_class_' + str(tasks)
    model = Model(class_model, cae_model)
    # print('[*] Initializing recording...')
    demonstration = []
    data = []
    record = False
    translation_mode = True
    start_time = time.time()
    scaling_trans = 0.1
    scaling_rot = 0.2
    assist = False
    steptime = 0.1

    # print('[*] Main loop...')
    go2home(conn)
    # print('[*] Waiting for start...')
    state = readState(conn)
    start_q = state["q"].tolist()
    start_pose = joint2pose(start_q)
    assist_start = 1.
    # gstar = np.asarray([0.50, 0.02665257, 0.25038403])
    gstar = np.asarray([gx, 0.02665257, 0.25038403])
    goal = np.random.multivariate_normal(GOAL_D, SIGMA_D)
    start_time = time.time()
    assist_time = time.time()
    prev_pose = np.zeros(3)
    dist = 1
    qdot = [0.01]*7
    while True:

        state = readState(conn)
        s = state["q"].tolist()
        pose = joint2pose(state["q"])

        u, start, mode, stop = interface.input()
        if stop or np.sum(np.abs(qdot)) < 0.05:
            # pickle.dump( demonstration, open( demos_savename, "wb" ) )
            # pickle.dump(data, open( data_savename, "wb" ) )
            # print(data[0])
            # print("[*] Done!")
            # print("[*] I recorded this many datapoints: ", len(demonstration))
            return pose

        xdot_h = np.zeros(6)
        # xdot_h[:3] = scaling_trans * np.asarray(u)
        xdot_h[:3] =  np.clip((gstar - pose), -0.1, 0.1)
        x_pos = joint2pose(state["q"])

        alpha = model.classify(start_pose.tolist() + x_pos.tolist() + xdot_h[:3].tolist())
        alpha = min(alpha, 0.85)
        z = model.encoder(start_pose.tolist() + x_pos.tolist())
        # a_robot = model.decoder(z, x_pos.tolist())
        xdot_r = np.zeros(6)
        xdot_r[:3] = np.clip((goal - pose), -0.1, 0.1)
        
        curr_time = time.time()
        if curr_time - assist_time >= assist_start and not assist:    
            # print("[*] Assistance started...")
            assist = True
        # assist = False
        if assist:
            xdot = alpha * xdot_r + (1 - alpha) * xdot_h 
        else:
            xdot = xdot_h

        if x_pos[2] < 0.1 and xdot[2] < 0:
            xdot[2] = 0  

        qdot = xdot2qdot(xdot, state)
        qdot = qdot.tolist()
        
        if curr_time - start_time >= steptime:
            s = state["q"].tolist()
            demonstration.append(start_q + s)
            elapsed_time = curr_time - assist_time
            qdot_h = xdot2qdot(xdot_h, state).tolist()
            qdot_r = xdot2qdot(xdot_r, state).tolist()
            data.append([elapsed_time] + [s] + [qdot_h] + [qdot_r] + [float(alpha)])
            start_time = curr_time
            # print(float(alpha))
            # print(pose[1])
        try:
            send2robot(conn, state, qdot)
        except:
            qdot = [0] * 6
            print("[*] Hit joint limits, resetting")

def main():
    print('[*] Connecting to low-level controller...')
    PORT = 8080
    conn = connect2robot(PORT)
    interface = Joystick()
    x = []
    g_range = np.arange(0.3,0.7,0.001)
    for gx in g_range:
        final_x = []
        for _ in range(1):
            final_state = run(conn, interface, gx)
            poi = final_state[0]
            final_x.append(poi)
            print("gx: {} iter: {} xreal: {}".format(gx,_,poi))
        x.append(np.mean(final_x))
    # print(x)
    # 
    # x = [0.4736032798278854, 0.4750286233409006, 0.48367007498627934, 0.47202315588031146, 0.46812730712872813, 0.4795119719098426, 0.47977061061129994, 0.47180256239829077, 0.4732387529883536, 0.4824530637772193, 0.4772610003376989, 0.48009314874715936, 0.4776388488009218, 0.4783976156413198, 0.4749568073080815, 0.4756597634641242, 0.4778670424941048, 0.4770844028435327, 0.4759223832892262, 0.4776602349827009, 0.48139576539190543, 0.48177160360393884, 0.491070048261213, 0.4804736785319089, 0.4831222567222095, 0.4820002158215443, 0.48996431749401714, 0.49134944955756926, 0.49105344923367966, 0.49350206643738326, 0.4830699355380268, 0.4956224224751751, 0.4957846522244702, 0.48380058500181755, 0.48954824335277286, 0.4898516383772295, 0.4950980895515134, 0.4924940054306148, 0.4938438071652442, 0.49755601966792595, 0.49683088290364824, 0.49585419804081865, 0.500406113297328, 0.5043438809481946, 0.5017353622301775, 0.5024058676212756, 0.49655647097071737, 0.506228024610861, 0.51138583619931, 0.5148246957500597, 0.5052584906771032, 0.515397665163025, 0.5178653050111087, 0.5195180642829622, 0.5209926969430314, 0.5238592800322179, 0.5240842611899512, 0.5259289608930857, 0.5291208249350806, 0.5313387172501046, 0.5355231276528082, 0.541288032095177, 0.5448400483355009, 0.5447538540351275, 0.5495371984865768, 0.548919067551586, 0.5541621823912518, 0.556437482044139, 0.5547642515362919, 0.5639065864234876]
    pickle.dump([g_range, x], open("final_state.pkl", "wb"))
    plt.plot(g_range.tolist(), x)
    plt.show()

        

if __name__ == "__main__":
    main()
