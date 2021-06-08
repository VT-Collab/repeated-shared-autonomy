import socket
import time
import numpy as np
import pickle
import pygame
import sys

import torch
import torch.nn as nn
import cv2
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, xyxy2xywh
from yolov5.utils.datasets import letterbox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear GPU memory from previous runs
torch.cuda.empty_cache()

from train_model_variation import CAE


class YOLODetector(nn.Module):
    def __init__(self, load_path):
        super(YOLODetector, self).__init__()
        self.yolo = attempt_load(load_path, map_location=device)
        self.n_classes = len(self.yolo.names)
        self.to(device)

    def get_object_features(self, img):
        pred, _ = self.yolo(img, augment=False)
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.25, agnostic=True)[0]
        if pred is None:
            return None
        obj_feats = []
        for object in pred:
            cls = int(object[5])
            xywh = (xyxy2xywh(object[:4].view(1, 4) / img.shape[-1]))
            obj_feat = torch.cat([xywh[0, :2],
                        torch.nn.functional.one_hot(torch.tensor(cls),
                        num_classes=self.n_classes).float().to(device)])
            obj_feats.append(obj_feat.tolist())
        return obj_feats


class Model(object):

    def __init__(self, modelname):
        self.model = CAE()
        model_dict = torch.load(modelname, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval

    def encoder(self, context):
        context = torch.FloatTensor(context)
        z_mean, z_log_var = self.model.encoder(context)
        return z_mean.tolist(), torch.exp(0.5*z_log_var).tolist()

    def decoder(self, z, s):
        z_tensor = torch.FloatTensor(z + s)
        a_predicted = self.model.decoder(z_tensor)
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
        START_pressed = self.gamepad.get_button(7) and (curr_time - self.lastpress > self.timeband)
        if A_pressed or START_pressed:
            self.lastpress = curr_time
        return [dx, dy, dz], A_pressed, START_pressed


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



def main():

    image_width = 640
    image_heigth = 640

    print('[*] Loading Pre-Trained YOLO-v5 Detector...')
    yolo_model = "yolov5/weights/best.pt"
    detector = YOLODetector(yolo_model)

    print('[*] Setting up WebCam Feed...')
    vc = cv2.VideoCapture(0)
    vc.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    assert vc.isOpened(), "Webcam doesn't appear to be online!"

    print('[*] Extracting Object Features...')
    while True:
        # read image from webcam
        all_ok, img_raw = vc.read()
        assert all_ok, "Webcam stopped working?"
        # preprocess image for yolov5
        img_raw = letterbox(img_raw, new_shape=(image_width, image_heigth))[0]
        img = img_raw[:, :, ::-1].copy()                # BGR to RGB
        img = img / 256.0                               # Normalize to [0, 1]
        img = np.transpose(img, (2, 0, 1))              # Channel axis first
        img = torch.FloatTensor([img]).to(device)
        # get object position and class
        obj_feat = detector.get_object_features(img)
        if obj_feat is not None:
            break

    print('[*] Visualizing the Object Features...')
    context = [0] * 4
    for object in obj_feat:
        if object[3]:
            print('[*] I see a VT notepad')
            obj_xy = (int(object[0] * image_width), int(object[1] * image_heigth))
            context[0] = object[0] - 0.5
            context[1] = object[1] - 0.5
            cv2.circle(img_raw, obj_xy, 25, (0, 0, 255), 10)
        elif object[4]:
            print('[*] I see a can of soup')
            obj_xy = (int(object[0] * image_width), int(object[1] * image_heigth))
            context[2] = object[0] - 0.5
            context[3] = object[1] - 0.5
            cv2.circle(img_raw, obj_xy, 25, (0, 255, 0), 10)
    print('[*] Here is the current context: ', context)
    cv2.imshow('frame', img_raw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    print('[*] Connecting to low-level controller...')
    PORT = 8080
    conn = connect2robot(PORT)
    interface = Joystick()
    model = Model("models/vae_model_r")

    print('[*] Initializing recording...')
    BETA = 1.0  # human in charge
    action_scale = 0.1
    state = readState(conn)
    home = context + state["q"].tolist()

    print('[*] Main loop...')
    while True:

        state = readState(conn)
        s = context + state["q"].tolist()

        z_mean, z_std = model.encoder(home + state["q"].tolist())
        z_mean = z_mean[0]
        z_std = z_std[0]

        qdot_r = model.decoder([z_mean], s)

        u, start, stop = interface.input()
        if stop:
            print("[*] Done!")
            return True
        if start:
            BETA = 0.0
            print('[*] Robot taking over...')

        xdot = [0]*6
        xdot[0] = action_scale * u[0]
        xdot[1] = action_scale * u[1]
        xdot[2] = action_scale * u[2]

        qdot_h = xdot2qdot(xdot, state)
        qdot = (1.0 - BETA) * qdot_r + BETA * qdot_h
        send2robot(conn, qdot)


if __name__ == "__main__":
    main()
