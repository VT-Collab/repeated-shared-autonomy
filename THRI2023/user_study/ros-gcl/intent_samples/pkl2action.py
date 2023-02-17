import pickle
import glob
import numpy as np 
from itertools import product
from geometry_msgs.msg import Twist
from utils import TrajectoryClient, convert_to_6d
# note that the action space can be condensed to (axes, trans_mode, gripper_ac, slow_mode)
# axes is a (1,3)
def generate_robot_action_table(axisValues=[-1, 0, 1]):
    a = []
    possibleAxes = list(product(axisValues, repeat=3))
    possibleLogics = [[1, 0, 1]]
    # possibleLogics = list(product([0], repeat=3))
    for z in possibleAxes:
        for y in possibleLogics:
            a.append(list(z) + list(y))
    return np.array(a)

def simpleTable():
    return np.array([[0, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 1],
            [-1, 0, 0, 1, 0, 1],
            [0, -1, 0, 1, 0, 1],
            [0, 0, -1, 1, 0, 1],])

def getActionOld(entry):
    axes = [0, 0, 0]
    slow_mode = int(entry["slow_mode"])
    trans_scaling = 0.2 - 0.1*slow_mode
    rot_scaling = 0.4 - 0.2*slow_mode
    # if entry["trans_mode"]:
    axes = entry["xdot_h"][:3]
    # scaling is done, now scale from
    # [-1, -0.5, 0, 0.5, 1] for each input, rounding in intervals of 0.5
    # axes = [0.5 * round(2 * ax/trans_scaling) for ax in axes]
    axes = [ax/ trans_scaling for ax in axes]
    # else: 
    #     # need to convert coordinate frames 
    #     axes = entry["xdot_h"][3:]
    #     # axes = [0.5 * round(2 * ax/rot_scaling) for ax in axes]
    #     axes = [ax / rot_scaling for ax in axes]
  
    # action = {"axes": axes, "trans_mode": entry["trans_mode"], 
    #     "gripper_ac": int(entry["gripper_ac"]), "slow_mode": slow_mode}
    # also need rotation about the x axis == roll 
    # note that we do not care about pitch or yaw
    roll = entry["xdot_h"][3] / rot_scaling
    action = axes + [roll] + [int(entry["trans_mode"])] + [int(entry["gripper_ac"])] + [1]
    return action

def getAction(x):
    return x["xdot_h"]

def getState(x):
    return x["curr_q"] + list(x["curr_pos_awrap"]) \
        + x["curr_gripper_pos"] + x["trans_mode"] + x["slow_mode"]

def indexOf(t, x):
    for idx, y in enumerate(t):
        if (x==y).all():
            return idx
    return 0
    # return indexOf(t, [0, 0, 0, 1, 0, 1])

def main(fname="downward/downward10.pkl"):
    data = pickle.load(open(fname, "rb"))
    newdata = []
    t = simpleTable()
    for x in data:
        action = getAction(x)
        actionIDX = indexOf(t, action)
        newdata.append((getState(x), actionIDX))
    return newdata 

def noiseData(x, nextX, noiselevel=0.0005, mover):
    curr_pos = np.asarray(x["curr_pos"])
    curr_q = np.asarray(x["curr_q"])
    curr_gripper_pos = np.asarray(x["curr_gripper_pos"])
    curr_trans_mode = np.asarray(x["trans_mode"])
    curr_slow_mode =np.asarray(x["slow_mode"])
    noise_pos = curr_pos.copy() + np.random.normal(0, noiselevel, len(curr_pos))

    next_pos = np.asarray(nextX["curr_pos"])
    next_q = np.asarray(nextX["curr_q"])
    next_gripper_pos = np.asarray(nextX["curr_gripper_pos"])
    # convert to twist for kdl_kin
    noise_pos_twist = Twist()
    noise_pos_twist.linear.x = noise_pos[0]
    noise_pos_twist.linear.y = noise_pos[1]
    noise_pos_twist.linear.z = noise_pos[2]
    noise_pos_twist.angular.x = noise_pos[3]
    noise_pos_twist.angular.y = noise_pos[4]
    noise_pos_twist.angular.z = noise_pos[5]

    noise_q = np.array(mover.pose2joint(noise_pos_twist, guess=curr_q))

    if None in noise_q:
        return None, None

    # Angle wrapping is a bitch
    noise_pos_awrap = convert_to_6d(noise_pos)
    # next_pos_awrap = convert_to_6d(next_pos)

    action = next_pos - noise_pos

    state = noise_q.tolist() + noise_pos_awrap.tolist() + curr_gripper_pos \
                + curr_trans_mode + curr_slow_mode
    return state, action


if __name__ == "__main__":
    strs = ["intent1", "intent2", "intent0"]

    mover = TrajectoryClient()
    noisesamples = 5
    lookahead = 5
    for s in strs:
        files = glob.glob("{}/*.pkl".format(s))
        newdata = []
        for f in files:
            data = pickle.load(open(f, "rb"))
            last = data[-1]
            lastState = getState(last)
            for i in range(len(data) - lookahead - 1):
                x = data[i]
                action = getAction(x)
                state = getState(x)
                newdata.append((state, action, [0], getState(data[i+1]), [0]))
                for _ in range(noisesamples):
                    state, action = noiseData(x, data[i+lookahead], mover)
                    if state is None: 
                        continue
                    newdata.append((state, action, [0], getState(data[i+1]), [0]))
                # newdata.append((state, action, [0], getState(data[i+1]), [0]))/
                # newdata.append([action, state, 0, getState(data[i+1]), 0])
        fname = "{}/all".format(s)
        print(newdata)
        pickle.dump(newdata, open(fname+"_actions_sac.pkl", "wb"))

