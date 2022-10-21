import pickle
import numpy as np 

# note that the action space can be condensed to (axes, trans_mode, gripper_ac, slow_mode)
# axes is a (1,3)


# # recall this   R = np.mat([[1, 0, 0],
#                             [0, 1, 0],
#                             [0, 0, 1]])
#                 P = np.array([0, 0, -0.10])
#                 P_hat = np.mat([[0, -P[2], P[1]],
#                                 [P[2], 0, -P[0]],
#                                 [-P[1], P[0], 0]])
                
#                 axes = np.array(axes)[np.newaxis]
#                 trans_vel = scaling_rot * P_hat * R * axes.T
#                 rot_vel = scaling_rot * R * axes.T
#                 xdot_h[:3] = trans_vel.T[:]
#                 xdot_h[3:] = rot_vel.T[:]

def getAction(entry):
    axes = [0, 0, 0]
    slow_mode = int(entry["slow_mode"])
    trans_scaling = 0.1 + 0.1*slow_mode
    rot_scaling = 0.2 + 0.2*slow_mode
    if entry["trans_mode"]:
        axes = entry["xdot_h"][:3]
        # scaling is done, now scale from
        # [-1, -0.5, 0, 0.5, 1] for each input, rounding in intervals of 0.5
        axes = [0.5 * round(2 * ax/trans_scaling) for ax in axes]
    else: 
        # need to convert coordinate frames 
        axes = entry["xdot_h"][3:]
        axes = [0.5 * round(2 * ax/rot_scaling) for ax in axes]
    
    return axes

def main(fname="intent0/place_1.pkl"):
    data = pickle.load(open(fname, "rb"))
    for x in data:
        print(getAction(x), x["xdot_h"])


if __name__ == "__main__":
    main()

