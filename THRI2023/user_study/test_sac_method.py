# Standard imports
import rospy
import sys, time, pickle, argparse
import numpy as np
from sac_gcl.gcl import GCL
import torch

# Imports from current directory
from utils import TrajectoryClient, JoystickControl, convert_to_6d
from glob import glob
# from model_utils import Model

np.set_printoptions(precision=2, suppress=True)

def convertAction(action, mover, slow_mode=True):
    axes = action[0:3]
    roll = action[3]
    trans_mode = action[4]
    gripper = action[5]
    # slow_mode = action[6]

    gripper_ac = 0
    scaling_trans = 0.2 - 0.1*slow_mode 
    scaling_rot = 0.4 - 0.2*slow_mode 
    
    xdot_h = np.zeros(6)
    xdot_h[:3] = scaling_trans * np.asarray(axes)
    xdot_h[3] = scaling_rot * roll
    return xdot_h        
    # qdot_h = mover.xdot2qdot(xdot_h)
    # qdot_h = qdot_h.tolist()
    
    # qdot_h = mover.compute_limits(qdot_h)
    # return np.transpose(qdot_h)


def run_test(args):

    mover = TrajectoryClient()
    joystick = JoystickControl()

    rate = rospy.Rate(1000)
    models = []
    filenames = ["intent0/sac_gcl_FINAL_intent0.pt", "intent1/sac_gcl_FINAL_intent1.pt","intent2/sac_gcl_FINAL_intent2.pt",]
    # for i in range(1, 40):
    #     i = 5 * i
    #     filenames.append("intent0/sac_gcl_{}intent0.pt".format(i))
    #     filenames.append("intent1/sac_gcl_{}intent1.pt".format(i))
    #     filenames.append("intent2/sac_gcl_{}intent2.pt".format(i))
    # for i in range(args.models_to_load):

    # for i in range(len(filenames)):    
    for i in range(args.num_intents):

        tmp = GCL()
        tmp.load_model_filename(filenames[i])
        models.append(tmp) 
    

    rospy.loginfo("Initialized, Moving Home")
    mover.go_home()
    mover.reset_gripper()
    rospy.loginfo("Reached Home, waiting for start")

    start_pos = None
    start_q = None
    while start_pos is None:
        start_pos = mover.joint2pose()
        start_q = mover.joint_states

    start_gripper_pos = None
    while start_gripper_pos is None:
        start_gripper_pos = mover.robotiq_joint_state

    step_time = 0.1
    start_time = time.time()
    assist_time = time.time()

    scaling_trans = 0.2
    scaling_rot = 0.4

    trans_mode = True
    slow_mode = False
    traj = []
    alphas = []
    
    assist = False
    assist_start = 1.0
    
    while not rospy.is_shutdown():

        q = np.asarray(mover.joint_states).tolist()
        curr_pos = mover.joint2pose()
        curr_gripper_pos = mover.robotiq_joint_state

        axes, gripper, mode, slow, start = joystick.getInput()
        if start:
            # pickle.dump(demonstration, open(filename, "wb"))
            mover.switch_controller(mode='position')
            mover.send_joint(q, 1.0)
            return 1
            
        # switch between translation and rotation
        if mode:
            trans_mode = not trans_mode
            rospy.loginfo("Translation Mode: {}".format(trans_mode))
            while mode:
                axes, gripper, mode, slow, start = joystick.getInput()
        
        # Toggle speed of robot
        if slow:
            slow_mode = not slow_mode
            rospy.loginfo("Slow Mode: {}".format(trans_mode))
            while slow:
                axes, gripper, mode, slow, start = joystick.getInput()
        
        if slow_mode:
            scaling_trans = 0.1
            scaling_rot = 0.2
        else:
            scaling_trans = 0.2
            scaling_rot = 0.4
            
        xdot_h = np.zeros(6)
        if trans_mode: 
            xdot_h[:3] = scaling_trans * np.asarray(axes)
        elif not trans_mode:
            R = np.mat([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1]])
            P = np.array([0, 0, -0.10])
            P_hat = np.mat([[0, -P[2], P[1]],
                            [P[2], 0, -P[0]],
                            [-P[1], P[0], 0]])
            
            axes = np.array(axes)[np.newaxis]
            trans_vel = scaling_rot * P_hat * R * axes.T
            rot_vel = scaling_rot * R * axes.T
            xdot_h[:3] = trans_vel.T[:]
            xdot_h[3:] = rot_vel.T[:]
            
        qdot_h = mover.xdot2qdot(xdot_h)
        qdot_h = qdot_h.tolist()[0]

        qdot_r = np.zeros(6).tolist()

        curr_time = time.time()
        if curr_time - start_time >= step_time:
            traj.append(start_pos + curr_pos + qdot_h)
            start_time = curr_time

        if traj:

            gripper_ac = 0
            if gripper and (mover.robotiq_joint_state > 0):
                mover.actuate_gripper(gripper_ac, 1, 0.)
                while gripper:
                    axes, gripper, mode, slow, start = joystick.getInput()

            elif gripper and (mover.robotiq_joint_state == 0):
                gripper_ac = 1
                mover.actuate_gripper(gripper_ac, 1, 0)
                while gripper:
                    axes, gripper, mode, slow, start = joystick.getInput()

            curr_pos_awrap = convert_to_6d(curr_pos)

            d = curr_pos_awrap.tolist()[:6] + q
            alphas = [model.cost_f(torch.FloatTensor(d)).detach().numpy()[0] for model in models]
            # rospy.loginfo("d: {}".format(d))
            rospy.loginfo("confidence: {}".format(alphas))
            modelToUse = models[alphas.index(min(alphas))]
            alpha = min(0.6, np.exp(min(alphas)))
            alpha = max(0.0, alpha)
            a_robot = modelToUse.select_action(d)
            # alpha = min(alpha, 0.6)
            # alphas.append(alpha)
            # alpha = 0.4
            a_robot = np.concatenate((a_robot, [1, 0, 1]))
            a_robot = convertAction(a_robot, mover)
            # z = model.encoder(q + curr_pos_awrap.tolist() + [curr_gripper_pos] + [float(trans_mode), float(slow_mode)])
            # a_robot = model.decoder(z, q + curr_pos_awrap.tolist() + [curr_gripper_pos] + [float(trans_mode), float(slow_mode)])

            a_robot = mover.xdot2qdot(a_robot)
            qdot_r = 2. * a_robot
            qdot_r = qdot_r.tolist()[0]

        if curr_time - assist_time >= assist_start and not assist:
            print("Assistance Started...")
            assist = True

        if assist:
            qdot = (alpha * 1.0 * np.asarray(qdot_r) + (1-alpha) * np.asarray(qdot_h))
            qdot = np.clip(qdot, -0.3, 0.3)
            qdot = qdot.tolist()
        else:
            qdot = qdot_h

        mover.send(qdot)
        rate.sleep()

def main():
    rospy.init_node("test_method_old")

    parser = argparse.ArgumentParser()
    # parser.add_argument("--models-to-load", type=int, help="cae model name", default=3)
    parser.add_argument("--num_intents", type=int, help="number of intents to use", default=3)
    args = parser.parse_args()
    
    run_test(args)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass 