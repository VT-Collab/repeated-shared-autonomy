# Standard imports
import rospy, actionlib, copy
import sys, time, pygame, pickle, tf, math
import numpy as np


# Imports from current directory
from utils import TrajectoryClient, JoystickControl
from model_utils import Model
from waypoints import HOME

np.set_printoptions(precision=2)
def get_rotation_mat(euler):
    R_x = np.mat([[1, 0, 0],
                  [0, np.cos(euler[0]), -np.sin(euler[0])],
                  [0, np.sin(euler[0]), np.cos(euler[0])]])

    R_y = np.mat([[np.cos(euler[1]), 0, np.sin(euler[1])],
                  [0, 1, 0],
                  [-np.sin(euler[1]), 0, np.cos([1])]])

    R_z = np.mat([[np.cos(euler[2]), -np.sin(euler[2]), 0],
                  [np.sin(euler[2]), np.cos(euler[2]), 0],
                  [0, 0, 1]])
    R = R_x * R_y * R_z
    return R
def run_test():

    mover = TrajectoryClient()
    joystick = JoystickControl()

    rate = rospy.Rate(1000)

    rospy.loginfo("Initialized, Moving Home")
    mover.go_home()
    mover.reset_gripper()
    rospy.loginfo("Reached Home, waiting for start")

    cae_model = 'models/' + 'cae'
    class_model = 'models/' + 'class'
    model = Model(class_model, cae_model)

    traj = []
    record = False
    step_time = 0.1
    trans_mode = True
    slow_mode = False
    scaling_trans = 0.2
    scaling_rot = 0.4
    start_pos = None
    while start_pos is None:
        start_pos = mover.joint2pose()
        # start_pos_cart = mover.joint2pose()
        # start_pos = np.zeros(7)
        # start_pos[:3] = start_pos_cart[:3]
        # start_pos[3:] = tf.transformations.quaternion_from_euler(start_pos_cart[3], start_pos_cart[4], start_pos_cart[5])
        # start_pos = start_pos.tolist()
    start_gripper_pos = None
    while start_gripper_pos is None:
        start_gripper_pos = mover.robotiq_joint_state
    start_time = time.time()
    alphas = []
    assist = False
    assist_start = 1.0
    start_time = time.time()
    assist_time = time.time()
    prev_pos = np.copy(start_pos)
    while not rospy.is_shutdown():

        q = np.asarray(mover.joint_states).tolist()
        curr_pos = mover.joint2pose()
        # curr_pos_cart = mover.joint2pose()
        # curr_pos = np.zeros(7)
        # curr_pos[:3] = curr_pos_cart[:3]
        # curr_pos[3:] = tf.transformations.quaternion_from_euler(curr_pos_cart[3], curr_pos_cart[4], curr_pos_cart[5])
        # curr_pos = curr_pos.tolist()
        curr_gripper_pos = mover.robotiq_joint_state

        axes, gripper, mode, slow, start = joystick.getInput()
        if start:
            # pickle.dump(demonstration, open(filename, "wb"))
            # # print(demonstration)
            # print("[*] Done!")
            # print("[*] I recorded this many datapoints: ", len(demonstration))
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
            # print(traj)
            # t = torch.Tensor(traj).unsqueeze(0)
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

            # print(t)
            d = np.array(start_pos + [0.] + curr_pos +[curr_gripper_pos] + xdot_h.tolist() + [gripper_ac])
            alpha = model.classify([d.tolist()])
            # alpha = model.classify([start_pos + s])
            # alpha = model.classify([s])
            # print(alpha)
            alpha = min(alpha, 0.6)
            if alpha < 0.28:
                alpha = 0.
            alphas.append(alpha)
            alpha = 0.4
            # z = model.encoder(start_pos + [0.0] + curr_pos + [curr_gripper_pos] + [float(trans_mode), float(slow_mode)])
            # history = noise_q + noise_pos + curr_gripper_pos + trans_mode + slow_mode
            # for i, ang in enumerate(curr_pos[3:]):
            #     ang = math.fmod(ang, np.pi)
            #     if ang < 0:
            #         ang += np.pi
            #     if ang < 0.15:
            #         ang += np.pi
            #     curr_pos[3+i] = ang
            # print(curr_pos[3:])
            curr_pos_awrap = np.zeros(9)
            curr_pos_awrap[:3] = curr_pos[:3]
            curr_pos_awrap[3:] = get_rotation_mat(curr_pos[3:]).flatten('F')[0,:6]
            # for i in range(3):
            #     curr_pos_awrap[3+i] = np.sin(curr_pos[3+i])
            #     curr_pos_awrap[6+i] = np.cos(curr_pos[3+i])
                
            z = model.encoder(q + curr_pos_awrap.tolist() + [curr_gripper_pos] + [float(trans_mode), float(slow_mode)])
            a_robot = model.decoder(z, q + curr_pos_awrap.tolist() + [curr_gripper_pos] + [float(trans_mode), float(slow_mode)])
            # print(np.array(q))
            # z = model.encoder(q + [curr_gripper_pos] + [float(trans_mode), float(slow_mode)])
            # a_robot = model.decoder(z, q + [curr_gripper_pos] + [float(trans_mode), float(slow_mode)])
            # R = np.mat([[1, 0, 0],
            #             [0, 1, 0],
            #             [0, 0, 1]])
            # P = np.array([0, 0, -0.10])
            # P_hat = np.mat([[0, -P[2], P[1]],
            #                 [P[2], 0, -P[0]],
            #                 [-P[1], P[0], 0]])
                
            # a_robot_trans = np.array(a_robot[:3])[np.newaxis]
            # a_robot_rot = np.array(a_robot[3:])[np.newaxis]
            # trans_vel = scaling_trans * R * a_robot_trans.T + scaling_rot * P_hat * R * a_robot_rot.T
            # rot_vel = scaling_rot * R * a_robot_rot.T
            # xdot_r = np.zeros(6)
            # xdot_r[:3] = trans_vel.T[:]
            # xdot_r[3:] = rot_vel.T[:]
            # a_robot[4] = -a_robot[4]
            # print(a_robot[3:])
            # a_robot[3:6] = np.zeros(3)
            # xdot_r = np.zeros(6)
            # xdot_r[:3] = a_robot[:3]
            # for i in range(3):
            #     xdot_r[3+i] = np.arctan2(a_robot[3+i], a_robot[6+i])
            # print(xdot_r[3:])
            # a_robot = mover.xdot2qdot(xdot_r)
            # xdot_r = mover.xdot2qdot(xdot_r)
            # a_robot[:3] = -a_robot[:3]
            # if not trans_mode:
            #     print("pos: {}, ac: {}".format(np.array(curr_pos), a_robot))
            a_robot = mover.xdot2qdot(a_robot)
            qdot_r = 2. * a_robot
            # qdot_r = 1 * xdot_r
            qdot_r = qdot_r.tolist()[0]

        if curr_time - assist_time >= assist_start and not assist:
            print("Assistance Started...")
            assist = True

        if assist:
            # alpha = 0
            qdot = (alpha * 1.0 * np.asarray(qdot_r) + (1-alpha) * np.asarray(qdot_h))
            qdot = np.clip(qdot, -0.3, 0.3)
            qdot = qdot.tolist()
            # qdot = qdot[0]
        else:
            qdot = qdot_h

        mover.send(qdot)
        prev_pos = list(curr_pos)
        rate.sleep()

def main():
    rospy.init_node("test_method_old")
    run_test()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass 