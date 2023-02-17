# Standard imports
import rospy
import sys, time, pickle
import numpy as np
import argparse

from utils import TrajectoryClient, JoystickControl

"""
TODO
- instantiate final rotation(R, P, P_hat) outside the loop
"""

def record_demo(args):

    mover = TrajectoryClient()
    joystick = JoystickControl()

    rate = rospy.Rate(1000)

    rospy.loginfo("Initialized, Moving Home")
    mover.go_home()
    mover.reset_gripper()
    rospy.loginfo("Reached Home, waiting for input")
    
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

    scaling_trans = 0.2
    scaling_rot = 0.4

    record = False
    trans_mode = True
    slow_mode = False
    demo = []
    run_start = False

    while not rospy.is_shutdown():

        # get current states
        q = mover.joint_states
        curr_pos = mover.joint2pose()
        curr_gripper_pos = mover.robotiq_joint_state

        axes, gripper, mode, slow, start = joystick.getInput()

        if record and start:
            rospy.loginfo("Trajectory recorded")
            rospy.loginfo("Datapoints in trajectory: {}".format(len(demo)))
            mover.switch_controller(mode='position')
            mover.send_joint(q, 1.0)
            return demo

        # elif not record and start:
        #     record = True
        #     rospy.loginfo("Ready for joystick inputs")
        #     # wait for start to turn false and user starts moving robot
        #     while start or np.sum(axes):
        #         axes, gripper, mode, slow, start = joystick.getInput()
        #     rospy.loginfo("Recording...")
        #     start_time = time.time()
                # Wait for human to start moving
        while np.sum(np.abs(axes)) < 1e-3 and not run_start:
            axes, gripper, mode, slow, start = joystick.getInput()
            start_time = time.time()
            # assist_time = time.time()
        
        if not run_start:
            rospy.loginfo("Start received")
            run_start = True
            record = True
        
        # actuate gripper
        gripper_ac = 0
        xdot_h = np.zeros(6)
        curr_time = time.time()

        if record:
            if gripper and (mover.robotiq_joint_state > 0):
                mover.actuate_gripper(gripper_ac, 1, 0.)
                while gripper:
                    axes, gripper, mode, slow, start = joystick.getInput()
            elif gripper and (mover.robotiq_joint_state == 0):
                gripper_ac = 1
                mover.actuate_gripper(gripper_ac, 1, 0)
                while gripper:
                    axes, gripper, mode, slow, start = joystick.getInput()

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

            # Get input velocities for robot            
            if trans_mode: 
                xdot_h[:3] = scaling_trans * np.asarray(axes)

            elif not trans_mode:
                # change coord frame from robotiq gripper to tool flange
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

            # Save waypoint
            if curr_time - start_time >= step_time:
                # rospy.loginfo("State : {}".format(curr_pos))
                data = {}
                data["start_q"] = start_q
                data["start_pos"] = start_pos
                data["start_gripper_pos"] = start_gripper_pos
                data["curr_q"] = q
                data["curr_pos"] = curr_pos
                data["curr_gripper_pos"] = curr_gripper_pos
                data["trans_mode"] = trans_mode
                data["slow_mode"] = slow_mode
                data["xdot_h"] = xdot_h.tolist()
                data["gripper_ac"] = gripper_ac
                demo.append(data)
                start_time = curr_time

        qdot_h = mover.xdot2qdot(xdot_h)
        qdot_h = qdot_h.tolist()
        qdot_h = mover.compute_limits(qdot_h[0])

        mover.send(qdot_h[0])
        rate.sleep()

def main():
    rospy.init_node("record_demo")
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="save name")
    parser.add_argument("--save-loc", type=str, default="./user_data", help="save location for demo")
    parser.add_argument("--user", type=str, default="user0", help="user number")
    parser.add_argument("--store", action="store_true", help="use to store demo or discard")
    args = parser.parse_args()
    
    if args.store:
        assert args.name is not None, "Please enter a valid name to save the demo"

    demo = record_demo(args)

    if args.store:
        savename  = args.save_loc + "/" + args.user + "/" + args.name + ".pkl"
        pickle.dump(demo, open(savename, "wb"))
        rospy.loginfo("Saved demo as: {}".format(savename))
        rospy.loginfo("Sample Datapoint : {}".format(demo[0]))
    
    else:
        rospy.loginfo("Demo not saved.")

if __name__ == "__main__":
        try:
            main()
        except rospy.ROSInterruptException:
            pass