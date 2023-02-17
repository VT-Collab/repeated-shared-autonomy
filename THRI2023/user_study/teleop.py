import rospy
import sys
import numpy as np

from waypoints import HOME
from utils import TrajectoryClient, JoystickControl

def main():
    rospy.init_node("teleop")

    scaling_trans = 0.1
    mover = TrajectoryClient()
    joystick = JoystickControl()

    rate = rospy.Rate(1000)

    rospy.loginfo("Initialized, Moving Home")
    mover.go_home()
    mover.reset_gripper()
    rospy.loginfo("Reached Home, waiting for start")

    trans_mode = True
    slow_mode = False
    scaling_trans = 0.2
    scaling_rot = 0.4

    while not rospy.is_shutdown():
        axes, gripper, mode, slow, start = joystick.getInput()

        # Toggling the gripper
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

        # End teleop
        if start:
            return True
      
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

        # Slow down robot when X pressed
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
            
        qdot_h = mover.xdot2qdot(xdot_h)
        qdot_h = qdot_h.tolist()
        
        qdot_h = mover.compute_limits(qdot_h)
        mover.send(qdot_h[0])
        rate.sleep()
        rospy.loginfo(mover.joint_states)

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass