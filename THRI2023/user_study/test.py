END1 = [0.4409816563129425, -1.3135612646686, -1.551652733479635, -1.920682732258932, 1.4868240356445312, 2.016606092453003]
END2 = [0.3971351385116577, -1.2446196714984339, -1.8191035429583948, -1.7183736006366175, 1.4836620092391968, 1.9727555513381958]
END3 = [0.5832183361053467, -0.9510038534747522, -1.9657209555255335, -1.880350414906637, 1.4980816841125488, 2.1588282585144043]

TASK = [END1, END2, END3]

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

    goal_idx = 0
    while not rospy.is_shutdown():
        q = np.asarray(mover.joint_states).tolist()

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
            mover.stop()
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

        goal= np.array(TASK[goal_idx])
        q = np.array(q)
        qdot_h = 0.2 * (goal - q)
        if np.linalg.norm(goal - q) < 0.1:
            if goal_idx < len(TASK)-1:
                    goal_idx += 1
            else:
                goal_idx = 0

        
            
        # qdot_h = mover.xdot2qdot(xdot_h)
        qdot_h = qdot_h.tolist()
        
        qdot_h = mover.compute_limits(qdot_h)
        mover.send(qdot_h[0])
        rate.sleep()
        # rospy.loginfo(mover.joint2pose())

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass