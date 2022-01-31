#!/usr/bin/env python

import rospy
import actionlib
import sys
import time
import numpy as np
import pygame
import pickle
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import copy
from collections import deque

from std_msgs.msg import Float64MultiArray, String

from robotiq_2f_gripper_msgs.msg import (
    CommandRobotiqGripperFeedback, 
    CommandRobotiqGripperResult, 
    CommandRobotiqGripperAction, 
    CommandRobotiqGripperGoal
)

from robotiq_2f_gripper_control.robotiq_2f_gripper_driver import (
    Robotiq2FingerGripperDriver as Robotiq
)

from controller_manager_msgs.srv import (
    SwitchController, 
    SwitchControllerRequest, 
    SwitchControllerResponse
)

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    GripperCommandAction,
    GripperCommandGoal,
    GripperCommand
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint
)
from sensor_msgs.msg import (
    JointState
)
from geometry_msgs.msg import(
    TwistStamped,
    Twist
)

HOME = [-1.571, -1.18997, -2.0167, -1.3992, 1.5407, 0.0]
##PUSHING TASKS
# END1 = [-1.571, -1.5710356871234339, -2.2512028853045862, -0.6734879652606409, 1.5406923294067383, 1.1984225238848012e-05]
# END2 = [-1.571, -2.1604259649859827, -1.5543845335589808, -0.8909743467914026, 1.5406923294067383, 0.00022770027862861753]

# END1 = [-0.800208870564596, -1.789145294819967, -2.005460564290182, -0.8623107115374964, 1.4747997522354126, 0.769659161567688]
# END2 = [-1.0631392637835901, -2.302997414265768, -1.2114885489093226, -1.119192902241842, 1.4926565885543823, 0.5063522458076477]

## OPENING TASKS
# END1 = [0.4409816563129425, -1.3135612646686, -1.551652733479635, -1.920682732258932, 1.4868240356445312, 2.016606092453003]
# END2 = [0.3971351385116577, -1.2446196714984339, -1.8191035429583948, -1.7183736006366175, 1.4836620092391968, 1.9727555513381958]
# END3 = [0.5832183361053467, -0.9510038534747522, -1.9657209555255335, -1.880350414906637, 1.4980816841125488, 2.1588282585144043]

# END1 = [-1.041595760975973, -1.8703201452838343, -1.036715332661764, -1.7285526434527796, 1.4909679889678955, 0.5278684496879578]
# END2 = [-1.0415361563311976, -1.8442123571978968, -1.2443326155291956, -1.5469968954669397, 1.4909679889678955, 0.5278684496879578]
# END3 = [-1.241943661366598, -1.7700746695147913, -1.3399184385882776, -1.511160675679342, 1.5078425407409668, 0.327643483877182]

## POURING TASKS
# END1 = [-1.3948305288897913, -1.7874186674701136, -1.1428821722613733, -1.6826689879046839, 1.522489070892334, 0.1752678006887436]
# END2 = [-1.4037821928607386, -1.9028056303607386, -1.0214160124408167, -2.451665703450338, 1.6441372632980347, 0.1319171041250229]

##SCOOPING TASKS
# END1 =  [-1.5709403196917933, -1.5436261335956019, -1.6257012526141565, -2.217304054890768, 1.5397224426269531, -0.0646899382220667]
# END2 =  [-1.5708087126361292, -1.9425666967975062, -1.7290032545672815, -1.7152803579913538, 1.539758324623108, -0.06461745897401983]
# END3 =  [-1.5708087126361292, -1.5888941923724573, -2.0001443068133753, -1.797647778187887, 1.5397343635559082, -0.06461745897401983]

# END1 = [-1.1448381582843226, -1.8685577551471155, -1.266087834035055, -2.1436808745013636, 1.7771531343460083, 0.3563782870769501]
# END2 = [-1.1448500792132776, -1.968987766896383, -1.629570786152975, -1.679755989705221, 1.7771531343460083, 0.35633033514022827]
# END3 = [-1.0765388647662562, -1.7488940397845667, -1.8186481634723108, -1.697524372731344, 1.81356680393219, 0.41587546467781067]

END1 =  [-1.5709522406207483, -1.16223651567568, -2.1478288809405726, -0.8249462286578577, 1.5407402515411377, 1.1984225238848012e-05]
END2 = [-1.1979368368731897, -1.6720030943499964, -2.222870651875631, -0.28124839464296514, 1.3416996002197266, 0.3186849057674408]
END3 = [-1.2630122343646448, -1.7763188521014612, -1.9321110884295862, -0.455576244984762, 1.375403642654419, 0.2618163824081421]

##CUTTING TASKS
# END1 = [-1.5969041029559534, -1.2059386412249964, -1.9943731466876429, -1.468011204396383, 1.6726858615875244, -1.4354031721698206]
# END2 = [-1.8035920302020472, -1.8345988432513636, -1.3963878790484827, -1.4591873327838343, 1.6795711517333984, -1.643170181904928]
# END3 = [-1.8035438696490687, -2.0096381346331995, -1.7424657980548304, -0.9381220976458948, 1.679523229598999, -1.6430981794940394]

# END1 = [-1.5556967894183558, -1.2045124212848108, -1.9810469786273401, -1.5529559294330042, 1.462847352027893, 1.5334954261779785]
# END2 =  [-1.0417879263507288, -1.8450630346881312, -1.3635037581073206, -1.5804713408099573, 1.490153431892395, 2.057386636734009]
# END3 = [-1.041811768208639, -2.007815663014547, -1.7200797239886683, -1.0612648169146937, 1.4901773929595947, 2.057410478591919]

STEP_SIZE_L = 0.15
STEP_SIZE_A = 0.2 * np.pi / 4
STEP_TIME = 0.01
DEADBAND = 0.1
MOVING_AVERAGE = 100

class JoystickControl(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.toggle = False
        self.action = None
        self.A_pressed = False
        self.B_pressed = False

    def getInput(self):
        pygame.event.get()
        toggle_angular = self.gamepad.get_button(4)
        toggle_linear = self.gamepad.get_button(5)
        self.A_pressed = self.gamepad.get_button(0)
        self.B_pressed = self.gamepad.get_button(1)
        if not self.toggle and toggle_angular:
            self.toggle = True
        elif self.toggle and toggle_linear:
            self.toggle = False
        return self.getEvent()

    def getEvent(self):
        z1 = self.gamepad.get_axis(1)
        z2 = -self.gamepad.get_axis(0)
        z3 = self.gamepad.get_axis(4)
        z = [z1, z2, z3]
        for idx in range(len(z)):
            if abs(z[idx]) < DEADBAND:
                z[idx] = 0.0
        stop = self.gamepad.get_button(7)
        B_pressed = self.gamepad.get_button(1)
        A_pressed = self.gamepad.get_button(0)
        return tuple(z), A_pressed, B_pressed, stop

    def getAction(self, z):
        if self.toggle:
            self.action = (0, 0, 0, STEP_SIZE_A * -z[1], STEP_SIZE_A * -z[0], STEP_SIZE_A * -z[2])
        else:
            self.action = (STEP_SIZE_L * -z[1], STEP_SIZE_L * -z[0], STEP_SIZE_L * -z[2], 0, 0, 0)

class TrajectoryClient(object):

    def __init__(self):
        # Action client for joint move commands
        self.client = actionlib.SimpleActionClient(
                '/scaled_pos_joint_traj_controller/follow_joint_trajectory',
                FollowJointTrajectoryAction)
        self.client.wait_for_server()
        # Velocity commands publishSTEP_SIZE_Ler
        self.vel_pub = rospy.Publisher('/joint_group_vel_controller/command',\
                 Float64MultiArray, queue_size=10)
        # Subscribers to update joint state
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_states_cb)
        # service call to switch controllers
        self.switch_controller_cli = rospy.ServiceProxy('/controller_manager/switch_controller',\
                 SwitchController)
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",\
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.base_link = "base_link"
        self.end_link = "wrist_3_link"
        self.joint_states = None
        self.robot_urdf = URDF.from_parameter_server()
        self.kdl_kin = KDLKinematics(self.robot_urdf, self.base_link, self.end_link)
        
        # Gripper action and client
        action_name = rospy.get_param('~action_name', 'command_robotiq_action')
        self.robotiq_client = actionlib.SimpleActionClient(action_name, \
                                CommandRobotiqGripperAction)
        self.robotiq_client.wait_for_server()
        # Initialize gripper
        goal = CommandRobotiqGripperGoal()
        goal.emergency_release = False
        goal.stop = False
        goal.position = 1.00
        goal.speed = 0.1
        goal.force = 5.0
        # Sends the goal to the gripper.
        self.robotiq_client.send_goal(goal)

        # store previous joint vels for moving avg
        self.qdots = deque(maxlen=MOVING_AVERAGE)
        for idx in range(MOVING_AVERAGE):
            self.qdots.append(np.asarray([0.0] * 6))

    def joint_states_cb(self, msg):
        try:
            if msg is not None:
                states = list(msg.position)
                states[2], states[0] = states[0], states[2]
                self.joint_states = tuple(states) 
        except:
            pass
    
    def switch_controller(self, mode=None):
        req = SwitchControllerRequest()
        res = SwitchControllerResponse()

        req.start_asap = False
        req.timeout = 0.0
        if mode == 'velocity':
            req.start_controllers = ['joint_group_vel_controller']
            req.stop_controllers = ['scaled_pos_joint_traj_controller']
            req.strictness = req.STRICT
        elif mode == 'position':
            req.start_controllers = ['scaled_pos_joint_traj_controller']
            req.stop_controllers = ['joint_group_vel_controller']
            req.strictness = req.STRICT
        else:
            rospy.logwarn('Unkown mode for the controller!')

        res = self.switch_controller_cli.call(req)

    def joint2pose(self):
        state = self.kdl_kin.forward(self.joint_states)
        xyz_lin = np.array(state[:,3][:3]).T
        xyz_lin = xyz_lin.tolist()
        R = state[:,:3][:3]
        beta = -np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
        gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        xyz_ang = [alpha, beta, gamma]
        xyz = xyz = np.asarray(xyz_lin[-1]).tolist() + np.asarray(xyz_ang).tolist()
        # print(xyz)
        # print(r.as_euler('xyz'))

        return xyz#self.kdl_kin.forward(self.joint_states)

    def pose2joint(self, pose):
        return self.kdl_kin.inverse(pose, self.joint_states)

    def xdot2qdot(self, xdot):
        J = self.kdl_kin.jacobian(self.joint_states)
        J_inv = np.linalg.pinv(J)
        return J_inv.dot(xdot)

    def send(self, xdot):
        qdot = xdot#self.xdot2qdot(xdot)
        # self.qdots.append(qdot)
        # qdot_mean = np.mean(self.qdots, axis=0).tolist()[0]
        cmd_vel = Float64MultiArray()
        cmd_vel.data = qdot
        self.vel_pub.publish(cmd_vel)

    def send_joint(self, pos, time):
        waypoint = JointTrajectoryPoint()
        waypoint.positions = pos
        waypoint.time_from_start = rospy.Duration(time)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names
        goal.trajectory.points.append(waypoint)
        goal.trajectory.header.stamp = rospy.Time.now()
        self.client.send_goal(goal)
        rospy.sleep(time)

    def actuate_gripper(self, pos, speed, force):
        Robotiq.goto(self.robotiq_client, pos=pos, speed=speed, force=force, block=True)
        return self.robotiq_client.get_result()


class RecordClient(object):

    def __init__(self):
        # Subscribers to update joint state
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_states_cb)
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",\
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.base_link = "base_link"
        self.end_link = "wrist_3_link"
        self.joint_states = None
        self.robot_urdf = URDF.from_parameter_server()
        self.kdl_kin = KDLKinematics(self.robot_urdf, self.base_link, self.end_link)
        self.script_pub = rospy.Publisher('/ur_hardware_interface/script_command', \
                                            String, queue_size=100)

        # Gripper action and client
        action_name = rospy.get_param('~action_name', 'command_robotiq_action')
        self.robotiq_client = actionlib.SimpleActionClient(action_name, \
                                CommandRobotiqGripperAction)
        self.robotiq_client.wait_for_server()
        # Initialize gripper
        goal = CommandRobotiqGripperGoal()
        goal.emergency_release = False
        goal.stop = False
        goal.position = 1.00
        goal.speed = 0.1
        goal.force = 5.0
        # Sends the goal to the gripper.
        self.robotiq_client.send_goal(goal)

    def joint_states_cb(self, msg):
        try:
            states = list(msg.position)
            states[2], states[0] = states[0], states[2]
            self.joint_states = tuple(states)
        except:
            pass

    def send_cmd(self, cmd):
        self.script_pub.publish(cmd)

    def actuate_gripper(self, pos, speed, force):
        Robotiq.goto(self.robotiq_client, pos=pos, speed=speed, force=force, block=True)
        return self.robotiq_client.get_result()


def main():
    demo_num = sys.argv[1]
    rospy.init_node("teleop")

    mover = TrajectoryClient()
    joystick = JoystickControl()
    recorder = RecordClient()

    start_time = time.time()
    rate = rospy.Rate(1000)

    print("[*] Initialized, Moving Home")
    mover.switch_controller(mode='position')
    mover.send_joint(HOME, 5.0)
    mover.client.wait_for_result()
    mover.switch_controller(mode='velocity')
    print("[*] Ready for joystick inputs")

    record = False
    flag1 = flag2 = True
    demonstration = []
    steptime  = 0.4
    action = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    # start_q = recorder.joint_states
    start_q = mover.joint2pose()
    filename = "demos/" + demo_num + ".pkl"

    while not rospy.is_shutdown():

        s_joint = recorder.joint_states
        s = mover.joint2pose()
        # print(np.asarray(s).tolist() + np.asarray(start_q).tolist())
        t_curr = time.time() - start_time
        axes, start, mode, stop = joystick.getInput()
        if stop or len(demonstration)>=60:
            pickle.dump(demonstration, open(filename, "wb"))
            print(demonstration)
            print("[*] Done!")
            print("[*] I recorded this many datapoints: ", len(demonstration))
            mover.switch_controller(mode='position')
            mover.send_joint(s_joint, 1.0)
            return True

        if start and not record:
            record = True
            start_time = time.time()
            print('[*] Recording the demonstration...')

        curr_time = time.time()
        if record and curr_time - start_time >= steptime:
            # print(s)
            demonstration.append(np.asarray(start_q).tolist() + np.asarray(s).tolist())
            print(len(demonstration))
            start_time = curr_time
      
        # joystick.getAction(axes)
        # action = np.array([-0.1,0.0,0.0,0.0,0.0,0.0])
        # mover.send(action)
        if np.linalg.norm(np.asarray(END1) - np.asarray(s_joint)) > 0.02 and flag1 and record:
            action = (np.asarray(END1) - np.asarray(s_joint))*0.5
            action = np.clip(action, -0.3, 0.3)
        elif np.linalg.norm(np.asarray(END2) - np.asarray(s_joint)) > 0.02 and flag2 and record:
            action = (np.asarray(END2) - np.asarray(s_joint))*0.5
            action = np.clip(action, -0.3, 0.3)
            flag1 = False
        elif record:
            action = (np.asarray(END3) - np.asarray(s_joint))*0.5
            action = np.clip(action, -0.3, 0.3)
            flag2 = False

        mover.send(action)
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass