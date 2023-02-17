from cmath import nan
import numpy as np
import torch
import time
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint, NonlinearConstraint
import pickle

class TrajOpt(object):

    def __init__(self, args, home, goal, state_len, waypoints=10):
        # demos = pickle.load(open("demos/run_user_3/demo_test.pkl", "rb"))
        # self.demo = np.array(demos[0])
        # initialize trajectory
        self.args = args
        if self.args.task == 'cup':
            self.n_waypoints = waypoints
        else:
            self.n_waypoints = 4*waypoints
        self.state_len = state_len
        self.home = home
        self.goal = goal
        self.n_joints = len(self.home)
        self.provide_demos = True
        self.xi0 = np.zeros((self.n_waypoints, self.n_joints))
        for idx in range(self.n_waypoints):
            self.xi0[idx,:] = self.home + idx/(self.n_waypoints - 1.0) * (self.goal - self.home)
        self.xi0 = self.xi0.reshape(-1)

        # create start constraint and action constraint
        self.B = np.zeros((self.n_joints, self.n_joints * self.n_waypoints))
        for idx in range(self.n_joints):
            self.B[idx,idx] = 1
            self.G = np.zeros((self.n_joints, self.n_joints * self.n_waypoints))
        for idx in range(self.n_joints):
            self.G[self.n_joints-idx-1,self.n_waypoints*self.n_joints-idx-1] = 1
        self.lincon = LinearConstraint(self.B, self.home, self.home)
        self.lincon2 = LinearConstraint(self.G, self.goal, self.goal)
        if self.args.task == 'cup':
            self.nonlincon_lin = NonlinearConstraint(self.nl_function_lin, -0.08, 0.08)
            self.nonlincon_ang = NonlinearConstraint(self.nl_function_ang, -0.32, 0.32)
        else:
            self.nonlincon_lin = NonlinearConstraint(self.nl_function_lin, -0.02, 0.02)
            self.nonlincon_ang = NonlinearConstraint(self.nl_function_ang, -0.04, 0.04)


    # each action cannot move more than 1 unit
    def nl_function_lin(self, xi):
        xi = xi.reshape(self.n_waypoints, self.n_joints)
        actions = xi[1:, :2] - xi[:-1, :2]
        return actions.reshape(-1)

    def nl_function_ang(self, xi):
        xi = xi.reshape(self.n_waypoints, self.n_joints)
        actions = xi[1:, 2:] - xi[:-1, 2:6]
        return actions.reshape(-1)
    
    # trajectory reward function
    def reward(self, xi, args, task=None):
        self.orientation = np.array([3.09817065, -0.053698958, -0.01449647])
        self.laptop = np.array([0.5, -0.35])
        R = 0
        self.args = args
        self.task = task
        if self.task == 'table':
            for idx in range(len(xi )-1):
                R -= 1.*np.linalg.norm(xi[idx, 6:9] - xi[idx, :3]) * 2
                R -= 1.2*abs(xi[idx, 2])
                if xi[idx,2]<0.08:
                    R += 1*xi[idx, 2]
        elif self.task == 'cup':
            for idx in range(len(xi)-1):
            # z_tilt = xi[idx, 5:6]
                reward_pos = -np.linalg.norm(xi[idx, 6:12] - xi[idx, :6])
                reward_ang = -np.linalg.norm(self.orientation - xi[idx, 3:6])
                R += 1.0*reward_pos + 0.75*reward_ang
        elif self.task == 'laptop':
            for idx in range(len(xi)-1):
                dist_laptop = np.linalg.norm(xi[idx, :2] - self.laptop)
                reward_target = -np.linalg.norm(xi[idx, :3] - xi[idx, 6:9])
                reward_laptop = -np.max([0.0, 1.0-dist_laptop])
                R += 1*reward_target + 1.0*reward_laptop
        return R

    # true reward 
    def trajcost_true(self, xi):
        xi = xi.reshape(self.n_waypoints, self.n_joints)
        R = 0
        for idx in range(self.n_waypoints):
            R -= 1*np.linalg.norm(self.context - xi[idx, :6]) * 2
            R -= 1.2*abs(xi[idx, 2])
            if xi[idx,2]<0.25:
                R += 1*xi[idx, 2]
            # R -= abs(xi[idx, -1])
        return -R

    # trajectory cost function
    def trajcost(self, xi):
        xi = xi.reshape(self.n_waypoints, self.n_joints)
        states = np.zeros((self.n_waypoints, self.state_len))
        # target = joint2pose(self.context)
        for idx in range(self.n_waypoints):
            states[idx, :] = np.concatenate((xi[idx,:], self.context), axis = None)
        states = torch.FloatTensor(states)
        R = 1*self.reward_model.reward(states)
        if not self.flag:
            print(R)
            self.flag = True
        return -R

    # run the optimizer
    def optimize(self, reward_model=None, context=None, method='SLSQP'):
        # print(reward_model)
        self.flag = False
        self.reward_model = reward_model
        self.context = np.copy(context)
        start_t = time.time()
        if self.reward_model:
            res = minimize(self.trajcost, self.xi0, method=method, constraints={self.lincon, self.lincon2,  self.nonlincon_lin, self.nonlincon_ang}, options={'eps': 1e-3, 'maxiter': 10000})
        else:
            print("CORRECTION/DEMO PROVIDED")
            res = minimize(self.trajcost_true, self.xi0, method=method, constraints={self.lincon,  self.nonlincon_lin}, options={'eps': 1e-3, 'maxiter': 1000})
        
        xi = res.x.reshape(self.n_waypoints, self.n_joints)
        states = np.zeros((self.n_waypoints, self.state_len))
        for idx in range(self.n_waypoints):
            # xi[idx, 3] = np.clip(xi[idx, 3], 3.13, -3.13)
            states[idx, :] = np.concatenate((xi[idx, :], self.context), axis=None)
        # print(self.reward_model.reward(torch.FloatTensor(self.demo)))
        # print(self.reward_model.reward(torch.FloatTensor(states)))
        return states, res, time.time() - start_t

