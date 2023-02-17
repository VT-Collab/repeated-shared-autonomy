from learn_intent import learn_intent
import argparse
import rospy
import numpy as np
import torch 
from cost import CostNN
from experts.PG import PG

def record_demos(foldername):
    pass


def beta(costf, state):
    # since dealing in state space, cannot use traj-opt like casa OG
    return min(1, 1/costf(state))

# For intent \in {intents}, if all beta_i(intent) < threshold, learn new intent
def casa_loop(args, epsilon=0.2):
    state = None
    state_shape = (12,) # curr_pos + curr_joint_states
    action_shape = 6 # xdot, trans, slow, gripperAC
    intents = [] # a list of policy-cost model pairs
    # load any intents that are of interest
    for premade_intent in args.premadeintents:
        costf = CostNN(state_shape[0] + 1)
        policy = PG(state_shape, n_actions=action_shape) 
        # load premade intents
        foldername = "intents/"+premade_intent
        costf.load_state_dict(torch.load(foldername)+"/cost_model")
        costf.eval()
        policy.load_state_dict(torch.load(foldername)+"/policy_model")
        policy.eval()
        intents.append({"cost": costf, "policy": policy})
    # learned intents initialized 
    while not rospy.is_shutdown():
        confidence = []
        for intent in intents:
            confidence.append(beta(intent["cost"], state))
        max_conf = max(confidence)
        if max_conf < epsilon:
            # record trajectories and learn new intent
            record_demos("intent"+str(len(confidence)))
            learn_intent("intent"+str(len(confidence)), True)
        else:
            # use current intent to assist 
            intent = intents[confidence.index(max_conf)]
            





if __name__ == "__main__":
    rospy.init_node("casa")
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", type=bool, help="whether to (re)generate the intent dataset for intentFolder", default=False)
    parser.add_argument("--premadeintents", nargs='+', default=[])

    args = parser.parse_args()
