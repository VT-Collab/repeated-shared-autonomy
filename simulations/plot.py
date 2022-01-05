import pickle
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

goals = pickle.load(open("goals4.pkl", "rb"))
traj1 = pickle.load(open("demos/1_1pkl", "rb"))
traj2 = pickle.load(open("demos/2_1pkl", "rb"))
# print(goals)

print(traj2)


