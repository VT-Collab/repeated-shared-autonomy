import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
GOAL_H = np.asarray([0.50, 0.02665257, 0.25038403])

# def collect_final_states():
#     folder = "runs"
#     for filename in os.listdir(folder):
#         if filename[0] != ".":
#             demo = pickle.load(open(folder + "/" + filename, "rb"))
#             final_state = demo[-1][1]
#             print(filename[-10:-6])

def main():
    # collect_final_states()
    x = pickle.load(open("final_state_vae.pkl", "rb"))
    x_range = x[0]
    goal = np.column_stack((np.repeat(0.5,len(x_range)), x_range, np.repeat(0.25038403, len(x_range))))
    final_state = np.asarray(x[1])
    print(x)
    dists = np.linalg.norm(goal - final_state, axis = 1)
    # print(dists.tolist())
    plt.plot(x_range, dists)
    plt.show()

if __name__ == '__main__':
    main()