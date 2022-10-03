import numpy as np
import pickle

def main():
    filename = "data/consolidated_data/task3_noise.pkl"
    data = pickle.load(open(filename, "rb"))
    ours = data[0]["007"]
    noassist = data[1]

    filename = "data/consolidated_data/task3.pkl"
    data = pickle.load(open(filename, "rb"))
    noassist = data[1]["s"]

    for traj in noassist:
        print(np.sum(traj))



if __name__ == '__main__':
    main()