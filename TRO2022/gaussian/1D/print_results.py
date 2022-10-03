import pickle
import numpy as np
import matplotlib.pyplot as plt

GOAL_H = np.asarray([0.50, 0.02665257, 0.25038403])

def main():
	x = pickle.load(open("final_state.pkl", "rb"))
	x_range = x[0]
	goal = np.column_stack((np.repeat(0.5,len(x_range)), x_range, np.repeat(0.25038403, len(x_range))))
	final_state = np.asarray(x[1])
	dists = np.linalg.norm(goal - final_state, axis = 1)
	print(dists.tolist())
	plt.plot(x_range, dists)
	plt.show()

if __name__ == '__main__':
	main()