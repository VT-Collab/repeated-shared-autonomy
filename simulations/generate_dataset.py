import numpy as np
import pickle


# input trajectory, output feature vector
def features(xi):
    dist2table = abs(0.1 - xi[-1][2]) / 0.6
    dist2goal = np.sqrt((0.8 - xi[-1][0])**2 + (-0.2 - xi[-1][1])**2) / np.sqrt(0.7**2 + 0.6**2)
    dist2obs_midpoint = abs(0.1 - xi[1][2]) / 0.6
    dist2obs_final = np.sqrt((0.6 - xi[-1][0])**2 + (0.1 - xi[-1][1])**2 + (0.1 - xi[-1][2])**2)  / np.sqrt(0.5**2 + 0.5**2 + 0.6**2)
    dist2obs = 0.5 * dist2obs_midpoint + 0.5 * dist2obs_final
    feature_vector = np.asarray([dist2table, dist2goal, dist2obs])
    return feature_vector

# input question, output vector with feature mean and variance
def Qfeatures(Q, n_questions=2, n_features=3):
    F = np.zeros((n_questions, n_features))
    for idx in range(n_questions):
        F[idx,:] = features(Q[idx])
    features_mean = np.mean(F, axis=0)
    features_std = np.std(F, axis=0)
    return np.concatenate((features_mean, features_std))



def main():

    dataset = []
    savename = 'data/questions.pkl'
    n_waypoints = 3
    n_questions = 5e2
    n_choices = 2

    for question in range(int(n_questions)):
        Q = []
        for q in range(n_choices):
            xi = np.zeros((n_waypoints, 3))
            # first waypoint fixed
            xi[0,:] = np.asarray([0.3, 0.9, 0.5])
            for waypoint in range(1, n_waypoints):
                if waypoint == 1:
                    # second waypoint above the ball
                    h = np.random.normal(0.4,0.2)
                    step = [0.6, 0.1, h]
                else:
                    # third waypoint random
                    step = np.random.multivariate_normal([0.35, 0.0, 0.4], np.diag([0.2, 0.2, 0.2]))
                xi[waypoint,:] = step
                # impose workspace limits
                if xi[waypoint, 0] < 0.1:
                    xi[waypoint, 0] = 0.1
                if xi[waypoint, 0] > 0.8:
                    xi[waypoint, 0] = 0.8
                if xi[waypoint, 1] < -0.4:
                    xi[waypoint, 1] = -0.4
                if xi[waypoint, 1] > 0.4:
                    xi[waypoint, 1] = 0.4
                if xi[waypoint, 2] < 0.1:
                    xi[waypoint, 2] = 0.1
                if xi[waypoint, 2] > 0.7:
                    xi[waypoint, 2] = 0.7
            # add trajectory to question
            Q.append(xi)
        Q.append(Qfeatures(Q))
        dataset.append(Q)

    pickle.dump(dataset, open(savename, "wb"))
    print("[*] I just saved this many questions: ", len(dataset))

    F = []
    for Q in dataset:
        F.append(Q[-1])
    F = np.asarray(F)
    print("mean features: ", np.mean(F, axis=0))
    print("stdv features: ", np.std(F, axis=0))

    mins, maxs = [] ,[]
    for idx in range(6):
        mins.append(np.min(F[:,idx]))
        maxs.append(np.max(F[:,idx]))
    print("min features: ", np.asarray(mins))
    print("max features: ", np.asarray(maxs))

if __name__ == "__main__":
    main()
