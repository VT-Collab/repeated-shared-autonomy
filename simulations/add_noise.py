import os
import copy
import numpy as np
import pickle
import matplotlib.pyplot as plt

def rescale(xi, n_waypoints, ratio):
    xi1 = copy.deepcopy(np.asarray(xi)[:, 0:6])
    total_waypoints = xi1.shape[0] * ratio
    index = np.linspace(0, total_waypoints, n_waypoints)
    index = np.around(index, decimals=0).astype(int)
    index[-1] = min([index[-1], xi1.shape[0]-1])
    return xi1[index,:]

def deform(xi, start, length, tau):
    xi1 = copy.deepcopy(np.asarray(xi))
    A = np.zeros((length+2, length))
    for idx in range(length):
        A[idx, idx] = 1
        A[idx+1,idx] = -2
        A[idx+2,idx] = 1
    R = np.linalg.inv(A.T @ A)
    U = np.zeros(length)
    gamma = np.zeros((length, 3))
    for idx in range(3):
        U[0] = tau[idx]
        gamma[:,idx] = R @ U
    end = min([start+length, xi1.shape[0]-1])
    xi1[start:end,:] += gamma[0:end-start,:]
    return xi1


savefolder = 'Noisy_Demos'
N_Waypoints = 10
N_Noisy = 5
folder = 'demos/Demos'


demos = []
D = []
N = []
pairs = pairs_clip = []
i = 0
for filename in os.listdir(folder):
    xi = pickle.load(open(folder + '/' + filename, "rb"))
    xi = np.asarray(xi)
    # demos.append(np.asarray(xi))
    ## DEMOS
    # D.append(rescale(xi, N_Waypoints, 1))
    # D.append(xi)
    ## ADDING NOISE
    j = 0
    for episode in range(N_Noisy):
        print(episode)
        savename = "demos/Noisy_Demos/" + str(i+1) + "_" + str(j+1) + ".pkl"
        length = 40 #np.random.randint(10,20)
        start = np.random.randint(0, len(xi)-20 - length)
        # start = 5
        tau = np.random.uniform([-0.01]*3, [0.01]*3)
        # tau1 = np.array([0,0,0,0])
        # tau = np.concatenate((tau,tau1), axis=0)
        xi1 = deform(xi, start, length, tau)
        # plt.plot(xi[:,0],xi[:,1])
        # plt.plot(xi1[:,0],xi1[:,1])
        # plt.show()
        # N.append(xi1)
        N_clip = xi1[start:start+length,:]
        D_clip = xi[start:start+length,:]
        p_clip = [D_clip,N_clip]
        # print(np.shape(xi1))
        # print(p_clip)
        pairs_clip.append([D_clip,N_clip])
        pickle.dump(xi1, open(savename, "wb"))
        # pairs.append([D[i],N[j]])
        # print(len(pairs_clip),len(pairs))

        # print(i,j)
        j = j+1
    i = i+1

    # print(len(pairs_clip))

# pickle.dump( N, open( savefolder + "/noisy.pkl", "wb" ) )
# pickle.dump( D, open( savefolder + "/demos.pkl", "wb" ) )
# pickle.dump( pairs, open( savefolder + "/pairs.pkl", "wb" ) )
# pickle.dump( pairs_clip, open( savefolder + "/clipped_pairs.pkl", "wb" ) )