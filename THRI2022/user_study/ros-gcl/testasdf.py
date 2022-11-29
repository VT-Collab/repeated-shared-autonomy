import numpy as np
import itertools
from itertools import product

def generate_lookup_table(action_space_dim, values=[-1, 0, 1]):
    return np.array(list(product(values, repeat=action_space_dim)))

def generate_robot_action_table(axisValues=[-1, -0.5, 0, 0.5, 1]):
    a = []
    possibleAxes = list(product(axisValues, repeat=3))
    possibleLogics = list(product([0,1], repeat=3))
    for z in possibleAxes:
        for y in possibleLogics:
            a.append(list(z) + list(y))
    return a

a = generate_robot_action_table()
b = generate_robot_action_table(axisValues=[-1, 0, 1])
print(len(a), len(b)) # 1000, 216

