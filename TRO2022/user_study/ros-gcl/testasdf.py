import numpy as np
import itertools
from itertools import product

def generate_lookup_table(action_space_dim, values=[-1, 0, 1]):
    return np.array(list(product(values, repeat=action_space_dim)))

a = generate_lookup_table(3)
for x in a:
    print(x)
print(len(a))
