import numpy as np
import copy

d = 2

def generate_unit_vecs(d:int) -> list:
    empty = np.zeros(d, dtype=int)

    vectors = []
    for i in range(d):
        vec = copy.copy(empty)
        vec[i] = 1
        vectors.append(vec)
    return vectors

print(generate_unit_vecs(d)[:, 1])
    
