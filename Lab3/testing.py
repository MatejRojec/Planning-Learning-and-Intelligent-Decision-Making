import numpy as np


def load_pomdp(file, gamma):
    pomdp = np.load(file)
    X = tuple(pomdp['X'])
    A = tuple(pomdp['A'])
    Z = tuple(pomdp['Z'])
    P = tuple(pomdp['P'])
    O = tuple(pomdp['O'])
    c = pomdp['c']

    return (S, A, Z, P, O, c, gamma)

