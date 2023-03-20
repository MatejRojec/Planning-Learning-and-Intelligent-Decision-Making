import numpy as np

def load_mdp(file, gamma):
    M = ()
    mdp_info = np.load(file)
    M += (tuple(mdp_info['X']), )
    M += (tuple(mdp_info['A']), )
    M += (tuple(mdp_info['P']), )
    M += (mdp_info['c'], )
    M += (gamma, )
    return M

M = load_mdp('garbage-big.npz', 0.99)
print(M[3])