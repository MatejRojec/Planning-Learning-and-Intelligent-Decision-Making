
# Activity nr.1 

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
# Activity nr.2

def noisy_policy(mdp, a, eps):
    policy = []
    for _ in range(len(mdp[0])):
        array = []
        for action in range(len(mdp[1])):
            if action == a:
                array.append(1-eps)
            else:
                array.append((eps/(len(mdp[1])-1)))
        policy.append(np.array(array))
    return np.array(policy)

pol_noiseless = noisy_policy(M, 2, 0.)

# Arbitrary state
s = 115 # State (8, 28, empty)

# Policy at selected state
print('Arbitrary state (from previous example):', M[0][s])
print('Noiseless policy at selected state (eps=0):', pol_noiseless[s, :])

# Noisy policy for action "Left" (action index: 2)
pol_noisy = noisy_policy(M, 2, 0.1)

# Policy at selected state
print('Noisy policy at selected state (eps=0.1):', np.round(pol_noisy[s, :], 2))

# Random policy for action "Left" (action index: 2)
pol_random = noisy_policy(M, 2, 0.75)

# Policy at selected state
print('Random policy at selected state (eps=0.75):', np.round(pol_random[s, :], 2))