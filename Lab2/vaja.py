
# Activity nr.1 

import numpy as np

def load_mdp(file, gamma):
    M = ()
    mdp = np.load(file)
    M += (tuple(mdp['X']), )
    M += (tuple(mdp['A']), )
    M += (tuple(mdp['P']), )
    M += (mdp['c'], )
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

# Activity nr. 3

def evaluate_pol(mdp, policy):
    identity = np.identity(len(mdp[0]))
    Ppi = np.zeros((len(mdp[0]),len(mdp[0])))
    for state1 in range(len(mdp[0])):
        for state2 in range(len(mdp[0])):
            for action in range(len(mdp[1])):
                Ppi[state1][state2] += mdp[2][action][state1][state2]*policy[state1][action]
    Cpi = []
    for state in range(len(mdp[0])):
        num = 0
        for action in range(len(mdp[1])):
            num += policy[state][action] * mdp[3][state][action]
        Cpi.append(num)
    Jpi = np.matmul(np.linalg.inv(identity - mdp[4] * Ppi), Cpi)
    return Jpi

pol_noisy = noisy_policy(M, 2, 0.1)
Jact2 = evaluate_pol(M, pol_noisy)

# Activety nr. 4





