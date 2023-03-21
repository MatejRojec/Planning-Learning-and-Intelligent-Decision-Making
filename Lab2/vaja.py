# Activity nr.1

import numpy as np

np.random.seed(42)


def load_mdp(file, gamma):
    M = ()
    mdp = np.load(file)
    M += (tuple(mdp['X']),)
    M += (tuple(mdp['A']),)
    M += (tuple(mdp['P']),)
    M += (mdp['c'],)
    M += (gamma,)
    return M


M = load_mdp('garbage-big.npz', 0.99)


# Activity nr.2

def noisy_policy(mdp, a, eps):
    policy = []
    for _ in range(len(mdp[0])):
        array = []
        for action in range(len(mdp[1])):
            if action == a:
                array.append(1 - eps)
            else:
                array.append((eps / (len(mdp[1]) - 1)))
        policy.append(np.array(array))
    return np.array(policy)


pol_noiseless = noisy_policy(M, 2, 0.)


# Activity nr. 3

def evaluate_pol(mdp, policy):
    identity = np.identity(len(mdp[0]))
    Ppi = np.zeros((len(mdp[0]), len(mdp[0])))
    for state1 in range(len(mdp[0])):
        for state2 in range(len(mdp[0])):
            for action in range(len(mdp[1])):
                Ppi[state1][state2] += mdp[2][action][state1][state2] * policy[state1][action]
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

import time


def value_iteration(mdp):
    X, A, p, c, gamma = mdp

    J = np.zeros((len(X), 1))
    err = 1

    niter = 0
    start = time.time()

    Q = np.zeros((len(X), len(A)))

    while err > 1e-8:
        for action in range(len(mdp[1])):
            Q[:, action, None] = c[:, action, None] + gamma * p[action].dot(J)
        Jnew = np.min(Q, axis=1, keepdims=True)
        err = np.linalg.norm(Jnew - J)
        niter += 1
        J = Jnew

    print("Execution time: ", round(time.time() - start, 3), " seconds")
    print("N. iterations: ", niter)

    return J


def policy_iteration(mdp):
    X, A, p, c, gamma = mdp

    # Initialize uniform policy
    policy = np.ones((len(X), len(A))) / len(A)

    V = np.zeros((len(X), 1))
    quit = False
    niter = 0

    start = time.time()
    Q = np.zeros((len(X), len(A)))

    while not quit:

        J = evaluate_pol(mdp, policy)
        print(J.shape)

        for action in range(len(A)):
            Q[:, action, None] = c[:, action, None] + gamma * p[action].dot(J)

        Qmin = np.min(Q, axis=1, keepdims=True)
        p_new = np.isclose(Q, Qmin, atol=1e-10, rtol=1e-10).astype(int)
        p_new = p_new / p_new.sum(axis=1, keepdims=True)
        quit = (policy == p_new).all()

        policy = p_new
        niter += 1

    print("Execution time: ", round(time.time() - start, 3), " seconds")
    print("N. iterations: ", niter)

    return policy


NRUNS = 100


def simulate(mdp, policy, x0, length):
    global NRUNS
    X, A, p, c, gamma = mdp

    # Initialize the state
    cost = 0

    for run in range(NRUNS):
        state = x0
        for t in range(length):
            # Select the action according to the policy
            action = np.random.choice(len(A), p=policy[state, :])

            # Compute the cost
            cost += c[state, action] * (gamma ** t)

            # Sample the next state
            state = np.random.choice(len(X), p=p[action][state])

    return cost / NRUNS


Jopt = value_iteration(M)

print('\nDimensions of cost-to-go:', Jopt.shape)

print('\nExample values of the optimal cost-to-go:')

s = 115  # State (8, 28, empty)
print('\nCost to go at state %s:' % M[0][s], Jopt[s])

s = 429  # (0, None, loaded)
print('Cost to go at state %s:' % M[0][s], Jopt[s])

s = 239  # State (18, 18, empty)
print('Cost to go at state %s:' % M[0][s], Jopt[s])

print('\nIs the policy from Activity 2 optimal?', np.all(np.isclose(Jopt, Jact2)))

popt = policy_iteration(M)

print('\nDimension of the policy matrix:', popt.shape)

print('\nExamples of actions according to the optimal policy:')

# Select random state, and action using the policy computed
s = 115  # State (8, 28, empty)
a = np.random.choice(len(M[1]), p=popt[s, :])
print('Policy at state %s: %s' % (M[0][s], M[1][a]))

# Select random state, and action using the policy computed
s = 429  # (0, None, loaded)
a = np.random.choice(len(M[1]), p=popt[s, :])
print('Policy at state %s: %s' % (M[0][s], M[1][a]))

# Select random state, and action using the policy computed
s = 239  # State (18, 18, empty)
a = np.random.choice(len(M[1]), p=popt[s, :])
print('Policy at state %s: %s' % (M[0][s], M[1][a]))

# Verify optimality of the computed policy

print('\nOptimality of the computed policy:')

Jpi = evaluate_pol(M, popt)
print('- Is the new policy optimal?', np.all(np.isclose(Jopt, Jpi)))

if not np.all(np.isclose(Jopt, Jpi)):
    print(f"The max difference between the two cost-to-go matrices is {np.max(np.abs(Jopt - Jpi))}")
    print(f"The average difference between the two cost-to-go matrices is {np.mean(np.abs(Jopt - Jpi))}")

# Select arbitrary state, and evaluate for the optimal policy
s = 115  # State (8, 28, empty)
print('Cost-to-go for state %s:' % M[0][s])
print('\tTheoretical:', np.round(Jopt[s], 4))
print('\tEmpirical:', np.round(simulate(M, popt, s, 1000), 4))

# Select arbitrary state, and evaluate for the optimal policy
s = 429  # (0, None, loaded)
print('Cost-to-go for state %s:' % M[0][s])
print('\tTheoretical:', np.round(Jopt[s], 4))
print('\tEmpirical:', np.round(simulate(M, popt, s, 1000), 4))

# Select arbitrary state, and evaluate for the optimal policy
s = 239  # State (18, 18, empty)
print('Cost-to-go for state %s:' % M[0][s])
print('\tTheoretical:', np.round(Jopt[s], 4))
print('\tEmpirical:', np.round(simulate(M, popt, s, 1000), 4))
