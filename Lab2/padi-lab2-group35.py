
# Activity 1.

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

import numpy.random as rand


# Activity 2.

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


# Activity 3.

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
    Cpi = np.array(Cpi).reshape(-1, 1)
    Jpi = np.matmul(np.linalg.inv(identity - mdp[4] * Ppi), Cpi)
    return Jpi

rand.seed(42)

# Activity 4

# Insert your code here.
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

# Activity 5

# Insert your code here.
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

# Activity 6

NRUNS = 100 # Do not delete this

# Insert your code here.
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
