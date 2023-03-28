# Activity 1.
import numpy as np


def load_pomdp(file, gamma):
    pomdp = np.load(file)
    X = tuple(pomdp['X'])
    A = tuple(pomdp['A'])
    Z = tuple(pomdp['Z'])
    P = tuple(pomdp['P'])
    O = tuple(pomdp['O'])
    c = pomdp['c']

    return (X, A, Z, P, O, c, gamma)


M = load_pomdp("garbage-big.npz", 0.99)

# Activity 2.

import numpy.random as rand


def gen_trajectory(pomdp, x0, n):
    X = np.array([x0] * (n + 1))
    A = np.array([0] * n)
    O = np.array([0] * n)

    for i in range(n):
        A[i] = rand.randint(len(pomdp[1]))
        probabilities = np.cumsum(pomdp[3][A[i]][X[i]])
        random = rand.rand()
        for j in range(len(probabilities)):
            if random < probabilities[j]:
                X[i + 1] = j
                break
        random = rand.rand()
        ob = np.cumsum(pomdp[4][A[i]][X[i + 1]])
        for j in range(len(ob)):
            if random < ob[j]:
                O[i] = j
                break

    return (X, A, O)


# Activity 3.

def belief_update(belief, action, observation):
    aux = belief.dot(action).dot(observation)
    return aux / np.sum(aux)


def sample_beliefs(pomdp, n):
    tol = 1e-3
    n_states = len(pomdp[0])
    x0 = rand.randint(n_states)
    traj = gen_trajectory(pomdp, x0, n)
    belief = np.array([[1 / n_states] * n_states])
    believes = [belief]

    for i in range(n):
        add = True
        a = traj[1][i]
        observation = traj[2][i]
        p = pomdp[3][a]
        o = np.diag(pomdp[4][a][:, observation])

        belief = belief.dot(p).dot(o) / np.sum(belief.dot(p).dot(o))

        for b in believes:
            if np.linalg.norm(b - belief) < tol:
                add = False
                break

        if add:
            believes.append(belief)

    sol = ()
    for belief in believes:
        sol += (belief,)

    return sol


# Activity 4

def solve_mdp(pomdp):
    mdp = (pomdp[0], pomdp[1], pomdp[3], pomdp[5], pomdp[6])

    J = np.zeros((len(mdp[0]), 1))
    err = 1

    # actions
    QQ = []
    for action in range(len(mdp[1])):
        QQ.append(np.zeros((len(mdp[0]), 1)))

    # error condtion     
    while err > 1e-8:
        for action in range(len(mdp[1])):
            QQ[action] = mdp[3][:, action, None] + mdp[4] * mdp[2][action].dot(J)
        J2 = np.min(QQ, axis=0)
        err = np.linalg.norm(J2 - J)
        J = J2

    # actions
    Q = []
    for action in range(len(mdp[1])):
        Q.append(np.zeros((len(mdp[0]), 1)))

    for action in range(len(mdp[1])):
        Q[action] = mdp[3][:, action, None] + mdp[4] * mdp[2][action].dot(J)

    QQQ = []
    actual = []
    for state in range(len(mdp[0])):
        actual = []
        for action in range(len(mdp[1])):
            actual.append(Q[action][state][0])
        QQQ.append(actual)

    return np.array(QQQ)


# Activity 5


def get_heuristic_action(b, q, h):
    if h == 'mls':
        return np.argmin(q[np.argmax(b)])
    elif h == 'q-mdp':
        return np.argmin(b.dot(q))
    elif h == 'av':
        voting = [0] * len(q[0])
        for i in range(len(b[0])):
            for j in range(len(q[0])):
                voting[j] += b[0][i] * q[i][j]
        return np.argmin(voting)


# Activity 6

import numpy as np


def solve_fib(pomdp):
    X, A, Z, P, O, c, gamma = pomdp
    n_states = len(X)
    n_actions = len(A)
    n_observations = len(Z)

    Q = np.zeros((n_states, n_actions))

    max_error = 1000
    tolerance = 1e-1
    while max_error > tolerance:
        Q_prev = np.copy(Q)
        Q = np.zeros((n_states, n_actions))
        for a in range(n_actions):
            for z in range(n_observations):
                Q[:, a] += np.min((P[a] * O[a][:, z]).dot(Q_prev), axis=1)
        Q = c + gamma * Q
        max_error = np.linalg.norm(Q - Q_prev)

    return Q

# From the results we can conclude that if we are in position (27, empty) all the algorithms give the same action that
# is down. This is logical because the belief that there is trash is in position 27 is 0, hence the only action with
# cost less than 1 is down and that is why the algorithm chooses this action.
