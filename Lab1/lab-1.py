import numpy as np


def load_chain(trans_matrix):
    trans_matrix = np.load(trans_matrix)
    array_length = trans_matrix.shape[1]
    states = []
    for x in range(array_length):
        states.append(str(x))
    return (states, trans_matrix)


M = load_chain('garbage-big.npy')

def prob_trajectory(markov_model, trajectory):
    trans_matrix = markov_model[1]
    prob = 1
    for i in range(len(trajectory) - 1):
        prob *= trans_matrix[int(trajectory[i])][int(trajectory[i + 1])]
    return prob


def stationary_dist(markov_model):
    trans_matrix = markov_model[1]
    eigvals, eigvecs = np.linalg.eig(trans_matrix.T)
    stationary = eigvecs[:, np.isclose(eigvals, 1)]
    stationary = stationary / np.sum(stationary)
    return np.real(stationary).T


u_star = stationary_dist(M)

# <font color='blue'> It tells us the long-term probabilites of the truck being in each location. If we assume that the chain has reached a steady state then the stationary distribution tells you how much time, as a percentage, the truck is expected to spend in each location. As stated before, we assume that the number of steps is large.</font>
# 
# <font color='blue'>If we look at an example, the states that have more connections, like 19,27 and 35, have higher probabilities as we are expected to be there more time. This is since we can get there from more possible path. On the other hand, states like 0,33 and 35, have low probabilities which is expected as they have only one connection, and the truck is expected to be there only a small part of the time.
# </font>


def compute_dist(markov_model, nd_array, N):
    trans_matrix = markov_model[1]
    matrix = np.linalg.matrix_power(trans_matrix, N)
    return np.dot(nd_array, matrix)


# <font color='blue'>In this task we have shown that for sufficiently large $n$ and for an initial distribution $\mu_0$ it holds that:
# 
# 
# $$
# \lim_{n\to \infty} \mu_0 P^n = \mu,
# $$
# 
# <font color='blue'>
# where $\mu$ is the stationary distribution.
# We have shown for some initial distribution $\mu_0$ that as $n$ increases the distribution tends to a stationary distribution.
# 
# The chain is also irreducible and aperiodic. This implies that the chain is ergodic.


def simulate(markov_model, init_dist, N):
    trans_matrix = markov_model[1]
    trajectory = ()
    current_state = np.random.choice(trans_matrix.shape[1], p=init_dist.ravel())
    for _ in range(0, N):
        trajectory = trajectory + (str(current_state),)
        current_state = np.random.choice(trans_matrix.shape[1], p=trans_matrix[int(current_state)])
    return trajectory


nS = len(M[0])

# Initial, uniform distribution
u = np.ones((1, nS)) / nS

np.random.seed(42)

import matplotlib.pyplot as plt

N = 50000
traj = np.array(simulate(M, u, N))

traj = traj.astype(int)

states, count = np.unique(traj, return_counts=True)
plt.figure(figsize=(10, 5))
plt.bar(range(M[1].shape[1]), count, align='center', label='Simulation')
plt.xticks(range(M[1].shape[1]), states, rotation=90)
stac = []
for i in range(len(u_star)):
    stac.append(u_star[i]*N)
stac = np.array(stac).reshape(states.shape)
plt.plot(states, stac, 'ro', label='Stationary Distribution')
plt.legend(loc='upper left')
plt.xlabel('States')
plt.ylabel('Absolute Frequency')
plt.title("Simulated Versus Stationary Distribution")
plt.show()
