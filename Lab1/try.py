import numpy as np

def load_chain(trans_matrix):
    trans_matrix = np.load(trans_matrix)
    array_length = trans_matrix.shape[1]
    states = [str(x) for x in range(array_length)]
    return (states, trans_matrix)

M = load_chain('garbage-big.npy')
print(M)

def prob_trajectory(tuple, trajectory):
    trans_matrix =   tuple[1]
    prob = 1
    for i in range(len(trajectory) - 1):
        prob *= trans_matrix[int(trajectory[i])][int(trajectory[i + 1])]
    return   prob

def stationary_dist(tuple):
    trans_matrix =   tuple[1]
    eigvals, eigvecs = np.linalg.eig(trans_matrix.T)
    stationary = eigvecs[:, np.isclose(eigvals, 1)]
    stationary = stationary / np.sum(stationary)
    return np.real(stationary).T

u_star = stationary_dist(M)

print('Stationary distribution:')
print(np.round(u_star, 2))

u_prime = u_star.dot(M[1])

print('\nIs u* * P = u*?', np.all(np.isclose(u_prime, u_star)))

import random as rnd

def compute_dist(tuple, nd_array, N):
    trans_matrix =   tuple[1]
    matrix = np.linalg.matrix_power(trans_matrix, N)
    return np.dot(nd_array, matrix)

nS = len(M[0])

rnd.seed(42)

# Initial random distribution
u = np.random.random((1, nS))
u = u / np.sum(u)

# Distrbution after 100 steps
v = compute_dist(M, u, 100)
print('\nIs u * P^100 = u*?', np.all(np.isclose(v, u_star)))

# Distrbution after 2000 steps
v = compute_dist(M, u, 2000)
print('\nIs u * P^2000 = u*?', np.all(np.isclose(v, u_star)))

def simulate(tuple , init_dist, N):
    trans_matrix = tuple[1]
    trajectory = ()
    current_state = np.random.choice(trans_matrix.shape[1], p=init_dist.ravel())
    for _ in range(0, N):
        trajectory = trajectory + (str(current_state), )
        current_state = np.random.choice(trans_matrix.shape[1] , p = trans_matrix[int(current_state)])
    return(trajectory)

nS = len(M[0])

# Initial, uniform distribution
u = np.ones((1, nS)) / nS

np.random.seed(42)

# Simulate short trajectory
traj = simulate(M, u, 10)
print('Small trajectory:', traj)

# Simulate a long trajectory
traj = simulate(M, u, 10000)
print('End of large trajectory:', traj[-10:])

import matplotlib.pyplot as plt

traj = np.array(simulate(M, u, 50000))

# Convert to integers
traj = traj.astype(int)

states, count = np.unique(traj, return_counts=True)

predicted = np.round(u_star * 50000, 0).astype(int)

plt.bar(range(M[1].shape[1]), count, align='center', label='simulation')
plt.xticks(range(M[1].shape[1]), states, rotation=90)
stac = [[u_star[i] * 50000] for i in range(len(u_star))]
stac = np.array(stac).reshape(states.shape)
print(f"shape of stac: {stac.shape}")
print(f"shape of states: {states.shape}")
plt.plot(states, stac, 'ro', label='stationary distribution')
plt.legend(loc='upper center')
plt.show()