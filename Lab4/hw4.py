import numpy as np

X = ("(E, 1, 0, 1)", "(F, 1, 0, 1)")
S4bR = 0; 
S5bR = 1
UP = 0; 
DOWN = 1; 
LEFT = 2; 
RIGHT = 5
GAMMA = 0.9
STEP_SIZE = 0.1
Q_t = np.array([[2.8, 2.8, 2.8, 2.8, 2.54, 2.0], [2.8, 2.8, 2.95, 2.0, 3.14, 2.8]])

# Slide 15 of lec21.pdf
def Q_learning_update(x_t, a_t, c_t, x_t1):
    result = np.copy(Q_t)
    result[x_t, a_t] = Q_t[x_t, a_t] + STEP_SIZE * (c_t + GAMMA * np.min(Q_t[x_t1]) - Q_t[x_t, a_t])
    return result

print("Q-values after a Q-learning update resulting from the transition at time step t:")
Q_t1_Q_learning = Q_learning_update(0, RIGHT, 0.2, 1)
for state, Q in zip(X, Q_t1_Q_learning):
    print(state + ":", Q)


def SARSA_update(x_t, a_t, c_t, x_t1, a_t1):
    result = np.copy(Q_t)
    result[x_t, a_t] = Q_t[x_t, a_t] + STEP_SIZE * (c_t + GAMMA * Q_t[x_t1, a_t1] - Q_t[x_t, a_t])
    return result

print("Q-values after a SARSA update resulting from the transition at time step t:")
Q_t1_SARSA = SARSA_update(0, RIGHT, 0.2, 1, RIGHT)
for state, Q in zip(X, Q_t1_SARSA):
    print(state + ":", Q)