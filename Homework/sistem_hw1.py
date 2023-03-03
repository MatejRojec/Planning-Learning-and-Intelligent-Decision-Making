import numpy as np

states= ["Recycling plant", 
            "Stop A", 
            "Stop B", 
            "Stop C", 
            "Stop D", 
            "Stop E", 
            "Stop F", ]

p = np.array([ [0, 1, 0, 0, 0, 0, 0] ,
               [1/4, 0, 1/4, 1/4, 1/4, 0, 0]  ,
               [0, 1/2, 0, 0, 0, 0, 1/2] ,
               [0, 1/2, 0, 0, 0, 1/2, 0] ,               
               [0, 1/2, 0, 0, 0, 0, 1/2] ,
               [0, 0, 0, 1/2, 0, 0, 1/2] ,
               [0, 0, 1/3, 0, 1/3, 1/3, 0]
                ])


p_squared = np.dot(p, p)[:,1]

#print(p)

indexRecycling_plant = states.index("Recycling plant")
indexStop_A = states.index("Stop A")
indexStop_B = states.index("Stop B")
indexStop_C = states.index("Stop C")
indexStop_D = states.index("Stop D")
indexStop_D = states.index("Stop E")
indexStop_D = states.index("Stop F")

p1 = p[indexRecycling_plant][indexStop_A] * p[indexStop_A][indexStop_B] 
print("Recycling plant - stop A - stop B", p1)

eigvals, eigvecs = np.linalg.eig(p.T)
print(eigvals)
print(eigvecs)
stationary = eigvecs[:, np.isclose(eigvals, 1)]
stationary = stationary.ravel() / np.sum(stationary)

print("The exact stationary distribution of p is:")
print(stationary)



# system of equsions

A = np.array([ [-1, 1/4, 1/4, 1/4, 0, 0] ,
               [1/2, -1, 0, 0, 0, 1/2] ,
               [1/2, 0, -1, 0, 1/2, 0] ,
               [1/2, 0, 0, -1, 0, 1/2] ,
               [0, 0, 1/2, 0, -1, 1/2] ,
               [0, 1/3, 0, 1/3, 1/3, -1] ])

b = np.array([-30 -1/4 * (40+70+55+30), -60, -55, -70, -37.5, 85])
x = np.linalg.solve(A, b)
print(x)


currentState = "Recycling plant"
frequencies = {}


init_dist = np.ones((1, 7)) / 7
nd_cumsum = np.cumsum(init_dist)


trajectory = []

current_state = np.random.choice(7, p=init_dist.ravel())
currentState = states[current_state]

for i in range(0, 10000):
    trajectory.append(states[current_state])
    current_state = np.random.choice(7 , p = p[current_state])

print("Trajectory length:", len(trajectory))
print("First 10 stops:", trajectory[0:10])



import matplotlib.pyplot as plt

# Simulate a long trajectory
traj = np.array(trajectory)

states, counters = np.unique(traj, return_counts=True)
print(states)
print(counters)

predicted = np.round(stationary * 10000, 0).astype(int)
print(predicted)

plt.bar(range(7), counters, align='center', label='simulation')
plt.xticks(range(7), states, rotation=90)
stac = [[stationary[i] * 10000] for i in range(len(stationary))]
plt.plot(states, stac, 'ro', label='stationary_dist')
plt.legend(loc='upper right')
plt.show()
