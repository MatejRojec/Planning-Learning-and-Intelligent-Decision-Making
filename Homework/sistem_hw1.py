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

print(sum(p))

p_squared = np.dot(p, p)[:,1]

print(p)
print(p_squared)

indexRecycling_plant = states.index("Recycling plant")
indexStop_A = states.index("Stop A")
indexStop_B = states.index("Stop B")
indexStop_C = states.index("Stop C")
indexStop_D = states.index("Stop D")
indexStop_D = states.index("Stop E")
indexStop_D = states.index("Stop F")

p1 = p[indexRecycling_plant][indexStop_A] * p[indexStop_A][indexStop_B] 
print("Recycling plant - stop A - stop B", p1)

eigenValues, eigenVectors = np.linalg.eig(p.T)
print(eigenValues)
print(eigenVectors)

p = np.array([ [0, 1, 0, 0, 0, 0, 0] ,
               [1/4, 0, 1/4, 1/4, 1/4, 0, 0]  ,
               [0, 1/2, 0, 0, 0, 0, 1/2] ,
               [0, 1/2, 0, 0, 0, 1/2, 0] ,               
               [0, 1/2, 0, 0, 0, 0, 1/2] ,
               [0, 0, 0, 1/2, 0, 0, 1/2] ,
               [0, 0, 1/3, 0, 1/3, 1/3, 0]
                ])


A = np.array([ [-1, 1/4, 1/4, 1/4, 0, 0] ,
               [1/2, -1, 0, 0, 0, 1/2] ,
               [1/2, 0, -1, 0, 1/2, 0] ,
               [1/2, 0, 0, -1, 0, 1/2] ,
               [0, 0, 1/2, 0, -1, 1/2] ,
               [0, 1/3, 0, 1/3, 1/3, -1] ])

b = np.array([-48.75, -60, -55, -70, -37.5, 85])
x = np.linalg.solve(A, b)
print(x)