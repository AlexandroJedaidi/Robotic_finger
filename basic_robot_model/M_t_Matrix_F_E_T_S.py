import numpy as np
from Calculating_M_matrix import M_matrix
from Trajectory_Testing import q, qd
M_t = np.empty((len(q[0]), 2, 2))
for i in range(len(q[0])):
    th1 = q[0][i]
    th2 = q[1][i]
    th1dot = qd[0][i]
    th2dot = qd[1][i]
    M_t[i] = M_matrix(th1, th2, th1dot, th2dot)
print(M_t)

