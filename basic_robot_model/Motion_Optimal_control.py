"""
Instead of doing reverse kinematics we can go to the robot motion control
Formulating the optimal control problem.
state variables x = [th1,th2,th1dot,th2dot]
control variables = [ta_1, ta_2]
system dynamics xd[i] =
[qd[0][i], qd[1][i], M_in[i] @ (np.array([[ta_1[i]],[ta_2[i]])- np.array([[R_t[0][i],[R_t[1][i]])
State variables
x[i] = np.array([[q[0][i]],[q[1][i]],[qd[0][i]],[qd[1][i]])
Objective function
sum(i = 1, len(q[0])), np.array([[ta_1[i]],[ta_2[i]])
Constraints
ta_1min[i] <= ta_1[i] <= ta_1max[i]
ta_2min[i] <= ta_2[i] < = ta_2max[i]
Initial condition

"""
# import AeroSandbox as asb
# import AeroSandbox.numpy as np


from numpy import linalg
from Calculating_M_matrix import M_matrix
from Calculating_rho_matrix import rho_matrix
from Trajectory_Testing import q, qd
import casadi as cs
import matplotlib.pyplot as plt
from M_t_Matrix_F_E_T_S import M_t
from R_t_matrix_F_E_T_S import R_t

N = 1000                 # number of control intervals
opti = .Opti()
X = opti.variable(4, N+1)
# X = np.array(h)
th_1 = X[0, :]
th_2 = X[1, :]
th_1d = X[2, :]
th_2d = X[3, :]
U = opti.variable(2, N)  # control variables
tau_1 = U[0, :]
tau_2 = U[1, :]
T = opti.variable()      # Final time minimization this is our cost function
opti.minimize(T)         # our objective function
# u = np.zeros((2, 1))
# initial condition
# th_1[0] = np.pi/3
# th_2[0] = np.pi/6
# th_1d[0] = 0
# th_2d[0] = 1

# f = lambda x, u,  m_inv, rho : [x[2, :], x[3, :], [m_inv @ (u - rho)]]
def gradient_f(th1dot, th2dot, u, M_inv, rho):
    grad = np.zeros((4, 1))
    grad[0] = th1dot
    grad[1] = th2dot
    grad[2:4, :] = M_inv @ (u - rho)
    return grad

dt = T/N
for k in range(N):
    th1 = th_1[k]
    th2 = th_2[k]
    th1dot = th_1d[k]
    th2dot = th_2d[k]
    # mat = M_matrix(th1, th2, th1dot, th2dot)
    M_inv = np.linalg.inv(M_t[k])
    rho = R_t[k]
    u = U[:, k]
    # k1 = f(X[:, k], U[:, k], M_inv, rho)
    # k2 = f((X[:, k] + dt / 2 * k1), U[:, k], M_inv, rho)
    # k3 = f(X[:, k] + (dt / 2 * k2), U[:, k], M_inv, rho)
    # k4 = f(X[:, k] + (dt * k3), U[:, k], M_inv, rho)
    x_next = X[:, k] + dt * (gradient_f(th1dot, th2dot, u, M_inv, rho))
    # x_next = X[:, k] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(X[:, k + 1] == x_next)

# setting the boundary conditions for control variables

opti.subject_to([U[0] > 0, U[1] > 0,U[0] <= 1000, U[1] <= 1000, T >= 0])

# opti.subject_to(U[1] > 0)
# opti.subject_to(U[0] <= 1000)
# opti.subject_to(U[1] <= 1000)
# opti.subject_to(T >= 0)

# providing initial guess

opti.set_initial(T, 1)
opti.set_initial(th_1, 1)
opti.set_initial(th_2, 1)
opti.set_initial(th_1d, 1)
opti.set_initial(th_2d, 1)

# solving the NLP using IPOPT

# opti.solver('ipopt')
sol = opti.solve()

pos_1 = sol.value(th_1)   # position of first link
pos_2 = sol.value(th_2)   # position of second link
vel_1 = sol.value(th_1d)  # velocity of first link
vel_2 = sol.value(th_2d)  # velocity of second link
pos = np.array([pos_1, pos_2])
Vel = np.array([vel_1, vel_2])


# plotting subplots
fig, axis = plt.subplots(2)
fig.suptitle("Solution of optimal control problem")

# Position
axis[0].set_title("Position")
axis[0].set(xlabel="Time", ylabel=" Position")

timesteps = np.linspace(0, 1, N)

for i in range(2): # numer of links = 2
    axis[0].plot(timesteps, pos[i])
    axis[0].legend(f"Joint{i + 1}")

# Velocity

axis[1].set_title("Velocity")
axis[1].set(xlabel="Time", ylabel="Velocity")

for i in range(2):
    axis[1].plot(timesteps, Vel[i])
    axis[1].legend(f"Joint{i + 1}")





























# xd(t) = [[[thd(t)],[np.inverse(th(t))(u(t) - rho(th,thd))]] , x =[[th],[thd]]]


# declaring the decision variables ( state and control )




