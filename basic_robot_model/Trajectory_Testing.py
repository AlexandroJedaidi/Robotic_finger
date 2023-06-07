import numpy as np

from Trajectory_Formulation import cubic_trajectory_plannign
from Trajectory_plotting import plot_joint_trajectory
q0 = np.array([0, 1.5])
qd0 = np.array([0, 0])
qf = np.array([1, 2])
qdf = np.array([0, 0])
q, qd, qdd = cubic_trajectory_plannign(q0, qf, qd0, qdf, m=1000)
plot_joint_trajectory(q, qd, qdd)


