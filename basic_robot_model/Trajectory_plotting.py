import numpy as np
import matplotlib.pyplot as plt

def plot_joint_trajectory(q, qd, qdd):
    """
     Function to plot the trajectories
     q - joint position (Dof * m) Every column will represent the position in respective time step.
     qd - joint velocity (Dof * m)
     qdd - joint acceleration (Dof * m)

    """
    m = q.shape[1]
    timesteps = np.linspace(0, 1, num=m)
    n = q.shape[0]
    #
    # # plotting subplots
    # fig, axis = plt.subplots(3)
    # fig.suptitle("Joint trajectory")
    #
    # # Position
    # axis[0].set_title("Position")
    # axis[0].set(xlabel="Time", ylabel=" Position")
    #
    # for i in range(n): # numer of links = 2
    #     axis[0].plot(timesteps, q[i])
    #     # axis[0].legend(f"Joint{i + 1}")
    #
    # # Velocity
    #
    # axis[1].set_title("Velocity")
    # axis[1].set(xlabel="Time", ylabel="Velocity")
    #
    # for i in range(n):
    #     axis[1].plot(timesteps, qd[i])
    #     # axis[1].legend(f"Joint{i + 1}")
    #
    # # Acceleration
    #
    # axis[2].set_title("Acceleration")
    # axis[2].set(xlabel="Time", ylabel="Acceleration")
    #
    # for i in range(n):
    #     axis[2].plot(timesteps, qdd[i])
    #     # axis[2].legend(f"Joint{i + 1}")
    # # legends
    # # legends = [f"Joint_{i+1}" for i in range(n)]
    # # axis[0].legend(legends)
    # # axis[1].legend(legends)
    # # axis[2].legend(legends)
    # # fig.tight_layout()
    # # plt.show()

