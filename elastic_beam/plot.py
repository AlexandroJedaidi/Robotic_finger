import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def plot_displacements(u_solution, v_solution, theta_solution, parameters, name):

    fig, ax = plt.subplots(4,1, figsize=(10, 15), constrained_layout=True)
    mid_point_index = int(u_solution.shape[1] / 2)
    t_space = np.linspace(0, parameters.T, num=parameters.num_time_steps)

    ax[0].plot(t_space, u_solution[:,-1], label="tip")
    ax[0].plot(t_space, u_solution[:,mid_point_index], label="mid")
    ax[0].legend()
    ax[0].set_title("Axial displacements of the beam")
    ax[0].set_xlabel(r"time $t$")
    ax[0].set_ylabel(r"distance $u$")

    ax[1].plot(t_space, v_solution[:,-1], label="tip")
    ax[1].plot(t_space, v_solution[:,mid_point_index], label="mid")
    ax[1].legend()
    ax[1].set_title("Transverse displacements of the beam")
    ax[1].set_xlabel(r"time $t$")
    ax[1].set_ylabel(r"distance $v$")

    ax[2].plot(t_space, theta_solution)
    ax[2].set_title("Rotation of the beam")
    ax[2].set_xlabel(r"time $t$")
    ax[2].set_ylabel(r"angle $\theta$")

    ax[3].plot(t_space, [parameters.tau(t_i) for t_i in t_space])
    ax[3].set_title("Supplied torque from actuator")
    ax[3].set_xlabel(r"time $t$")
    ax[3].set_ylabel(r"torque $\tau$")

    # fig0.suptitle("")
    fig.savefig(f"elastic_beam/out/{name}.png")
    # fig.savefig("out/displacement")


def animate_displacements_frame(frame, x_displacement, y_displacement, x_no_displacement, y_no_displacement, lines):
    lines[0].set_data(x_displacement[frame], y_displacement[frame])
    lines[1].set_data(x_no_displacement[frame], y_no_displacement[frame])
    # ax.set_title(f"Dynamics of rotating Euler-bernoulli beam at t={frame * T / NUM_TIME_STEPS}s")
    return lines


def animate_displacements(x_displacement, y_displacement, x_no_displacement, y_no_displacement, parameters, name):

    # START ANIMATION
    fig, ax = plt.subplots()
    min_displacement = min(x_displacement.min(), y_displacement.min())
    max_displacement = max(x_displacement.max(), y_displacement.max())

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xlim(min_displacement, max_displacement)
    ax.set_ylim(min_displacement, max_displacement)
    ax.set_title("Dynamics of rotating Euler-bernoulli beam")

    line1, = ax.plot(x_displacement[0], y_displacement[0], label="elastic")
    line2, = ax.plot(x_no_displacement[0], y_no_displacement[0], "--", label="rigid")
    ax.legend()
    ax.add_patch(plt.Circle((0, 0), parameters.R))

    ani = FuncAnimation(
        fig, animate_displacements_frame, x_displacement.shape[0],
        fargs=[x_displacement, y_displacement, x_no_displacement, y_no_displacement, [line1, line2]],
        interval=25, blit=True
    )
    ani.save(f"elastic_beam/out/{name}.gif")
    # ani.save("out/displacement.gif")
