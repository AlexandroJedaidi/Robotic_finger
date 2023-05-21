import numpy as np
from scipy.optimize import fsolve
import time


def ode_lhs(t, y, parameters, system):
    dim_u, dim_v = system.dim_u, system.dim_v

    u = y[0 : dim_u]
    v = y[dim_u : dim_u+dim_v]

    m_alpha = np.array(
        parameters.I_h + system.I_b + u.T.dot(system.K_u1).dot(u) + 2 * system.K_u2.dot(u) + v.T.dot(system.K_v1 - system.K_v2).dot(v)
    ).reshape(1,1)

    mass_matrix_lower_right_block = np.block([
        [
            system.M_u,
            np.zeros((dim_u,dim_v)),
            (-system.C_uv.dot(v)).reshape(dim_u, 1)
        ],
        [
            np.zeros((dim_v,dim_u)),
            system.M_v,
            (system.C_vu.dot(u) + system.C_v).reshape(dim_v, 1)
        ],
        [
            (-v.T.dot(system.C_vu)).reshape(1, dim_u),
            (u.T.dot(system.C_uv) + system.C_v.T).reshape(1, dim_v),
            m_alpha
        ]
    ])

    dim = dim_u + dim_v + 1
    mass_matrix = np.block([
        [np.eye(dim), np.zeros((dim, dim))],
        [np.zeros((dim, dim)), mass_matrix_lower_right_block]
    ]).reshape((y.shape[0], y.shape[0]))

    return mass_matrix


def ode_rhs(t, y, parameters, system):
    dim_u, dim_v = system.dim_u, system.dim_v

    u = y[0: dim_u]
    v = y[dim_u: dim_u + dim_v]
    p = y[dim_u + dim_v + 1: 2 * dim_u + dim_v + 1]
    q = y[2 * dim_u + dim_v + 1: 2 * (dim_u + dim_v) + 1]
    alpha = y[-1]

    k_alpha = 2 * p.T.dot(system.K_u1).dot(u) + 2 * system.K_u2.dot(p) + 2 * q.T.dot(system.K_v1 - system.K_v2).dot(v)

    right_hand_side_lower_block = np.block([
        [
            (2 * alpha * system.C_uv.dot(q) - (system.K_u - alpha**2 * system.K_u1).dot(u) + alpha**2 * system.K_u2).reshape(dim_u, 1)
        ],
        [
            (-2 * alpha * system.C_vu.dot(p) - (system.K_v - alpha**2 * system.K_v1 + alpha**2 * system.K_v2).dot(v)).reshape(dim_v, 1)
        ],
        [
            np.array(parameters.tau(t) - k_alpha * alpha + p.T.dot(system.C_uv).dot(q) + q.T.dot(system.C_vu).dot(p))
        ],
    ])

    dim = dim_u + dim_v + 1
    right_hand_side = np.block([
        [y[dim :].reshape(dim, 1)],
        [right_hand_side_lower_block]
    ]).reshape((y.shape[0],))

    return right_hand_side


def ode_solver(fun_lhs, fun_rhs, t_span, steps, y0, parameters, system):
    dim = y0.shape[0]
    y = np.zeros((steps, dim))
    y[0] = y0
    # TODO: write function instead of lambda with x -> Vec s.t. fun_lhs/rhs -> Mat/Vec for efficiency

    def implicit_midpoint_method(x):
        return (
            fun_lhs(
                t_n + step_size / 2,
                ((y[n] + x) / 2).reshape((dim,)),
                parameters, system
            ).dot(((x - y[n]).reshape((dim,)))) -
            step_size * fun_rhs(
                t_n + step_size / 2,
                ((y[n] + x) / 2).reshape((dim,)),
                parameters, system
            )
        )

    step_size = (t_span[1] - t_span[0]) / steps
    t_n = 0
    start_solve = time.time()
    for n in range(steps-1):
        start = time.time()
        y[n+1] = fsolve(implicit_midpoint_method, y[n])
        end = time.time()
        t_n += step_size
        print(f"Solved PDE for t={t_n:.2f} in {(end - start):.4f}s")
    end_solve = time.time()
    print(f"Solved PDE in {(end_solve - start_solve):.4f}s")

    return y