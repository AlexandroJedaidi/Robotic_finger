# import ufl
# from mpi4py import MPI
# import numpy as np
# import basix
# import basix.ufl_wrapper
# from basix import ElementFamily, CellType, LagrangeVariant
# from dolfinx.mesh import create_interval
# from dolfinx.fem import assemble, FunctionSpace
# from ufl import (SpatialCoordinate, TestFunction, TrialFunction, nabla_grad, nabla_div, dx, MixedElement)
# import dolfinx
#
# from IPython import embed
# embed()


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
from mpi4py import MPI
import basix
from petsc4py import PETSc
from dolfinx.fem import (
    Function, FunctionSpace, dirichletbc, form, assemble_scalar, set_bc, locate_dofs_topological, Constant, Expression
)
from dolfinx.geometry import BoundingBoxTree, compute_collisions, compute_colliding_cells
from dolfinx.fem.petsc import assemble_matrix, assemble_vector		# https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.petsc.html
from dolfinx.mesh import locate_entities_boundary, create_interval, locate_entities
from ufl import (SpatialCoordinate, inner, nabla_grad, nabla_div, dx, MixedElement, TestFunction, TrialFunction)
from scipy.optimize import fsolve


"""
valid configs: 
T = 5, steps = 150, element = 50, tau = 0.1 * sin(1.5t), alpha_initial = 0.3

"""


NUM_ELEMENTS = 50
NUM_TIME_STEPS = 150
T = 5 # 7.5


def tau__(t):
    amp = 2.
    freq = 5 # 1.
    return amp * np.cos(freq * t)

def tau_(t): return 1.

def tau(t):
    t1 = T / 3.
    t2 = 2 * T / 3.
    if t < t2:
        if t < t1:
            tmp = 1.
        else:
            tmp = -1.
    else:
        tmp = 0.
    return 2 * tmp

L = 1.1               # beam length
rho = 2700          # density
A = 1e-4               # cross-sectional area
E = 71e9              # Young's modules
I = 8.33e-12              # area moment of inertia
R = 0.1               # radius of rigid hub
I_h = 3.84             # mass moment of inertia of hub

# initial rotation position and velocity of hub beam system
theta_initial = 0   # -45 * (np.pi / 180)   # np.pi / 2
alpha_initial = 0   # 0.3


mesh = create_interval(MPI.COMM_WORLD, NUM_ELEMENTS, [0, L])
x = SpatialCoordinate(mesh)

# TODO: does not work with mixed element
# (u, v) = TrialFunctions(function_space)
# (w, z) = TestFunctions(function_space)
# function_space = FunctionSpace(mesh, MixedElement([u_element, v_element]))
u_element = basix.ufl_wrapper.create_element(
    basix.ElementFamily.P,
    basix.CellType.interval,
    1,
)
v_element = basix.ufl_wrapper.create_element(
    basix.ElementFamily.Hermite,
    basix.CellType.interval,
    3
)

u_function_space = FunctionSpace(mesh, u_element)           # u_function_space.dofmap.list
v_function_space = FunctionSpace(mesh, v_element)

u_ = TrialFunction(u_function_space)
v_ = TrialFunction(v_function_space)
w = TestFunction(u_function_space)
z = TestFunction(v_function_space)


# from IPython import embed
# embed()

left_boundary = locate_entities_boundary(
    mesh, mesh.topology.dim-1, lambda s: np.isclose(s[0], 0)
)
# right_boundary = locate_entities_boundary(
#     mesh, mesh.topology.dim-1, lambda s: np.isclose(s[0], L)
# )
u_boundary_dofs = locate_dofs_topological(u_function_space, mesh.topology.dim-1, left_boundary)
v_boundary_dofs = locate_dofs_topological(v_function_space, mesh.topology.dim-1, left_boundary)
# u_boundary_dofs_ = locate_dofs_topological(u_function_space, mesh.topology.dim-1, right_boundary)
# v_boundary_dofs_ = locate_dofs_topological(v_function_space, mesh.topology.dim-1, right_boundary)

boundary_conditions = [
    dirichletbc(PETSc.ScalarType(0), u_boundary_dofs, u_function_space),
    dirichletbc(PETSc.ScalarType(0), v_boundary_dofs, v_function_space),
    # dirichletbc(PETSc.ScalarType(0), u_boundary_dofs_, u_function_space),
    # dirichletbc(PETSc.ScalarType(0), v_boundary_dofs_, v_function_space),
]

def mat_to_np(A):
    dim1, dim2 = A.size
    return np.array([[A.getValue(i,j) for i in range(dim1)] for j in range(dim2)])

def vec_to_np(A):
    dim = A.size
    return np.array([A.getValue(i) for i in range(dim)])

def d_dx(f): return nabla_grad(f)

def d2_dx2(f): return nabla_div(nabla_grad(f))


system_matrices = {
    "M_u": rho * A * u_ * w * dx,
    "M_v": rho * A * v_ * z * dx,
    "C_uv": rho * A * u_ * z * dx,
    "C_vu": rho * A * v_ * w * dx,
    "K_u": E * A * inner(d_dx(u_), d_dx(w)) * dx,
    "K_v": E * I * d2_dx2(v_) * d2_dx2(z) * dx,
    "K_u1": rho * A * u_ * w * dx,
    "K_v1": rho * A * v_ * z * dx,
    "K_v2": rho * A * (R * (L - x[0]) + 0.5 * (L + x[0]) * (L - x[0])) * inner(d_dx(v_), d_dx(z)) * dx,
}
system_vectors = {
    "K_u2": rho * A * (R + x[0]) * w * dx,
    "C_v": rho * A * (R + x[0]) * z * dx,
}
system_scalars = {
    "I_b": rho * A * (R + x[0])**2 * dx,
}

built_matrices = {name: assemble_matrix(form(mat), bcs=boundary_conditions) for name, mat in system_matrices.items()}
for mat in built_matrices.values():
    mat.assemble()
# built_np_matrices = built_matrices
built_np_matrices = {
    name:
        mat_to_np(      # TODO: here instead get csr matrix
            built_matrices[name]
        )
    for name in system_matrices.keys()
}
built_vectors = {name: assemble_vector(form(vec)) for name, vec in system_vectors.items()}
for vec in built_vectors.values():
    vec.assemble()
    set_bc(vec, bcs=boundary_conditions)
built_np_vectors = {
    name:
        vec_to_np(
            built_vectors[name]
        )
    for name in system_vectors.keys()
}
built_scalars = {name: assemble_scalar(form(scal)) for name, scal in system_scalars.items()}


# TODO: use petsc4py.PETSc .Mat and .Vec instead of np arrays
def ode_lhs(t, y, args):
    u_dim, v_dim = args["u_dim"], args["v_dim"]

    u = y[0 : u_dim]
    v = y[u_dim : u_dim+v_dim]
    # theta = y[u_dim+v_dim]
    # p = y[u_dim+v_dim+1 : 2*u_dim+v_dim+1]
    # q = y[2*u_dim+v_dim+1 : 2*(u_dim+v_dim)+1]
    # alpha = y[-1]

    m_alpha = np.array(
        I_h + args["I_b"] + u.T.dot(args["K_u1"]).dot(u) + 2 * args["K_u2"].dot(u) + v.T.dot(args["K_v1"] - args["K_v2"]).dot(v)
    ).reshape(1,1)

    mass_matrix_lower_right_block = np.block([
        [
            args["M_u"],
            np.zeros((u_dim,v_dim)),
            (-args["C_uv"].dot(v)).reshape(u_dim, 1)
        ],
        [
            np.zeros((v_dim,u_dim)),
            args["M_v"],
            (args["C_vu"].dot(u) + args["C_v"]).reshape(v_dim, 1)
        ],
        [
            (-v.T.dot(args["C_vu"])).reshape(1, u_dim),
            (u.T.dot(args["C_uv"]) + args["C_v"].T).reshape(1, v_dim),
            m_alpha
        ]
    ])

    dim = u_dim + v_dim + 1
    mass_matrix = np.block([
        [np.eye(dim), np.zeros((dim, dim))],
        [np.zeros((dim, dim)), mass_matrix_lower_right_block]
    ]).reshape((y.shape[0], y.shape[0]))

    return mass_matrix


def ode_rhs(t, y, args):
    u_dim, v_dim = args["u_dim"], args["v_dim"]

    u = y[0: u_dim]
    v = y[u_dim: u_dim + v_dim]
    # theta = y[u_dim + v_dim]
    p = y[u_dim + v_dim + 1: 2 * u_dim + v_dim + 1]
    q = y[2 * u_dim + v_dim + 1: 2 * (u_dim + v_dim) + 1]
    alpha = y[-1]

    k_alpha = 2 * p.T.dot(args["K_u1"]).dot(u) + 2 * args["K_u2"].dot(p) + 2 * q.T.dot(args["K_v1"] - args["K_v2"]).dot(v)

    right_hand_side_lower_block = np.block([
        [
            (2 * alpha * args["C_uv"].dot(q) - (args["K_u"] - alpha**2 * args["K_u1"]).dot(u) + alpha**2 * args["K_u2"]).reshape(u_dim, 1)
        ],
        [
            (-2 * alpha * args["C_vu"].dot(p) - (args["K_v"] - alpha**2 * args["K_v1"] + alpha**2 * args["K_v2"]).dot(v)).reshape(v_dim, 1)
        ],
        [
            np.array(args["tau"](t) - k_alpha * alpha + p.T.dot(args["C_uv"]).dot(q) + q.T.dot(args["C_vu"]).dot(p))
        ],
    ])

    dim = u_dim + v_dim + 1
    right_hand_side = np.block([
        [y[dim :].reshape(dim, 1)],
        [right_hand_side_lower_block]
    ]).reshape((y.shape[0],))

    return right_hand_side


def implicit_midpoint_method(fun_lhs, fun_rhs, t_span, steps, y0, args):
    dim = y0.shape[0]
    y = np.zeros((steps, dim))
    y[0] = y_initial
    # TODO: write function instead of lambda with x -> Vec s.t. fun_lhs/rhs -> Mat/Vec for efficiency

    step_size = (t_span[1] - t_span[0]) / steps
    t_n = 0
    for n in range(steps-1):
        y[n+1] = fsolve(
            lambda x:
            fun_lhs(
                t_n + step_size / 2,
                ((y[n] + x) / 2).reshape((dim,)),
                args
            ).dot(((x - y[n]).reshape((dim,)))) -
            step_size * fun_rhs(
                t_n + step_size / 2,
                ((y[n] + x) / 2).reshape((dim,)),
                args
            ),
            y[n]
        )
        t_n += step_size

    return y


dimu = built_np_matrices["M_u"].shape[0]     # for petsc use size instead of shape
dimv = built_np_matrices["M_v"].shape[0]
args_mat = built_np_matrices | built_np_vectors | built_scalars | {"u_dim": dimu, "v_dim": dimv, "tau": tau}

y_initial = np.block([
    [np.zeros((dimu + dimv, 1))],
    [np.array(theta_initial).reshape(1, 1)],
    [np.zeros((dimu + dimv, 1))],
    [np.array(alpha_initial).reshape(1, 1)],
]).reshape(-1,)

y_solution = implicit_midpoint_method(
    ode_lhs,
    ode_rhs,
    (0, T),
    NUM_TIME_STEPS,
    y_initial,
    args=args_mat
)

u_sol = y_solution[:, : dimu]
v_sol = y_solution[:, dimu : dimu+dimv]
theta_sol = y_solution[:, dimu+dimv]
# u_sol = y_solution[:, dimu+dimv+1 : 2*dimu+dimv+1]
# v_sol = y_solution[:, 2*dimu+dimv+1 : 2*(dimu+dimv)+1]
# theta_sol = y_solution[:, -1]


x_coordinates = mesh.geometry.x

bb_tree = BoundingBoxTree(mesh, mesh.topology.dim)
cell_candidates = compute_collisions(bb_tree, x_coordinates)
colliding_cells = compute_colliding_cells(mesh, cell_candidates, x_coordinates)
cells = []
points_on_proc = []
for i, point in enumerate(x_coordinates):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])
points_on_proc = np.array(points_on_proc, dtype=np.float64)


u_eval = np.zeros((y_solution.shape[0], points_on_proc.shape[0]))
v_eval = np.zeros((y_solution.shape[0], points_on_proc.shape[0]))
for i in range(y_solution.shape[0]):
    u_loc = Function(u_function_space)
    u_loc.x.array[:] = u_sol[i]
    u_eval[i] = u_loc.eval(points_on_proc, cells).reshape(-1,)
    v_loc = Function(v_function_space)
    v_loc.x.array[:] = v_sol[i]
    v_eval[i] = v_loc.eval(points_on_proc, cells).reshape(-1,)


x_displacement, y_displacement = np.zeros_like(u_eval), np.zeros_like(u_eval)
x_no_displacement, y_no_displacement = np.zeros_like(u_eval), np.zeros_like(u_eval)
for i in range(x_displacement.shape[0]):
    x_displacement[i] = (R + points_on_proc[:, 0] + u_eval[i]) * np.cos(theta_sol[i]) \
                        - v_eval[i] * np.sin(theta_sol[i])
    y_displacement[i] = (R + points_on_proc[:, 0] + u_eval[i]) * np.sin(theta_sol[i]) \
                        + v_eval[i] * np.cos(theta_sol[i])
    x_no_displacement[i] = (R + points_on_proc[:, 0]) * np.cos(theta_sol[i])
    y_no_displacement[i] = (R + points_on_proc[:, 0]) * np.sin(theta_sol[i])


# plt.clf()
# for time_step in range(x_displacement.shape[0]):
#     if time_step % 10 == 0:
#         # plt.clf()
#         plt.plot(x_displacement[time_step], y_displacement[time_step])
#         plt.savefig(f"displacement_{time_step}.png")
# x_circle = np.linspace(0, R, num=10)
# plt.plot(x_circle, np.sqrt(R**2 - x_circle ** 2))
# plt.xlim((min_displacement, max_displacement))
# plt.ylim((min_displacement, max_displacement))
# plt.savefig(f"displacement_all.png")

min_displacement = min(x_displacement.min(), y_displacement.min())
max_displacement = max(x_displacement.max(), y_displacement.max())



u_tip_displacement = np.zeros(x_displacement.shape[0])
v_tip_displacement = np.zeros(x_displacement.shape[0])

# mid = int(x_displacement.shape[1] / 2)
for i in range(x_displacement.shape[0]):
    u_tip_displacement[i] = u_eval[i, -1]
    v_tip_displacement[i] = v_eval[i, -1]
plt.plot(np.linspace(0, T, num=x_displacement.shape[0]), u_tip_displacement)
plt.savefig(f"out/displacement_tip_u.png")
plt.clf()
plt.plot(np.linspace(0, T, num=x_displacement.shape[0]), v_tip_displacement)
plt.savefig(f"out/displacement_tip_v.png")
plt.clf()



# START ANIMATION
fig, ax = plt.subplots()
ax.set_xlim(min_displacement, max_displacement)
ax.set_ylim(min_displacement, max_displacement)

line1, = ax.plot(x_displacement[0], y_displacement[0])
line2, = ax.plot(x_no_displacement[0], y_no_displacement[0], "--")
ax.add_patch(plt.Circle((0, 0), R))


def animate(frame, x, y, x_, y_, lines):
    lines[0].set_data(x[frame], y[frame])
    lines[1].set_data(x_[frame], y_[frame])
    return lines


ani = FuncAnimation(
    fig, animate, x_displacement.shape[0],
    fargs=[x_displacement, y_displacement, x_no_displacement, y_no_displacement, [line1, line2]],
    interval=25, blit=True
)
ani.save("out/displacement.gif")



from IPython import embed
embed()




# links regarding csr matrix
# https://stackoverflow.com/questions/36969886/using-a-sparse-matrix-versus-numpy-array
# https://fenicsproject.discourse.group/t/converting-to-scipy-sparse-matrix-without-eigen-backend/847
# https://stackoverflow.com/questions/36782588/dot-product-sparse-matrices



# print(original, displacement, original - displacement)


# https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.html
# https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.cpp.common.html



# u_index_x = u_function_space.tabulate_dof_coordinates()[:, 0]
# v_index_x = v_function_space.tabulate_dof_coordinates()[:, 0]


# x_coordinates = mesh.geometry.x
# u_values = [
#     u_1.eval(x_coordinates[j+1], j)
#     for j in range(10)
# ]
#
# v_values = [
#     v_1.eval(x_coordinates[j+1], j)
#     for j in range(10)
# ]



# num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
#
# v_dolfmap_list = [v_function_space.dofmap.cell_dofs(i) for i in range(num_cells)]



#
# x_indices = mesh.entities_to_geometry(
#     mesh, mesh.topology.dim, np.arange(num_cells, dtype=np.int32), False)
# points = mesh.geometry.x
# for cell in range(num_cells):
#     vertex_coords = points[x_indices[cell]]




# sol = ode(y_initial, dimu, dimv, args, tau_test)



# for scal in built_scalars.values():
#     scal.assemble()
# built_np_scalars = {
#     name:
#         scal_to_np(
#             built_scalars[name]
#         )
#     for name in system_scalars.keys()
# }

# M_u = assemble_matrix(
#     form(rho * A * u * w * dx),
#     bcs=boundary_conditions
# )
# M_v = assemble_matrix(
#     form(rho * A * v * z * dx),
#     bcs=boundary_conditions
# )
# C_uv = assemble_matrix(
#     form(rho * A * u * z * dx)
# )
# C_uv_ = assemble_matrix(
#     form(rho * A * v * w * dx)
# )	# TOOD: seems to be C_vu, so the transposed variant # C_vu = C_uv.T
# K_u = assemble_matrix(
#     form(E * A * inner(d_dx(u), d_dx(w)) * dx)
# )
# K_v = assemble_matrix(
#     form(E * I * d2_dx2(v) * d2_dx2(z) * dx)
# )
# K_u1 = M_u
# K_v1 = M_v
#
# K_u2 = assemble_vector(
#     form(rho * A * (R + x[0]) * w * dx)		# to get numpy use .array (only for vector)
# )
# K_v2 = assemble_matrix(				# to get numpy diags use .getValuesCSR() -> Compressed Row Storage
#     form(rho * A * (R * (L - x[0]) + 0.5 * (L + x[0]) * (L - x[0])) * inner(d_dx(v), d_dx(z)) * dx)
# )
# C_v = assemble_vector(
#     form(rho * A * (R + x[0]) * z * dx)
# )
# I_b = assemble_scalar(
#     form(rho * A * (R + x[0])**2 * dx)
# )




# https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.Mat-class.html
# use .assemble() and then with .getValues(i,j) iter over object to get matrix

# M_u.assemble()
# u_shape = M_u.getSize()[0]
# M_u_np = np.array([M_u.getValues(i,j) for i in range(u_shape)] for j in range(u_shape))


# https://fenicsproject.org/olddocs/dolfinx/dev/python/_autogenerated/dolfinx.fem.assemble.html
#I_h missing, should be in m_alpha


# https://jsdokken.com/dolfinx-tutorial/chapter2/linearelasticity_code.html#stress-computation





