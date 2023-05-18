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
import numpy as np
from mpi4py import MPI
import basix
from petsc4py import PETSc
from dolfinx.fem import (Function, FunctionSpace, dirichletbc, form,
                         locate_dofs_topological, Constant, Expression)
# from dolfinx.io import VTKFile
# from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector		# https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.petsc.html
from dolfinx.fem import assemble_scalar, locate_dofs_topological, dirichletbc
from dolfinx.mesh import locate_entities_boundary, create_interval
from ufl import (SpatialCoordinate, inner, TestFunctions, TrialFunctions, nabla_grad, nabla_div, dx, MixedElement, TestFunction, TrialFunction)
from scipy.optimize import fsolve

from scipy.integrate import solve_ivp


NUM_ELEMENTS = 10
NUM_TIME_STEPS = 100
T = 1.0


def tau(t): return 0.1 * np.sin(t)


L = 1.1               # beam length
rho = 2700          # density
A = 1e-4               # cross-sectional area
E = 71e9              # Young's modules
I = 8.33e-12              # area moment of inertia
R = 0.1               # radius of rigid hub
I_h = 3.84             # mass moment of inertia of hub

# initial rotation position and velocity of hub beam system
theta_initial = np.pi / 2
alpha_initial = 0


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
u_boundary_dofs = locate_dofs_topological(u_function_space, mesh.topology.dim-1, left_boundary)
v_boundary_dofs = locate_dofs_topological(v_function_space, mesh.topology.dim-1, left_boundary)

boundary_conditions = [
    dirichletbc(PETSc.ScalarType(0), u_boundary_dofs, u_function_space),
    dirichletbc(PETSc.ScalarType(0), v_boundary_dofs, v_function_space),
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
built_np_matrices = {
    name:
        mat_to_np(
            built_matrices[name]
        )
    for name in system_matrices.keys()
}
built_vectors = {name: assemble_vector(form(vec)) for name, vec in system_vectors.items()}
for vec in built_vectors.values():
    vec.assemble()
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
        args["I_b"] + u.T.dot(args["K_u1"]).dot(u) + 2 * args["K_u2"].dot(u) + v.T.dot(args["K_v1"] - args["K_v2"]).dot(v)
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
            y[0]
        )
        t_n += step_size

    return y


dimu = built_np_matrices["M_u"].shape[0]
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

u_sol = y_solution[:, dimu+dimv+1 : 2*dimu+dimv+1]
v_sol = y_solution[:, 2*dimu+dimv+1 : 2*(dimu+dimv)+1]
theta_sol = y_solution[:, -1]




# https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.fem.html
# https://docs.fenicsproject.org/dolfinx/main/python/generated/dolfinx.cpp.common.html

u_1 = Function(u_function_space)
u_1.x.array[:] = u_sol[1]

v_1 = Function(v_function_space)
v_1.x.array[:] = v_sol[1]

from IPython import embed
embed()

v_index_x = v_function_space.tabulate_dof_coordinates() #[:,0]
v_1.eval(v_index_x, v_1)

# v_function_space.dofmap.list




# TODO: check how to interpolate array over whole function_space to get Function() object
## or get all spatial coordinates
u_ = Function(u_function_space)
u_.vector.set_local(u_sol.reshape(-1,))
u_.vector.apply("insert")





from IPython import embed
embed()




















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
# I_b = assemble_scalar(		# TODO: assemble_scalar does not work ImportError
#     form(rho * A * (R + x[0])**2 * dx)
# )




# https://www.mcs.anl.gov/petsc/petsc4py-current/docs/apiref/petsc4py.PETSc.Mat-class.html
# use .assemble() and then with .getValues(i,j) iter over object to get matrix

# M_u.assemble()
# u_shape = M_u.getSize()[0]
# M_u_np = np.array([M_u.getValues(i,j) for i in range(u_shape)] for j in range(u_shape))


# https://fenicsproject.org/olddocs/dolfinx/dev/python/_autogenerated/dolfinx.fem.assemble.html
# TODO: I_h missing, should be in m_alpha


# https://jsdokken.com/dolfinx-tutorial/chapter2/linearelasticity_code.html#stress-computation




# TODO: after solving ODE and getting an matrix with each col being the sol for given timestep 
## split after that in (u, v, theta) vector and calculate 
## Point_new = rotation_matrix_from_theta * [R + Point_old_x + u, v]




