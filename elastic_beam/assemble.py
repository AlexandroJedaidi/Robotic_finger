import numpy as np
from dolfinx.geometry import BoundingBoxTree, compute_collisions, compute_colliding_cells
from dolfinx.fem import Function, FunctionSpace, locate_dofs_topological, dirichletbc
from dolfinx.mesh import locate_entities_boundary, create_interval
from petsc4py import PETSc
import basix
from mpi4py import MPI


def create_mesh_and_function_spaces(parameters):
    mesh = create_interval(MPI.COMM_WORLD, parameters.num_space_elements, [0, parameters.L])
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
    return mesh, [FunctionSpace(mesh, u_element), FunctionSpace(mesh, v_element)]


def define_boundary_conditions(mesh, function_spaces):
    left_boundary = locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda y: np.isclose(y[0], 0)
    )
    boundary_dofs = [
        locate_dofs_topological(function_space, mesh.topology.dim - 1, left_boundary)
        for function_space in function_spaces
    ]
    return [
        dirichletbc(PETSc.ScalarType(0), boundary_dof, function_space)
        for boundary_dof, function_space in zip(boundary_dofs, function_spaces)
    ]


def get_nodes_for_function_eval(u_solution, v_solution, mesh, function_spaces, parameters):
    mesh_coordinates = mesh.geometry.x

    bb_tree = BoundingBoxTree(mesh, mesh.topology.dim)
    cell_candidates = compute_collisions(bb_tree, mesh_coordinates)
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, mesh_coordinates)
    cells = []
    x_coordinates = []
    for i, point in enumerate(mesh_coordinates):
        if len(colliding_cells.links(i)) > 0:
            x_coordinates.append(point)
            cells.append(colliding_cells.links(i)[0])
    x_coordinates = np.array(x_coordinates, dtype=np.float64)

    u_eval = np.zeros((parameters.num_time_steps, x_coordinates.shape[0]))
    v_eval = np.zeros((parameters.num_time_steps, x_coordinates.shape[0]))
    for i in range(parameters.num_time_steps):
        u_loc = Function(function_spaces[0])
        u_loc.x.array[:] = u_solution[i]
        u_eval[i] = u_loc.eval(x_coordinates, cells).reshape(-1, )
        v_loc = Function(function_spaces[1])
        v_loc.x.array[:] = v_solution[i]
        v_eval[i] = v_loc.eval(x_coordinates, cells).reshape(-1, )

    return u_eval, v_eval, x_coordinates



def calculate_displacements_in_origin_frame(u_solution, v_solution, theta_solution, x_coordinates, parameters):
    x_displacement, y_displacement = np.zeros_like(u_solution), np.zeros_like(u_solution)
    x_no_displacement, y_no_displacement = np.zeros_like(u_solution), np.zeros_like(u_solution)
    for i in range(x_displacement.shape[0]):
        x_displacement[i] = (parameters.R + x_coordinates + u_solution[i]) * np.cos(theta_solution[i]) \
                            - v_solution[i] * np.sin(theta_solution[i])
        y_displacement[i] = (parameters.R + x_coordinates + u_solution[i]) * np.sin(theta_solution[i]) \
                            + v_solution[i] * np.cos(theta_solution[i])
        x_no_displacement[i] = (parameters.R + x_coordinates) * np.cos(theta_solution[i])
        y_no_displacement[i] = (parameters.R + x_coordinates) * np.sin(theta_solution[i])

    return x_displacement, y_displacement, x_no_displacement, y_no_displacement
