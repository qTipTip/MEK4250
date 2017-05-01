from fenics import *
from pandas import DataFrame
import numpy as np

set_log_active(False)


def solve_system(N, degree_V, degree_Q, file_dump=False):

    mesh = UnitSquareMesh(N, N)

    # Create mixed element space
    V = VectorElement("Lagrange", mesh.ufl_cell(), degree_V)
    Q = FiniteElement("Lagrange", mesh.ufl_cell(), degree_Q)
    W = FunctionSpace(mesh, V * Q)

    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Specific problem
    f = Expression(
        [
            "pi * pi * sin(pi * x[1]) - 2 * pi * cos(2*pi*x[0])",
            "pi * pi * cos(pi * x[0])"
        ],
        degree=degree_V)
    p_analytical = Expression("sin(2 * pi * x[0])", degree=degree_Q)
    u_analytical = Expression(
        ["sin(pi * x[1])", "cos(pi * x[0])"], degree=degree_V)

    # boundary conditions
    def u_boundary(x):
        return x[0] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS or x[
            1] < DOLFIN_EPS

    def p_boundary(x):
        return x[0] > 1.0 - DOLFIN_EPS

    bc_fluid = DirichletBC(W.sub(0), u_analytical, u_boundary)
    bc_press = DirichletBC(W.sub(1), p_analytical, p_boundary)

    # weak formulation
    a = inner(grad(u), grad(v)) * dx + div(u) * q * dx + div(v) * p * dx
    L = inner(f, v) * dx

    UP = Function(W)
    A, b = assemble_system(a, L, [bc_fluid, bc_press])
    solve(A, UP.vector(), b, 'lu')

    U, P = UP.split()

    if file_dump:
        fluid_file = File(
            'data/fluid_{n}_{dv}_{dq}.pvd'.format(n=N, dv=degree_V, dq=degree_Q))
        pressure_file = File('data/pressure_d{n}_{dv}_{dq}.pvd'.format(
            n=N, dv=degree_V, dq=degree_Q))
        fluid_file << U
        pressure_file << P

    error_u = errornorm(u_analytical, U, 'H1', degree_rise=2)
    error_p = errornorm(p_analytical, P, 'L2', degree_rise=2)
    error_sum = error_u + error_p
    return error_u, error_p, error_sum


def estimate_error(error):
    """
    Finds the line best suiting the data and returns intersection and slope.
    
    :param error: pandas DataFrame
    """
    
    N = error.index.values
    parameters = error.columns.values
    best_fit = DataFrame(
        index=parameters, columns=['convergence rate', 'coefficient'])
    h_log = [np.log(1.0 / n) for n in N]
    error_log = error.applymap(np.log)
    error_fit = np.polyfit(h_log, error_log['u + p'], deg=1)
    # exponentiate to regain coefficients
    error_fit[1] = np.exp(error_fit[1])
    return list(error_fit)

if __name__ == "__main__":

    element_degrees = [(4, 3), (4, 2), (3, 2), (3, 1)]
    N_values = [8, 16, 32, 64]
    error_table = DataFrame(index=N_values, columns=['u + p'])
    approximation_table = DataFrame(index=element_degrees, columns=['convergence rate', 'coefficient'])
    for element in element_degrees:
        V_deg, Q_deg = element
        for N in N_values:
            errors = solve_system(N, V_deg, Q_deg, file_dump=True)
            error_table.set_value(N, 'u + p', errors[2])
        best_fit_table = estimate_error(error_table)
        approximation_table.set_value(element, 'convergence rate', best_fit_table[0])
        approximation_table.set_value(element, 'coefficient', best_fit_table[1])
    print(approximation_table.to_latex())
