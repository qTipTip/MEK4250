from fenics import *
from pandas import DataFrame

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
            "pi * pi * sin(pi * x[1]) - 2 * pi * cos(x[0])",
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
            'fluid_{n}_{dv}_{dq}.pvd'.format(n=N, dv=degree_V, dq=degree_Q))
        pressure_file = File('pressure_d{n}_{dv}_{dq}.pvd'.format(
            n=N, dv=degree_V, dq=degree_Q))
        fluid_file << U
        pressure_file << P

    error_u = errornorm(u_analytical, U, 'H1', degree_rise=2)
    error_p = errornorm(p_analytical, P, 'L2', degree_rise=2)

    return error_u, error_p


if __name__ == "__main__":

    element_degrees = [(4, 3), (4, 2), (3, 2), (3, 1)]
    N_values = [8, 16, 32, 64]
    for element in element_degrees:
        error_table = DataFrame(index=N_values, columns=['u', 'p'])
        V_deg, Q_deg = element
        for N in N_values:
            errors = solve_system(N, V_deg, Q_deg, file_dump=True)
            error_table.set_value(N, 'u', errors[0])
            error_table.set_value(N, 'p', errors[1])
        print(error_table.to_latex())
