from fenics import *
from pandas import DataFrame
import numpy as np
set_log_active(False)

def table_formatter(x):
    return "{:.6f}".format(x)

def solver(N=10, lam=1, mu=1, degree=1, file_dump=False):

    omega = UnitSquareMesh(N, N)

    V = VectorFunctionSpace(omega, 'Lagrange', degree)
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Expression(
        [
            "mu * (pi * pi * ( pi * x[0] * (x[0] * x[0] + x[1] * x[1])*cos(pi * x[0]*x[1]) + 2 * x[1] * sin(pi * x[0] * x[1])))",
            "mu * (- pi * pi * ( pi * x[1] * (x[0] * x[0] + x[1] * x[1])*cos(pi * x[0]*x[1]) + 2 * x[0] * sin(pi * x[0] * x[1])))"
        ],
        degree=degree,
        mu=mu)

    u_exact = Expression(
        ["pi * x[0] * cos(pi * x[0] * x[1])", "-pi*x[1]*cos(pi*x[0] * x[1])"],
        degree=degree)

    # boundary conditions
    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_exact, boundary)
    # weak formulation
    a = mu * inner(grad(u), grad(v)) * dx + lam * inner(div(u), div(v)) * dx
    L = inner(f, v) * dx

    u_solution = Function(V)
    solve(a == L, u_solution, bc)
    
    if file_dump:
        output = File('data/linear_elasticity_n{n}_m{mu}_l{lam}.pvd'.format(
            n=N, mu=mu, lam=lam))
        output << u_solution
    
    error = errornorm(u_exact, u_solution, 'L2', degree_rise=2)

    return error

def compute_convergence(error_table, N_vals, l_vals):
    convergence_table = DataFrame(index=l_vals, columns=['alpha'])
    for column, lam in zip(error_table.transpose().get_values(), l_vals):
        h_log = [np.log(1.0 / n) for n in N_vals]
        error_log = error_table.applymap(np.log)
        convergence_rate = np.polyfit(h_log, error_log[lam], deg=1)[0]
        convergence_table.set_value(lam, 'alpha', convergence_rate)
    return convergence_table

def linear_elasticity(polynomial_order=1):
    
    l_values = [1, 10, 100, 1000]
    N_values = [8, 16, 32, 64]
    mu = 1
    error_table = DataFrame(index=N_values, columns=l_values)
    for lam in l_values:
        for N in N_values:
            error_table.set_value(N, lam, solver(N=N, lam=lam, mu=mu, degree=polynomial_order))
    convergence = compute_convergence(error_table, N_values, l_values)

    return error_table, convergence

if __name__ == "__main__":
    for p in [1, 2]:
        error, convergence = linear_elasticity(polynomial_order = p)
        error = error.applymap(table_formatter)
        convergence = convergence.applymap(table_formatter)
        
        print("Polynomial order:", p)
        print("Error:")
        print(error.to_latex())
        print("Convergence rates:")
        print(convergence.to_latex())
