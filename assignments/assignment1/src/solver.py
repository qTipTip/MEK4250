from fenics import *
from pandas import DataFrame
import numpy as np

set_log_active(False)

class Right(SubDomain):
    def inside(self, x, on_boundary, eps=1.e-14):
        return x[0] > 1 - eps

class Left(SubDomain):
    def inside(self, x, on_boundary, eps=1.e-14):
        return x[0] < eps

def solve_system_two(N=8, mu=1, degree=1, SUPG=False):
    """
    solves the boundary value problem in exercise 2
    given number of mesh elements, the parameter mu and the
    element degree. Supports SUPG.
    """
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, 'Lagrange', degree)
    u_numerical = Function(V)

    bcs = [
        DirichletBC(V, Constant(0.0), Left()),
        DirichletBC(V, Constant(1.0), Right())
    ]

    u = TrialFunction(V)
    v = TestFunction(V)

    if SUPG:
        beta = 0.5 * mesh.hmin()
        v = v + beta * v.dx(0)

    f = Constant(0.0)
    g = Constant(0.0)

    a = mu * inner(grad(u), grad(v)) * dx + u.dx(0) * v * dx
    L = f * v * dx + g * v * ds

    solve(a == L, u_numerical, bcs)

    return u_numerical, V, mesh


def solve_system_one(N=8, k=1, degree=1):
    """
    solves the boundary value problem in exercise 1
    given number of mesh elements, the frequency k and the
    element degree.
    """

    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, 'Lagrange', degree)
    u_numerical = Function(V)

    bc = DirichletBC(V, Constant(0.0), 'near(x[0], 0) || near(x[0], 1)')

    u = TrialFunction(V)
    v = TestFunction(V)

    f = Expression(
        '2*pi*pi*k*k*sin(pi*k*x[0])*cos(pi*k*x[1])', k=k, degree=degree)
    g = Constant(0.0)

    a = -inner(grad(u), grad(v)) * dx
    L = f * v * dx + g * v * ds

    solve(a == L, u_numerical, bc)

    return u_numerical, V, mesh


def exercise_2_b(degree, SUPG=False):
    mu_values = [1, 0.1, 0.01, 0.002]
    N_values = [8, 16, 32, 64]

    errors_L2 = DataFrame(index=N_values, columns=mu_values)
    errors_H1 = DataFrame(index=N_values, columns=mu_values)

    for mu in mu_values:
        for N in N_values:
            u_numerical, V, omega = solve_system_two(
                N=N, mu=mu, degree=degree, SUPG=SUPG)
            u_exact = Expression(
                '(exp(1 / mu * x[0]) - 1) / (exp(1 / mu) - 1)',
                mu=mu,
                degree=degree)

            L2 = errornorm(u_exact, u_numerical, 'L2', degree_rise=3)
            H1 = errornorm(u_exact, u_numerical, 'H1', degree_rise=3)

            errors_L2.set_value(N, mu, L2)
            errors_H1.set_value(N, mu, H1)

    return errors_L2, errors_H1


def exercise_1_b(degree):
    """
    returns the L2 and H1 errors when using lagrange elements of given degree.
    """
    frequencies = [1, 10]
    N_values = [8, 16, 32, 64]

    errors_L2 = DataFrame(index=N_values, columns=frequencies)
    errors_H1 = DataFrame(index=N_values, columns=frequencies)

    for k in frequencies:
        for N in N_values:
            u_numerical, V, omega = solve_system_one(N=N, k=k, degree=degree)
            u_exact = Expression(
                'sin(k*pi*x[0])*cos(k*pi*x[1])', k=k, degree=degree)

            L2 = errornorm(u_exact, u_numerical, 'l2', degree_rise=3)
            H1 = errornorm(u_exact, u_numerical, 'h1', degree_rise=3)

            errors_L2.set_value(N, k, L2)
            errors_H1.set_value(N, k, H1)

    return errors_L2, errors_H1


def estimate_error(L2, H1):
    """
    estimates the error using the least square method
    for each k.
    """

    N = L2.index.values
    parameters = L2.columns.values

    best_fit = DataFrame(
        index=parameters, columns=['alpha', 'C_alpha', 'beta', 'C_beta'])
    h_log = [np.log(1.0 / n) for n in N]
    L2_log = L2.applymap(np.log)
    H1_log = H1.applymap(np.log)

    for p in parameters:
        L2_fit = np.polyfit(h_log, L2_log[p], deg=1)
        H1_fit = np.polyfit(h_log, H1_log[p], deg=1)

        # exponentiate to regain coefficients
        L2_fit[1] = np.exp(L2_fit[1])
        H1_fit[1] = np.exp(H1_fit[1])
    
        best_fit.loc[p] = list(L2_fit) + list(H1_fit)

    return best_fit


if __name__ == "__main__":
    print("Exercise 1.b - computing norms")
    P1_L2, P1_H1 = exercise_1_b(degree=1)
    P2_L2, P2_H1 = exercise_1_b(degree=2)
    print(P1_L2)
    print(P1_H1)
    print(P2_L2)
    print(P2_H1)

    print("Exercise 1.c - computing error estimate")
    P1_best_fit = estimate_error(P1_L2, P2_H1)
    P2_best_fit = estimate_error(P2_L2, P2_H1)
    print(P1_best_fit)
    print(P2_best_fit)

    print("Exercise 2.b - computing norms - using degree 1 elements")
    P1_L2, P1_H1 = exercise_2_b(degree=1, SUPG=True)
    print(P1_L2)
    print(P1_H1)

    print("Exercise 2.c - computing error estimate")
    P1_best_fit = estimate_error(P1_L2, P1_H1)
    print(P1_best_fit)
