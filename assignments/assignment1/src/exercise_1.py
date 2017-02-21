"""
Compute the L2, H1 errors for 1 / h with h = 8, 16, 32, 64 when using first and
second order Lagrangian elements, when k = 1, 10.
"""
from fenics import *


def boundary(x, eps=1.0E-14):
    return x[0] < eps or x[1] > 1 - eps


def solve_system(N=8, degree=1, k=1, f=None, g=None):
    # initialize mesh, function space, and dummy function
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, 'Lagrange', degree)
    uh = Function(V)

    # boundary conditions
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, boundary)

    # variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    if f is None:
        f = Expression(
            '2*pi*pi*k*k*cos(pi*k*x[0])*sin(pi*k*x[1])', k=k, degree=1)
    if g is None:
        g = Constant(0.0)

    # forms
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx + g * v * ds

    # solution
    solve(a == L, uh, bc)

    return uh, V, mesh


if __name__ == "__main__":
    solve_system()
