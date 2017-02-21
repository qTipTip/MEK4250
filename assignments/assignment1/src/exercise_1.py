"""
Compute the L2, H1 errors for 1 / h with h = 8, 16, 32, 64 when using first and
second order Lagrangian elements, when k = 1, 10.
"""
from fenics import *

def u_exact(k=1, degree=1, function_space=None):
    """
    Returns the exact expression, interpolated to a given function space if
    specified. If not, just returns the expression.
    """
    u = Expression('cos(k*pi*x[0])*sin(k*pi*x[1])', k=k, degree=degree)
    if function_space:
        return interpolate(u, function_space)
    else:
        return u

def boundary(x, eps=1.0E-14):
    """
    Represents the Dirichlet boundary conditions.
    """
    return x[0] < eps or x[1] > 1 - eps

def compute_norm(f, norm='L2'):
    if norm == 'L2':
        return sqrt(assemble(f**2*dx))
    elif norm == 'H1':
        return sqrt(assemble(f**2*dx + inner(grad(f), grad(f))*dx))
    else:
        raise NotImplementedError('Only supports L2, H1 norm')

def solve_system(N=8, degree=1, k=1, f=None, g=None):
    """
    Given the number of grid elements N, a polynomial degree, a frequency k, a
    source term f and a boundary condition g, both as FeniCS Expressions,
    solves the system a(u, v) = L(v), and returns solution, mesh and function
    space.
    """
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
    
    # defaults tailored to this exercise
    if f is None:
        f = Expression(
            '2*pi*pi*k*k*cos(pi*k*x[0])*sin(pi*k*x[1])', k=k, degree=1)
    if g is None:
        g = Constant(0.0)

    # forms
    a = -inner(grad(u), grad(v)) * dx
    L = f * v * dx + g * v * ds

    # solution
    solve(a == L, uh, bc)

    return uh, V, mesh


if __name__ == "__main__":
    solve_system()
