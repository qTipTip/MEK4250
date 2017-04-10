from fenics import *


def solver(N=10, lam=1, mu=1, degree=1):

    omega = UnitSquareMesh(N, N)

    V = VectorFunctionSpace(omega, 'Lagrange', degree)
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Expression(
        [
            "-mu * (- pi * pi * ( pi * x[0] * (x[0] * x[0] + x[1] * x[1])*cos(pi * x[0]*x[1]) + 2 * x[1] * sin(pi * x[0] * x[1])))",
            "-mu * (pi * pi * ( pi * x[1] * (x[0] * x[0] + x[1] * x[1])*cos(pi * x[0]*x[1]) + 2 * x[0] * sin(pi * x[0] * x[1])))"
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

    output = File('data/linear_elasticity_n{n}_m{mu}_l{lam}.pvd'.format(n=N, mu=mu, lam=lam))
    output << u_solution

if __name__ == "__main__":
    
    mu = 1
    for N in [8, 16, 32, 62, 124]:
        for lam in [1, 10, 100, 1000]:
            solver(N = N, lam = lam, mu = mu, degree = 1)
