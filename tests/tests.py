import casadi as cs
import numpy as np
from scipy.sparse import diags

def f():
    sym_t = cs.SX
    x = sym_t.sym('x', 4)
    dx = sym_t.sym('dx', 4)
    ddx = sym_t.sym('ddx', 4)
    qp = cs.vertcat(*x, *dx, *ddx)

    g = []

    for i in range(4):
        constraint1 = x[i] - dx[i]
        g.append(constraint1)

        constraint2 = x[i]
        g.append(constraint2)

        constraint3 = dx[i]
        g.append(constraint3)


    print(g)
    print(*g)
    print(cs.vertcat(*g))

    xx = []
    for k in range(4):
        x_k = sym_t.sym('x_' + str(k), 1)
        xx.append(x_k)
    print(x, xx)



f()
