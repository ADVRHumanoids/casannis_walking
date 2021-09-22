import casadi as cs
import numpy as np
import scipy.interpolate as ip
from matplotlib import pyplot as plt
from scipy.stats import norm


def get_skew_symmetric(vector):
    skew_matrix = sym_t.zeros(3, 3)
    skew_matrix[0, 1] = - vector[2]
    skew_matrix[0, 2] = vector[1]

    skew_matrix[1, 0] = vector[2]
    skew_matrix[1, 2] = - vector[0]

    skew_matrix[2, 0] = - vector[1]
    skew_matrix[2, 1] = vector[0]

    print('Skew symmetric matrix: ', skew_matrix)

    return skew_matrix


if __name__ == "__main__":

    sym_t = cs.SX

    c = sym_t.sym('c', 3)
    dc = sym_t.sym('dc', 3)
    ddc = sym_t.sym('ddc', 3)
    theta = cs.vertcat(c, dc, ddc)

    x = c[0]
    y = c[1]
    z = c[2]

    C_matrix = sym_t.zeros(3, 3)

    C_matrix[0, 0] = cs.cos(y) * cs.cos(z)
    C_matrix[0, 1] = - cs.sin(z)
    C_matrix[1, 0] = cs.cos(y) * cs.sin(z)
    C_matrix[1, 1] = cs.cos(z)
    C_matrix[2, 0] = -cs.sin(y)
    C_matrix[2, 2] = 1.0

    # sym_t([[cs.cos(y) * cs.cos(z), - cs.sin(z), 0.0],
    #       [cs.cos(y) * cs.sin(z), cs.cos(z), 0.0],
    #       [-cs.sin(y), 0.0, 1.0]])
    print('C matrix is: ', C_matrix)

    omega = cs.mtimes(C_matrix, dc)
    print('Omega is: ', omega)

    omega_skew = get_skew_symmetric(omega)

    dC_matrix = cs.mtimes(omega_skew, C_matrix)
    print('dC matrix is: ', dC_matrix)

    omega_dot = cs.mtimes(dC_matrix, dc) + cs.mtimes(C_matrix, ddc)
    print('Omega dot is: ', omega_dot)
