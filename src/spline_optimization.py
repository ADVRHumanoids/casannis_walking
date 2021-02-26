import casadi as cs
import numpy as np


class spline_optimization_z:

    def __init__(self, N):

        sym_t = cs.SX
        self._N = N

        # waypoints
        x = sym_t.sym('x', self._N)

        # spline coefficients
        a = sym_t.sym('a', self._N)
        b = sym_t.sym('b', self._N)
        c = sym_t.sym('c', self._N)
        d = sym_t.sym('d', self._N)

        # time periods between waypoints
        delta_t = sym_t.sym('delta_t', self._N - 1)

        
        alpha = [2 / x for x in delta_t]
        beta1 = [4 / x for x in delta_t]
        beta2 = [4 / x for x in delta_t[1:]]
        beta = np.add(beta1, beta2)

        # matrices
        #h1 = sym_t.sym('h1', self._N, self._N)
        #h2 = sym_t.sym('h2', self._N, self._N)
        #h3 = sym_t.sym('h3', self._N, self._N)
        #h4 = sym_t.sym('h4', self._N, self._N)


        #h1[0] = 0
        #print(h1)
        al = 5
        # function in casadi
        '''xf = cs.vertcat(
            c + dc * delta_t + 0.5 * ddc * delta_t ** 2 + 1.0 / 6.0 * u * delta_t ** 3,  # position
            dc + ddc * delta_t + 0.5 * u * delta_t ** 2,  # velocity
            ddc + u * delta_t  # acceleration
        )

        # wrap the expression into a function
        self._integrator = cs.Function('integrator', [x, u, delta_t], [xf], ['x0', 'u', 'delta_t'], ['xf'])'''

        # QP high-level
        '''x = SX.sym('x');
        y = SX.sym('y')
        qp = {'x': vertcat(x, y), 'f': x ** 2 + y ** 2, 'g': x + y - 10}
        S = qpsol('S', 'qpoases', qp)
        print(S)'''


if __name__ == "__main__":

    spline_optimization_z(3)