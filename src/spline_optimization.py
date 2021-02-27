import casadi as cs
import numpy as np


class spline_optimization_z:

    def __init__(self, N, timings):

        sym_t = cs.SX
        self._N = N

        # waypoints
        x = sym_t.sym('x', self._N)
        dx = sym_t.sym('dx', self._N)
        ddx = sym_t.sym('ddx', self._N)

        # spline coefficients
        a = sym_t.sym('a', self._N)
        b = sym_t.sym('b', self._N)
        c = sym_t.sym('c', self._N)
        d = sym_t.sym('d', self._N)

        # time periods between waypoints
        delta_t = sym_t.sym('delta_t', self._N - 1)

        # matrices
        h3 = sym_t(self._N, self._N)
        h4 = sym_t(self._N, self._N)
        h5 = sym_t(self._N, self._N)
        h6 = sym_t(self._N, self._N)

        for i in range(self._N-1):
            h3[i, i] = -3 / (delta_t[i]**2)
            h3[i, i+1] = 3 / (delta_t[i] ** 2)

            h4[i, i] = -2 / delta_t[i]
            h4[i, i + 1] = -1 / delta_t[i]

            h5[i, i] = 2 / (delta_t[i] ** 3)
            h5[i, i + 1] = -2 / (delta_t[i] ** 3)

            h6[i, i] = 1 / (delta_t[i] ** 2)
            h6[i, i + 1] = 1 / (delta_t[i] ** 2)

        alpha = []
        beta = []
        gama = []
        eta = []

        for i in range(self._N - 1):
            alpha.append(2 / delta_t[i])
            gama.append(6 / (delta_t[i]**2))

        for i in range(self._N - 2):
            beta.append(4 / delta_t[i] + 4 / delta_t[i + 1])
            eta.append(6 / (delta_t[i + 1] ** 2) - 6 / (delta_t[i] ** 2))

        h1 = sym_t(self._N, self._N)
        h2 = sym_t(self._N, self._N)

        for i in range(1, self._N - 1, 1):
            h1[i, i] = beta[i - 1]
            h1[i, i - 1] = alpha[i - 1]
            h1[i, i + 1] = alpha[i]

            h2[i, i] = eta[i - 1]
            h2[i, i - 1] = -gama[i - 1]
            h2[i, i + 1] = gama[i]

        # constraints
        g = []  # list of constraint expressions
        J = []  # list of cost function expressions

        for k in range(self._N):

            # velocity at waypoints
            vel = cs.mtimes(h1[k, :], dx) - cs.mtimes(h2[k, :], x)

            g.append(vel)

            # acceleration at waypoints
            acc = ddx[k] - 0.5 * (cs.mtimes(h3[k, :], x) - cs.mtimes(h4[k, :], dx))

            g.append(acc)

            # coefficients' linear equations
            eq1 = a[k] - x[k]
            eq2 = cs.mtimes(h1[k, :], b) - cs.mtimes(h2[k, :], x)
            eq3 = c[k] - cs.mtimes(h3[k, :], x) - cs.mtimes(h4[k, :], b)
            eq4 = d[k] - cs.mtimes(h5[k, :], x) - cs.mtimes(h6[k, :], b)

            g.append(eq1)
            g.append(eq2)
            g.append(eq3)
            g.append(eq4)

            # timings as constraints
            if not k == self._N-1:
                spline_time = delta_t[k] - timings[k]

            g.append(spline_time)

            # objective function
            j_k = dx[k]**2

            J.append(j_k)

            # Error g is not dense
            g[0] = 0
            g[3] = 0
            g[-4] = 0
            g[-7] = 0

        # QP solver high-level
        qp = {'x': cs.vertcat(x, dx, ddx), #cs.vertcat(x, dx, ddx, a, b, c, d, delta_t),
              'f': sum(J),
              'g': cs.vertcat(*g)
              }

        self._solver = cs.qpsol('solver', 'qpoases', qp)

        print(h1)
        al = 5

    def solver(self, position):

        Xl = list()  # position lower bounds
        Xu = list()  # position upper bounds
        DXl = list()  # velocity lower bounds
        DXu = list()  # velocity upper bounds
        DDXl = list()  # acceleration lower bounds
        DDXu = list()  # acceleration upper bounds
        Timel = list()  # time lower bounds
        Timeu = list()  # time upper bounds
        gl = list()  # constraint lower bounds
        gu = list()  # constraint upper bounds

        '''Au = []
        Al = []
        Bu = []
        Bl = []
        Cu = []
        Cl = []
        Du = []
        Dl = []'''
        for k in range(self._N):

            # variables
            # position bounds
            if k == 0 or k == self._N - 1:
                x_max = position[k]
                x_min = position[k]

            else:
                x_max = position[k] + 0.1
                x_min = position[k]

            Xu.append(x_max)
            Xl.append(x_min)

            # velocity bounds
            if k == 0 or k == self._N - 1:
                dx_max = 0
                dx_min = 0

            else:
                dx_max = cs.inf
                dx_min = - x_max

            DXu.append(dx_max)
            DXl.append(dx_min)

            # acceleration bounds
            ddx_max = cs.inf
            ddx_min = - x_max

            DDXu.append(ddx_max)
            DDXl.append(ddx_min)

            # a,b,c,d
            '''a_max = cs.inf
            a_min = - a_max

            Au.append(a_max)
            Al.append(a_min)

            b_max = cs.inf
            b_min = - b_max

            Bu.append(b_max)
            Bl.append(b_min)

            c_max = cs.inf
            c_min = - c_max

            Cu.append(c_max)
            Cl.append(c_min)

            d_max = cs.inf
            d_min = - d_max

            Du.append(d_max)
            Dl.append(d_min)'''

            # constraints vel, acc, equations, timings
            gl.append(np.concatenate([np.zeros(8)]))
            gu.append(np.concatenate([np.zeros(8)]))

        # format bounds and params according to solver
        lbv = cs.vertcat(*Xl, *DXl, *DDXl)
        ubv = cs.vertcat(*Xu, *DXu, *DDXu)
        lbg = cs.vertcat(*gl)
        ubg = cs.vertcat(*gu)

        sol = self._solver(lbx=lbv, ubx=ubv, lbg=lbg, ubg=ubg)

        #cs.qpsol('solver', 'qpoases', qp)

        print(sol)

        return sol


if __name__ == "__main__":

    times = [2.0, 0.5, 0.5, 1.0]
    my_object = spline_optimization_z(5, times)
    solution = my_object.solver([0.0, 0.2, 0.22, 0.2, 0.0])