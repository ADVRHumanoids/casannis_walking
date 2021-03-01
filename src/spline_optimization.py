import casadi as cs
import numpy as np
from matplotlib import pyplot as plt


class spline_optimization_z:

    def __init__(self, N, timings):

        sym_t = cs.SX
        self._N = N

        # position, vel, acc at waypoints
        x = sym_t.sym('x', self._N)
        dx = sym_t.sym('dx', self._N)
        ddx = sym_t.sym('ddx', self._N)

        # time periods between waypoints
        delta_t = timings

        # matrices
        self._h3 = sym_t.zeros(self._N, self._N)
        self._h4 = sym_t.zeros(self._N, self._N)
        self._h5 = sym_t.zeros(self._N, self._N)
        self._h6 = sym_t.zeros(self._N, self._N)

        for i in range(self._N-1):
            self._h3[i, i] = -3 / (delta_t[i]**2)
            self._h3[i, i+1] = 3 / (delta_t[i] ** 2)

            self._h4[i, i] = -2 / delta_t[i]
            self._h4[i, i + 1] = -1 / delta_t[i]

            self._h5[i, i] = 2 / (delta_t[i] ** 3)
            self._h5[i, i + 1] = -2 / (delta_t[i] ** 3)

            self._h6[i, i] = 1 / (delta_t[i] ** 2)
            self._h6[i, i + 1] = 1 / (delta_t[i] ** 2)

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

        self._h1 = sym_t.zeros(self._N, self._N)
        self._h2 = sym_t.zeros(self._N, self._N)

        for i in range(1, self._N - 1, 1):
            self._h1[i, i] = beta[i - 1]
            self._h1[i, i - 1] = alpha[i - 1]
            self._h1[i, i + 1] = alpha[i]

            self._h2[i, i] = eta[i - 1]
            self._h2[i, i - 1] = -gama[i - 1]
            self._h2[i, i + 1] = gama[i]

        # inverse of h1
        inv_h1 = cs.solve(self._h1, sym_t.eye(self._h1.size1()))
        prod_12 = cs.times(inv_h1, self._h2)

        # constraints
        g = []  # list of constraint expressions
        J = []  # list of cost function expressions

        for k in range(self._N):

            # velocity at waypoints
            velocity = cs.mtimes(self._h1[k, :], dx) - cs.mtimes(self._h2[k, :], x)
            #velocity = cs.mtimes(prod_12[k, :], x)
            g.append(velocity)

            # acceleration at waypoints
            acceleration = ddx[k] - 0.5 * (cs.mtimes(self._h3[k, :], x) - cs.mtimes(self._h4[k, :], dx))
            #acceleration = 0.5 * (cs.mtimes(self._h3[k, :], x) - cs.mtimes(self._h4[k, :], dx))

            g.append(acceleration)

            # objective function
            j_k = dx[k]**2+ 1e-10*ddx[k]**2

            J.append(j_k)

        # QP solver high-level
        qp = {'x': cs.vertcat(x, dx, ddx), #cs.vertcat(x, dx, ddx, a, b, c, d, delta_t),
              'f': sum(J),
              'g': cs.vertcat(*g)
              }

        self._solver = cs.qpsol('solver', 'qpoases', qp)

    def solver(self, position):

        Xl = []  # position lower bounds
        Xu = []  # position upper bounds
        DXl = []  # velocity lower bounds
        DXu = []  # velocity upper bounds
        DDXl = []  # acceleration lower bounds
        DDXu = []  # acceleration upper bounds
        gl = []  # constraint lower bounds
        gu = []  # constraint upper bounds

        for k in range(self._N):

            # variables
            # position bounds
            if k == 0 or k == self._N - 1:
                x_max = position[k]
                x_min = position[k]

            else:
                x_max = position[k] + 0.1
                x_min = position[k] + 0.02

            Xu.append(x_max)
            Xl.append(x_min)

            # velocity bounds
            if k == 0 or k == self._N - 1:
                dx_max = 0#cs.inf
                dx_min = 0

            elif k == 2:
                dx_max = 0
                dx_min = 0
            else:
                dx_max = cs.inf
                dx_min = - dx_max

            DXu.append(dx_max)
            DXl.append(dx_min)

            # acceleration bounds
            if k == 2:
                ddx_max = cs.inf
                ddx_min = - cs.inf
            else:
                ddx_max = cs.inf
                ddx_min = - cs.inf

            DDXu.append(ddx_max)
            DDXl.append(ddx_min)

            # constraints vel, acc, equations, timings
            gl.append(np.zeros(2))
            gu.append(np.zeros(2))

        # format bounds and params according to solver
        lbv = cs.vertcat(*Xl, *DXl, *DDXl)
        ubv = cs.vertcat(*Xu, *DXu, *DDXu)
        lbg = cs.vertcat(*gl)
        ubg = cs.vertcat(*gu)

        sol = self._solver(lbx=lbv, ubx=ubv, lbg=lbg, ubg=ubg)

        return sol

    def get_splines(self, waypoints, delta_t):

        # matrices
        h3 = np.zeros((self._N, self._N))
        h4 = np.zeros((self._N, self._N))
        h5 = np.zeros((self._N, self._N))
        h6 = np.zeros((self._N, self._N))

        for i in range(self._N - 1):
            h3[i, i] = -3 / (delta_t[i] ** 2)
            h3[i, i + 1] = 3 / (delta_t[i] ** 2)

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
            gama.append(6 / (delta_t[i] ** 2))

        for i in range(self._N - 2):
            beta.append(4 / delta_t[i] + 4 / delta_t[i + 1])
            eta.append(6 / (delta_t[i + 1] ** 2) - 6 / (delta_t[i] ** 2))

        h1 = np.zeros((self._N, self._N))
        h2 = np.zeros((self._N, self._N))

        for i in range(1, self._N - 1, 1):
            h1[i, i] = beta[i - 1]
            h1[i, i - 1] = alpha[i - 1]
            h1[i, i + 1] = alpha[i]

            h2[i, i] = eta[i - 1]
            h2[i, i - 1] = -gama[i - 1]
            h2[i, i + 1] = gama[i]

        #print(np.linalg.det(h1))
        # pseudo - inverse
        inv_h1 = np.linalg.pinv(h1)

        a = waypoints
        b = np.matmul(inv_h1, np.matmul(h2, waypoints))
        c = np.matmul(h3, waypoints) + np.matmul(h4, b)
        d = np.matmul(h5, waypoints) + np.matmul(h6, b)

        coeffs = []
        polynomials = []

        for i in range(self._N - 1):

            coeffs.append([a[i], b[i], c[i], d[i]])

            polynomials.append(np.polynomial.polynomial.Polynomial(coeffs[i]))

        return polynomials


if __name__ == "__main__":

    N = 5
    times = [0.0, 2.0, 2.5, 3.0, 4.0]
    waypoints = [0.0, 0.2, 0.3, 0.2, 0.0]

    dt = [times[i+1]-times[i] for i in range(N-1)]

    my_object = spline_optimization_z(N, dt)
    solution = my_object.solver(waypoints)

    print('Positions:', solution['x'][0:N])
    print('Velocities:', solution['x'][N:2*N])
    print('Accelerations:', solution['x'][2*N:3 * N])

    splines = my_object.get_splines(solution['x'][0:N], dt)

    # print results
    s = [np.linspace(0, dt[i], 100) for i in range(N - 1)]

    plt.figure()
    plt.plot(times, solution['x'][0:5])
    for i in range(N-1):
        plt.plot([x + times[i] for x in s[i]], splines[i](s[i]))
    plt.grid()
    plt.xlabel('time [s]')
    plt.ylabel('z position [m]')
    plt.show()