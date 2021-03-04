import casadi as cs
import numpy as np
from matplotlib import pyplot as plt
import time


class Spline_optimization_z:

    def __init__(self, N, delta_t):

        sym_t = cs.SX
        self._N = N

        # position, vel, acc at waypoints
        x = sym_t.sym('x', self._N)
        dx = sym_t.sym('dx', self._N)
        ddx = sym_t.sym('ddx', self._N)

        # position, vel at midpoints
        x_mid = sym_t.sym('x_mid', self._N - 1)
        dx_mid = sym_t.sym('dx_mid', self._N - 1)

        # time intervals for midpoints
        delta_t_midpoint = [0.5 * x for x in delta_t]

        # matrices (CasADi type)
        # hi matrices are derived from the cubic polynomials and their conditions
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

        self._h1 = sym_t.zeros(self._N, self._N)
        self._h2 = sym_t.zeros(self._N, self._N)
        alpha = []
        beta = []
        gama = []
        eta = []

        for i in range(self._N - 1):
            alpha.append(2 / delta_t[i])
            gama.append(6 / (delta_t[i]**2))

        for i in range(self._N - 2):
            beta.append(4 / delta_t[i] + 4 / delta_t[i + 1])
            eta.append(-6 / (delta_t[i + 1] ** 2) + 6 / (delta_t[i] ** 2))

        for i in range(1, self._N - 1):
            self._h1[i, i] = beta[i - 1]
            self._h1[i, i - 1] = alpha[i - 1]
            self._h1[i, i + 1] = alpha[i]

            self._h2[i, i] = eta[i - 1]
            self._h2[i, i - 1] = -gama[i - 1]
            self._h2[i, i + 1] = gama[i]

        # matrices for intermediate points
        # gi matrices are derived from cubic polynomials of midpoints combined with waypoints
        g_terms = {
            'g1': [sym_t.zeros(self._N - 1, self._N - 1) for i in range(3)],
            'g2': [sym_t.zeros(self._N - 1, self._N - 1) for i in range(3)],
            'g3': [sym_t.zeros(self._N - 1, self._N - 1) for i in range(2)],
            'g4': [sym_t.zeros(self._N - 1, self._N - 1) for i in range(3)],
        }

        for i in range(self._N-1):
            g_terms['g1'][0][i, :] = self._h3[i, : -1] * (delta_t_midpoint[i] ** 2)
            g_terms['g1'][1][i, :] = self._h5[i, : -1] * (delta_t_midpoint[i] ** 3)
            g_terms['g1'][2][i, i] = 1

            g_terms['g2'][0][i, i] = delta_t_midpoint[i]
            g_terms['g2'][1][i, :] = self._h4[i, : -1] * (delta_t_midpoint[i] ** 2)
            g_terms['g2'][2][i, :] = self._h6[i, : -1] * (delta_t_midpoint[i] ** 3)

            g_terms['g3'][0][i, :] = self._h3[i, : -1] * 2 * delta_t_midpoint[i]
            g_terms['g3'][1][i, :] = self._h5[i, : -1] * 3 * (delta_t_midpoint[i] ** 2)

            g_terms['g4'][0][i, :] = self._h4[i, : -1] * 2 * delta_t_midpoint[i]
            g_terms['g4'][1][i, :] = self._h6[i, : -1] * 3 * (delta_t_midpoint[i] ** 2)
            g_terms['g4'][2][i, i] = 1

        # all 4 matrices in one list
        self._g = [sum(g_terms[x]) for x in g_terms]

        # constraints - objective function
        g = []  # list of constraint expressions
        J = []  # list of cost function expressions

        # loop over waypoints
        for k in range(self._N):

            # velocity at waypoints
            vel_waypoints = cs.mtimes(self._h1[k, :], dx) - cs.mtimes(self._h2[k, :], x)
            g.append(vel_waypoints)

            # acceleration at waypoints
            accel_waypoints = ddx[k] - 2.0 * (cs.mtimes(self._h3[k, :], x) + cs.mtimes(self._h4[k, :], dx))
            g.append(accel_waypoints)

            # loop over midpoints
            if not k == self._N - 1:

                # position at midpoints
                pos_midpoints = x_mid[k] - cs.mtimes(self._g[0][k, :], x[: -1]) - cs.mtimes(self._g[1][k, :], dx[: -1])
                g.append(pos_midpoints)

                # velocity at midpoints
                vel_midpoints = dx_mid[k] - cs.mtimes(self._g[2][k, :], x[: -1]) - cs.mtimes(self._g[3][k, :], dx[: -1])
                g.append(vel_midpoints)

                # objective function
                j_k = dx[k] ** 2 + dx_mid[k]**2

            else:
                j_k = dx[k] ** 2
            J.append(j_k)

        # QP solver
        self._qp = {'x': cs.vertcat(x, dx, ddx, x_mid, dx_mid),
                    'f': sum(J),
                    'g': cs.vertcat(*g)
                    }

        self._solver = cs.qpsol('solver', 'qpoases', self._qp)

    def solver(self, position, ramps, obst):

        # waypoints
        Xl = []  # position lower bounds
        Xu = []  # position upper bounds
        DXl = []  # velocity lower bounds
        DXu = []  # velocity upper bounds
        DDXl = []  # acceleration lower bounds
        DDXu = []  # acceleration upper bounds

        # midpoints
        X_mid_u = []
        X_mid_l = []
        DX_mid_u = []
        DX_mid_l = []

        gl = []  # constraint lower bounds
        gu = []  # constraint upper bounds

        for k in range(self._N):

            # variable bounds
            # trj phases
            is_ramp = (1 <= k <= ramps)
            is_obstacle = (1 + ramps <= k < 1 + ramps + obst)
            is_obstacle_max = (k == int(1 + ramps + obst/2))
            is_landing = (1 + ramps + obst <= k <= self._N - 1)
            is_start_slow_down = (k == 1 + ramps + obst)

            # position bounds
            # start, end position
            if k == 0 or k == self._N - 1:
                print('initial,final', k)
                x_max = position[k]
                x_min = position[k]

            # ramp
            elif is_ramp:
                print('ramp', k)
                x_max = cs.inf #position[k+1]
                x_min = position[0]

            # main obstacle - clearance
            elif is_obstacle:

                if is_obstacle_max:
                    print('is obstacle max', k)
                    x_max = position[k] + 0.02
                    x_min = position[k]

                else:
                    print('obstacle', k)
                    x_max = position[k]
                    x_min = position[k] - 0.05

            elif is_landing:

                print('landing', k)
                x_max = cs.inf #position[k] + 0.02
                x_min = position[-1] #position[k+1]

            else:
                x_max = cs.inf
                x_min = -cs.inf

            Xu.append(x_max)
            Xl.append(x_min)

            # velocity bounds

            # start, end velocity
            if k == 0 or k == self._N - 1:
                dx_max = 0.0
                dx_min = 0.0

            # obstacle max clearance
            elif is_obstacle_max:
                dx_max = 0.0
                dx_min = 0.0

            # landing
            #elif is_start_slow_down:
                #dx_max = 0.0
                #dx_min = - 0.03

            else:
                dx_max = cs.inf
                dx_min = - cs.inf

            DXu.append(dx_max)
            DXl.append(dx_min)

            # acceleration bounds

            # landing
            if is_ramp:
                ddx_max = cs.inf
                ddx_min = 0.2

            elif is_obstacle:
                ddx_max = - 0.0
                ddx_min = - cs.inf

            elif is_landing:

                if is_start_slow_down:
                    print('start slow down', k)
                    ddx_max = -0.0001
                    ddx_min = -cs.inf

                else:
                    ddx_max = cs.inf
                    ddx_min = 0.0

            else:
                ddx_max = cs.inf
                ddx_min = - cs.inf

            DDXu.append(ddx_max)
            DDXl.append(ddx_min)

            # midpoints - variable constraints
            if not k == self._N - 1:
                x_mid_max = cs.inf
                x_mid_min = - cs.inf

                X_mid_u.append(x_mid_max)
                X_mid_l.append(x_mid_min)

                dx_mid_max = cs.inf
                dx_mid_min = - cs.inf

                DX_mid_u.append(dx_mid_max)
                DX_mid_l.append(dx_mid_min)

                gl.append(np.zeros(2))
                gu.append(np.zeros(2))

            # constraints pos, vel, acc at waypoints
            gl.append(np.zeros(2))
            gu.append(np.zeros(2))

        # format bounds and params according to solver
        lbv = cs.vertcat(*Xl, *DXl, *DDXl, *X_mid_l, *DX_mid_l)
        ubv = cs.vertcat(*Xu, *DXu, *DDXu, *X_mid_u, *DX_mid_u)
        lbg = cs.vertcat(*gl)
        ubg = cs.vertcat(*gu)

        # call solver
        self._sol = self._solver(lbx=lbv, ubx=ubv, lbg=lbg, ubg=ubg)

        return self._sol

    def get_splines(self, optimal_knots, delta_t):

        # numerically evaluate matrices
        h1 = self.evaluate(self._sol['x'], self._h1)
        h2 = self.evaluate(self._sol['x'], self._h2)
        h3 = self.evaluate(self._sol['x'], self._h3)
        h4 = self.evaluate(self._sol['x'], self._h4)
        h5 = self.evaluate(self._sol['x'], self._h5)
        h6 = self.evaluate(self._sol['x'], self._h6)

        # pseudo-inverse
        inv_h1 = np.linalg.pinv(h1)

        a = optimal_knots
        b = np.matmul(inv_h1, np.matmul(h2, optimal_knots))
        c = np.matmul(h3, optimal_knots) + np.matmul(h4, b)
        d = np.matmul(h5, optimal_knots) + np.matmul(h6, b)

        pos_coeffs = []
        pos_polynomials = []
        vel_coeffs = []
        vel_polynomials = []
        acc_coeffs = []
        acc_polynomials = []

        for i in range(self._N - 1):

            pos_coeffs.append([a[i], b[i], c[i], d[i]])
            pos_polynomials.append(np.polynomial.polynomial.Polynomial(pos_coeffs[i]))

            vel_coeffs.append([b[i], 2*c[i], 3*d[i]])
            vel_polynomials.append(np.polynomial.polynomial.Polynomial(vel_coeffs[i]))

            acc_coeffs.append([2*c[i], 6*d[i]])
            acc_polynomials.append(np.polynomial.polynomial.Polynomial(acc_coeffs[i]))

        print("bi coeffs are:", b)
        return {
            'pos': pos_polynomials,
            'vel': vel_polynomials,
            'acc': acc_polynomials
        }

    def evaluate(self, solution, expr):
        """ Evaluate a given expression

        Args:
            solution: given solution
            expr: expression to be evaluated

        Returns:
            Numerical value of the given expression

        """

        # casadi function that symbolically maps the _nlp to the given expression
        expr_fun = cs.Function('expr_fun', [self._qp['x']], [expr], ['v'], ['expr'])

        expr_value = expr_fun(v=solution)['expr'].toarray()

        return expr_value


if __name__ == "__main__":

    initial_pos = 0.0
    target_pos = 0.05
    terrain_conf = 0.05
    swing_time = [0.0, 6.0]
    N = 18  # number of waypoints

    ramp_points = 5
    ramp_step = 0.01

    obstacle_points = 3
    h_obs = 0.2

    slow_down_points = N - ramp_points - obstacle_points

    # times
    ramp_time = np.linspace(swing_time[0], 0.5 * (swing_time[1]-swing_time[0]), ramp_points + obstacle_points + 2).tolist()
    fast_time = np.linspace(swing_time[0], 0.4 * (swing_time[1]-swing_time[0]), ramp_points + obstacle_points + 2).tolist()
    slow_time = np.linspace(0.4 * (swing_time[1]-swing_time[0]), swing_time[1], N - (ramp_points + obstacle_points + 1)).tolist()
    times = fast_time + slow_time[1:]

    #times = np.linspace(swing_time[0], swing_time[1], N)
    N = len(times)
    dt = [times[i + 1] - times[i] for i in range(N - 1)]

    time_midpoints = [(times[i] + 0.5 * dt[i]) for i in range(N-1)]

    # Construct waypoints
    waypoints = []

    # ramp
    waypoints.append(initial_pos)
    for i in range(1, ramp_points+1):
        waypoints.append(initial_pos + i * ramp_step)

    # max height - obstacle
    for i in range(obstacle_points):
        waypoints.append(h_obs)

    # slow down
    slow_down = np.linspace(target_pos + terrain_conf, target_pos, N - len(waypoints)).tolist()

    for i in range(len(slow_down)):
        waypoints.append(slow_down[i])

    start = time.time()

    my_object = Spline_optimization_z(N, dt)

    solution = my_object.solver(waypoints, ramp_points, obstacle_points)

    splines = my_object.get_splines(solution['x'][0:N], dt)

    end = time.time()

    print('Positions:', solution['x'][0:N])
    print('Velocities:', solution['x'][N:2 * N])
    print('Accelerations:', solution['x'][2 * N:3 * N])
    print('Computation time:', 100*(end-start), 'ms')

    # print results
    s = [np.linspace(0, dt[i], 100) for i in range(N - 1)]

    plt.figure()
    plt.plot(times, solution['x'][0:N], "o")
    plt.plot(time_midpoints, solution['x'][3 * N:(4 * N - 1)], ".")
    plt.plot(times, waypoints, "x")
    for i in range(N-1):
        plt.plot([x + times[i] for x in s[i]], splines['pos'][i](s[i]))
    plt.grid()
    plt.legend(['assigned knots', 'assigned midpoints', 'initial knots'])
    plt.xlabel('time [s]')
    plt.ylabel('z position [m]')

    plt.figure()
    plt.plot(times, solution['x'][N:2 * N], "o")
    plt.plot(time_midpoints, solution['x'][(4 * N - 1):(5 * N - 2)], ".")
    for i in range(N - 1):
        plt.plot([x + times[i] for x in s[i]], splines['vel'][i](s[i]))
    plt.legend(['assigned knots', 'assigned midpoints', 'initial knots'])
    plt.grid()
    plt.xlabel('time [s]')
    plt.ylabel('z velocity [m/s]')

    plt.figure()
    plt.plot(times, solution['x'][2 * N:3 * N], "o")
    for i in range(N - 1):
        plt.plot([x + times[i] for x in s[i]], splines['acc'][i](s[i]))
    plt.plot(times, solution['x'][2 * N:3 * N], "o")
    plt.grid()
    plt.xlabel('time [s]')
    plt.ylabel('z acceleration [m/s^2]')

    # x-z plot
    '''x = np.polynomial.polynomial.Polynomial([0, 0.0375, 0.00001, 0.0001])

    plt.figure()
    for i in range(N - 1):
        plt.plot([x(j + times[i]) for j in s[i]], splines['pos'][i](s[i]))'''
    plt.show()