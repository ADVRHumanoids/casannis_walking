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
            'g1': [sym_t.zeros(self._N - 1, self._N) for i in range(3)],
            'g2': [sym_t.zeros(self._N - 1, self._N) for i in range(3)],
            'g3': [sym_t.zeros(self._N - 1, self._N) for i in range(2)],
            'g4': [sym_t.zeros(self._N - 1, self._N) for i in range(3)],
        }

        for i in range(self._N-1):
            g_terms['g1'][0][i, :] = self._h3[i, : ] * (delta_t_midpoint[i] ** 2)
            g_terms['g1'][1][i, :] = self._h5[i, : ] * (delta_t_midpoint[i] ** 3)
            g_terms['g1'][2][i, i] = 1

            g_terms['g2'][0][i, i] = delta_t_midpoint[i]
            g_terms['g2'][1][i, :] = self._h4[i, :] * (delta_t_midpoint[i] ** 2)
            g_terms['g2'][2][i, :] = self._h6[i, :] * (delta_t_midpoint[i] ** 3)

            g_terms['g3'][0][i, :] = self._h3[i, :] * 2 * delta_t_midpoint[i]
            g_terms['g3'][1][i, :] = self._h5[i, : ] * 3 * (delta_t_midpoint[i] ** 2)

            g_terms['g4'][0][i, :] = self._h4[i, : ] * 2 * delta_t_midpoint[i]
            g_terms['g4'][1][i, :] = self._h6[i, : ] * 3 * (delta_t_midpoint[i] ** 2)
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
            if k < self._N - 1:

                # position at midpoints
                pos_midpoints = x_mid[k] - cs.mtimes(self._g[0][k, :], x) - cs.mtimes(self._g[1][k, :], dx)
                g.append(pos_midpoints)

                # velocity at midpoints
                vel_midpoints = dx_mid[k] - cs.mtimes(self._g[2][k, :], x) - cs.mtimes(self._g[3][k, :], dx)
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

    def solver(self, waypoints_pos, midpoints_pos, ramps, obst):

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
            is_ramp = (0 < k < ramps)  # excluding initial
            is_obstacle = (ramps <= k < ramps + obst)
            is_obstacle_max = (k == int(ramps + obst / 2))
            is_landing = (ramps + obst <= k < self._N - 1)  # excluding final
            is_start_slow_down = (k == ramps + obst)

            # position bounds

            if k == 0 or k == self._N - 1:  # initial/target position
                x_max = waypoints_pos[k]
                x_min = waypoints_pos[k]

            elif is_ramp:  # ramp part
                print('ramp', k)
                x_max = cs.inf
                x_min = waypoints_pos[0]

            elif is_obstacle:  # main obstacle - clearance part

                if is_obstacle_max:  # maximum clearance
                    print('is obstacle max', k)
                    x_max = waypoints_pos[k]
                    x_min = waypoints_pos[k]

                else:  # lower than maximum clearance
                    print('obstacle', k)
                    x_max = waypoints_pos[k]
                    x_min = -cs.inf

            elif is_landing:  # landing part

                if is_start_slow_down:  # first point
                    x_max = waypoints_pos[k]
                    x_min = waypoints_pos[k]
                else:
                    x_max = cs.inf
                    x_min = waypoints_pos[-1]

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
            elif is_landing:
                dx_max = cs.inf
                dx_min = - 0.03

            else:
                dx_max = cs.inf
                dx_min = - cs.inf

            DXu.append(dx_max)
            DXl.append(dx_min)

            # acceleration bounds

            # ramp
            if is_ramp:
                ddx_max = cs.inf
                ddx_min = 0.0

            elif is_obstacle:
                ddx_max = 0.0
                ddx_min = - cs.inf

            # elif is_landing:

            # if is_start_slow_down:
            # print('start slow down', k)
            # ddx_max = -0.0001
            # ddx_min = -cs.inf

            # else:
            # ddx_max = cs.inf
            # ddx_min = 0.0

            else:
                ddx_max = cs.inf
                ddx_min = - cs.inf

            DDXu.append(ddx_max)
            DDXl.append(ddx_min)

            # midpoints - variable constraints
            if not k == self._N - 1:

                # position
                if is_ramp:
                    x_mid_max = cs.inf
                    x_mid_min = waypoints_pos[k]
                # elif is_obstacle_max:
                # x_mid_max = waypoints_pos[k]
                # x_mid_min = - cs.inf
                # elif is_landing:
                # x_mid_max = cs.inf
                # x_mid_min = -cs.inf
                else:
                    x_mid_max = cs.inf
                    x_mid_min = -cs.inf

                X_mid_u.append(x_mid_max)
                X_mid_l.append(x_mid_min)

                # velocity
                if is_obstacle_max:
                    dx_mid_max = -0.000
                    dx_mid_min = - cs.inf

                elif is_landing:
                    if is_start_slow_down:
                        dx_mid_max = 0.0
                        dx_mid_min = -cs.inf
                    else:
                        dx_mid_max = 0.0
                        dx_mid_min = -cs.inf

                else:
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

    def get_splines(self, optim_variables, delta_t):

        # numerically evaluate matrices
        h3 = self.evaluate(self._sol['x'], self._h3)
        h4 = self.evaluate(self._sol['x'], self._h4)
        h5 = self.evaluate(self._sol['x'], self._h5)
        h6 = self.evaluate(self._sol['x'], self._h6)

        # pseudo-inverse
        #inv_h1 = np.linalg.pinv(h1)

        a = optim_variables['x'][0:N]
        b = optim_variables['x'][N:2 * N] #np.matmul(inv_h1, np.matmul(h2, a))
        c = np.matmul(h3, a) + np.matmul(h4, b)
        d = np.matmul(h5, a) + np.matmul(h6, b)

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

        print("ai coeffs are:", a)
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

    # main specs of the trajectory
    initial_pos = 0.0
    target_pos = 0.05
    terrain_conf = 0.03
    swing_time = [0.0, 6.0]
    clearance = 0.1
    N = 11 # number of waypoints

    ramp_points = 5  # including initial
    ramp_step = 0.005

    obstacle_points = 3

    if initial_pos >= target_pos:
        max_height = initial_pos + clearance
    else:
        max_height = target_pos + clearance

    slow_down_points = N - ramp_points - obstacle_points

    # times
    duration = swing_time[1] - swing_time[0]

    ramp_time = np.linspace(swing_time[0], 0.05 * duration, ramp_points).tolist()
    obst_time = np.linspace(0.05 * duration, 0.3 * duration, obstacle_points + 1).tolist()
    slow_time = np.linspace(0.3 * duration, swing_time[1], N + 1 - (ramp_points + obstacle_points)).tolist()

    times = ramp_time[: -1] + obst_time[: -1] + slow_time
    dt = [times[i + 1] - times[i] for i in range(N - 1)]

    time_midpoints = [(times[i] + 0.5 * dt[i]) for i in range(N - 1)]

    # Construct waypoints
    waypoints = []

    # ramp
    waypoints.append(initial_pos)
    for i in range(1, ramp_points):
        waypoints.append(initial_pos + i * ramp_step)

    # max clearance - obstacle
    for i in range(obstacle_points):
        waypoints.append(max_height)

    # slow down
    slow_down = np.linspace(target_pos + terrain_conf, target_pos, N - len(waypoints)).tolist()

    for i in range(len(slow_down)):
        waypoints.append(slow_down[i])

    # midpoints positions
    midpoints = [waypoints[i] + 0.5 * (waypoints[i + 1] - waypoints[i]) for i in range(N - 1)]

    start = time.time()

    my_object = Spline_optimization_z(N, dt)
    solution = my_object.solver(waypoints, midpoints, ramp_points, obstacle_points)
    splines = my_object.get_splines(solution, dt)

    end = time.time()

    print('Waypoints Positions:', solution['x'][0:N])
    print('Velocities:', solution['x'][N:2 * N])
    print('Accelerations:', solution['x'][2 * N:3 * N])
    print('------------------------------------------')
    print('Midpoints Positions:', solution['x'][3 * N:4 * N - 1])
    print('Midpoints Velocities:', solution['x'][4 * N - 1:5 * N - 2])
    print('Computation time:', 1000 * (end - start), 'ms')

    # print results
    s = [np.linspace(0, dt[i], 100) for i in range(N - 1)]

    plt.figure()
    plt.plot(times, solution['x'][0:N], "o")
    plt.plot(time_midpoints, solution['x'][3 * N:(4 * N - 1)], ".")
    plt.plot(times, waypoints, "x")
    # plt.plot(time_midpoints, midpoints, ".")
    for i in range(N - 1):
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

    plt.show()