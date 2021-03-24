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
            self._h3[i, i] = - 3 / (delta_t[i] ** 2)
            self._h3[i, i+1] = - self._h3[i, i]

            self._h4[i, i] = - 2 / delta_t[i]
            self._h4[i, i + 1] = 0.5 * self._h4[i, i]

            self._h5[i, i] = 2 / (delta_t[i] ** 3)
            self._h5[i, i + 1] = - self._h5[i, i]

            self._h6[i, i] = 1 / (delta_t[i] ** 2)
            self._h6[i, i + 1] = self._h6[i, i]

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
            g_terms['g1'][0][i, :] = self._h3[i, :] * (delta_t_midpoint[i] ** 2)
            g_terms['g1'][1][i, :] = self._h5[i, :] * (delta_t_midpoint[i] ** 3)
            g_terms['g1'][2][i, i] = 1

            g_terms['g2'][0][i, i] = delta_t_midpoint[i]
            g_terms['g2'][1][i, :] = self._h4[i, :] * (delta_t_midpoint[i] ** 2)
            g_terms['g2'][2][i, :] = self._h6[i, :] * (delta_t_midpoint[i] ** 3)

            g_terms['g3'][0][i, :] = self._h3[i, :] * 2 * delta_t_midpoint[i]
            g_terms['g3'][1][i, :] = self._h5[i, :] * 3 * (delta_t_midpoint[i] ** 2)

            g_terms['g4'][0][i, :] = self._h4[i, :] * 2 * delta_t_midpoint[i]
            g_terms['g4'][1][i, :] = self._h6[i, :] * 3 * (delta_t_midpoint[i] ** 2)
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
                j_k = dx[k] ** 2 + dx_mid[k] ** 2

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

        # max landing velocity
        vel_max = 0.08

        for k in range(self._N):

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
                    x_max = waypoints_pos[k] + 0.02
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
            if k == 0 or k == self._N - 1:  # start, end velocity
                dx_max = 0.0
                dx_min = 0.0

            elif is_obstacle_max:   # obstacle max clearance
                dx_max = 0.0
                dx_min = 0.0

            elif is_landing:    # landing
                dx_max = cs.inf
                dx_min = - vel_max

            else:
                dx_max = cs.inf
                dx_min = - cs.inf

            DXu.append(dx_max)
            DXl.append(dx_min)

            # acceleration bounds
            if is_ramp:     # ramp
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
                elif is_landing:
                    x_mid_max = cs.inf
                    x_mid_min = waypoints_pos[-1]
                else:
                    x_mid_max = cs.inf
                    x_mid_min = -cs.inf

                X_mid_u.append(x_mid_max)
                X_mid_l.append(x_mid_min)

                # velocity
                if is_obstacle_max:
                    dx_mid_max = -0.0
                    dx_mid_min = - cs.inf

                elif is_landing:
                    if is_start_slow_down:
                        dx_mid_max = 0.0
                        dx_mid_min = - vel_max
                    else:
                        dx_mid_max = 0.0
                        dx_mid_min = - vel_max

                else:
                    dx_mid_max = cs.inf
                    dx_mid_min = - cs.inf

                DX_mid_u.append(dx_mid_max)
                DX_mid_l.append(dx_mid_min)

                gl.append(np.zeros(2))
                gu.append(np.zeros(2))

            # constraints vel, acc at waypoints
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

    def get_splines(self):

        # numerically evaluate matrices
        h5 = self.evaluate(self._sol['x'], self._h5)
        h6 = self.evaluate(self._sol['x'], self._h6)

        # evaluate optimal decision variables
        optim_variables_evaluated = self.evaluate(self._sol['x'], self._sol['x'])
        optim_variables = [x for t in optim_variables_evaluated for x in t]

        a = optim_variables[0:self._N]
        b = optim_variables[self._N:2 * self._N] #np.matmul(inv_h1, np.matmul(h2, a))
        c = [0.5 * x for x in optim_variables[2 * self._N:3 * self._N]] #np.matmul(h3, a) + np.matmul(h4, b)
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
        print("ci coeffs are:", c)
        print("di coeffs are:", d)


        return {
            'pos': pos_polynomials,
            'vel': vel_polynomials,
            'acc': acc_polynomials
        }

    def interpolate_trj(self, cubic_splines, dt, inter_freq=300):

        # spline times
        s = [np.linspace(0, dt[i], round(inter_freq * dt[i]) + 1) for i in range(self._N - 1)]

        # list of lists
        trj_list = [cubic_splines['pos'][i](s[i]).tolist()[: -1] for i in range(self._N - 1)]

        # flat list
        trj_list = [i for x in trj_list for i in x]
        trj_list.append(cubic_splines['pos'][self._N - 2](s[self._N - 2][-1]))  # append last points

        # plot interpolated trj
        plt.figure()
        plt.plot(np.linspace(0.0, sum(dt), len(trj_list)), trj_list)
        plt.show()
        print(trj_list)

        return trj_list

    def evaluate(self, opt_solution, expr):
        """ Evaluate a given expression

        Args:
            opt_solution: given solution
            expr: expression to be evaluated

        Returns:
            Numerical value of the given expression

        """

        # casadi function that symbolically maps the _nlp to the given expression
        expr_fun = cs.Function('expr_fun', [self._qp['x']], [expr], ['v'], ['expr'])

        expr_value = expr_fun(v=opt_solution)['expr'].toarray()

        return expr_value

    def print_results(self, point_plan, optimal_vars, polynoms):

        # print waypoints
        '''plt.figure()
        plt.plot(point_plan['wayp_times'], point_plan['waypoints'], 'o')
        plt.xlabel('time [s]')
        plt.ylabel('z position [m]')'''

        print('Waypoints Positions:', optimal_vars['x'][0:self._N])
        print('Velocities:', optimal_vars['x'][self._N:2 * self._N])
        print('Accelerations:', optimal_vars['x'][2 * self._N:3 * self._N])
        print('------------------------------------------')
        print('Midpoints Positions:', optimal_vars['x'][3 * self._N:4 * self._N - 1])
        print('Midpoints Velocities:', optimal_vars['x'][4 * self._N - 1:5 * self._N - 2])

        # print results
        s = [np.linspace(0, point_plan['wayp_dt'][i], 100) for i in range(self._N - 1)]

        plt.figure()
        plt.plot(point_plan['wayp_times'], optimal_vars['x'][0:self._N], "o")
        plt.plot(point_plan['mid_times'], optimal_vars['x'][3 * self._N:(4 * self._N - 1)], ".")
        plt.plot(point_plan['wayp_times'], point_plan['waypoints'], "x")
        # plt.plot(time_midpoints, midpoints, ".")
        for i in range(self._N - 1):
            plt.plot([x + point_plan['wayp_times'][i] for x in s[i]], polynoms['pos'][i](s[i]))
        plt.grid()
        plt.legend(['assigned knots', 'assigned midpoints', 'initial knots'])
        plt.xlabel('time [s]')
        plt.ylabel('z position [m]')

        plt.figure()
        plt.plot(point_plan['wayp_times'], optimal_vars['x'][self._N:2 * self._N], "o")
        plt.plot(point_plan['mid_times'], optimal_vars['x'][(4 * self._N - 1):(5 * self._N - 2)], ".")
        for i in range(self._N - 1):
            plt.plot([x + point_plan['wayp_times'][i] for x in s[i]], polynoms['vel'][i](s[i]))
        plt.legend(['assigned knots', 'assigned midpoints', 'initial knots'])
        plt.grid()
        plt.xlabel('time [s]')
        plt.ylabel('z velocity [m/s]')

        plt.figure()
        plt.plot(point_plan['wayp_times'], optimal_vars['x'][2 * self._N:3 * self._N], "o")
        for i in range(self._N - 1):
            plt.plot([x + point_plan['wayp_times'][i] for x in s[i]], polynoms['acc'][i](s[i]))
        plt.plot(point_plan['wayp_times'], optimal_vars['x'][2 * self._N:3 * self._N], "o")
        plt.grid()
        plt.xlabel('time [s]')
        plt.ylabel('z acceleration [m/s^2]')
        plt.show()


def original_plan(num, initial, target, height_conf, swing_t, clearance, ramp_num, obstacle_num):

    # parameters
    ramp_step = 0.005
    slow_down_points = num - ramp_num - obstacle_num

    # max height point
    if initial >= target:
        max_height = initial + clearance
    else:
        max_height = target_pos + clearance

    # times
    duration = swing_t[1] - swing_t[0]
    ramp_time = np.linspace(swing_t[0], 0.05 * duration, ramp_num).tolist()
    obst_time = np.linspace(0.05 * duration, 0.3 * duration, obstacle_num + 1).tolist()
    slow_time = np.linspace(0.4 * duration, swing_t[1], slow_down_points).tolist()

    times = ramp_time[: -1] + obst_time[:] + slow_time

    dt = [times[i + 1] - times[i] for i in range(num - 1)]

    time_midpoints = [(times[i] + 0.5 * dt[i]) for i in range(num - 1)]

    # Construct waypoints
    waypoints = []
    waypoints.append(initial)   # ramp
    for i in range(1, ramp_num):
        waypoints.append(initial + i * ramp_step)

    for i in range(obstacle_num):   # max clearance - obstacle
        waypoints.append(max_height)

    slow_down = np.linspace(target + height_conf, target, num - len(waypoints)).tolist()

    for i in range(len(slow_down)): # slow down
        waypoints.append(slow_down[i])

    # midpoints positions
    midpoints = [waypoints[i] + 0.5 * (waypoints[i + 1] - waypoints[i]) for i in range(num - 1)]

    return {
        'wayp_dt': dt,
        'waypoints': waypoints,
        'midpoints': midpoints,
        'wayp_times': times,
        'mid_times': time_midpoints
    }


if __name__ == "__main__":

    # main specs of the trajectory
    initial_pos = 0.0
    target_pos = -0.05
    terrain_conf = 0.04
    swing_time = [0.0, 3.0]
    clear = 0.05
    N = 9   # number of waypoints
    ramp_points = 3  # including initial
    obstacle_points = 3

    plan = original_plan(num=N, initial=initial_pos, target=target_pos, height_conf=terrain_conf, swing_t=swing_time,
                         clearance=clear, ramp_num=ramp_points, obstacle_num=obstacle_points)

    start = time.time()

    my_object = Spline_optimization_z(N, plan['wayp_dt'])
    solution = my_object.solver(plan['waypoints'], plan['midpoints'], ramp_points, obstacle_points)
    splines = my_object.get_splines()
    # trj_points = my_object.interpolate_trj(splines , plan['wayp_dt'])

    end = time.time()
    print('Computation time:', 1e3 * (end - start), 'ms')

    my_object.print_results(plan, solution, splines)