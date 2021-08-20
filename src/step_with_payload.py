import casadi as cs
import numpy as np
from matplotlib import pyplot as plt

import costs
import trj_interpolation as interpol
import time

import cubic_hermite_polynomial as cubic_spline
import constraints

gravity = np.array([0.0, 0.0, -9.81])
# gravity = np.array([-1.703, 0.0, -9.661])   # 10 deg pitch
# gravity = np.array([-3.3552, 0.0, -9.218])   # 20 deg pitch
# gravity = np.array([-2.539, -0.826, -9.44])   # 15 deg pitch, 5 deg roll

class Walking:
    """
    Trajectory Optimization for a single step with payloads on the robot arms, modeled as virtual moving contacts
    """

    def __init__(self, mass, N, dt, payload_mass):
        """Walking class constructor

        Args:
            mass (float): robot mass
            N (int): horizon length
            dt (float): discretization step
        """

        self._Nseg = N
        self._dt = dt  # dt used for optimization knots
        self._problem_duration = N * dt
        self._mass = mass

        self._knot_number = knot_number = N + 1  # number of knots is one more than segments number
        self._tjunctions = [(i * dt) for i in range(knot_number)]  # time junctions from first to last

        gloabal_gravity = np.array([0, 0, -9.81])

        # define dimensions
        sym_t = cs.SX
        self._dimc = dimc = 3
        self._dimx = dimx = 3 * dimc
        self._dimu = dimu = dimc
        self._dimf = dimf = dimc
        self._dimp_mov = dimp_mov = 3
        self._ncontacts = ncontacts = 4
        self._dimf_tot = dimf_tot = ncontacts * dimf

        # define cs variables
        delta_t = sym_t.sym('delta_t', 1)  # symbolic in order to pass values for optimization/interpolation
        c = sym_t.sym('c', dimc)
        dc = sym_t.sym('dc', dimc)
        ddc = sym_t.sym('ddc', dimc)
        x = cs.vertcat(c, dc, ddc)
        u = sym_t.sym('u', dimc)

        # expression for the integrated state
        xf = cs.vertcat(
            c + dc * delta_t + 0.5 * ddc * delta_t ** 2 + 1.0 / 6.0 * u * delta_t ** 3,  # position
            dc + ddc * delta_t + 0.5 * u * delta_t ** 2,  # velocity
            ddc + u * delta_t  # acceleration
        )

        # wrap the state expression into a function
        self._integrator = cs.Function('integrator', [x, u, delta_t], [xf], ['x0', 'u', 'delta_t'], ['xf'])

        # construct the optimization problem (variables, cost, constraints, bounds)
        X = sym_t.sym('X', knot_number * dimx)  # state is an SX for all knots
        U = sym_t.sym('U', knot_number * dimu)  # for all knots
        F = sym_t.sym('F', knot_number * (ncontacts * dimf))  # for all knots

        # moving contact
        P_mov_l = sym_t.sym('P_mov_l', knot_number * dimp_mov)   # position knots for the virtual contact
        P_mov_r = sym_t.sym('P_mov_r', knot_number * dimp_mov)  # position knots for the virtual contact
        DP_mov_l = sym_t.sym('DP_mov_l', knot_number * dimp_mov)   # velocity knots for the virtual contact
        DP_mov_r = sym_t.sym('DP_mov_r', knot_number * dimp_mov)  # velocity knots for the virtual contact

        # virtual force
        f_pay = np.array([0, 0, gloabal_gravity[2] * payload_mass])   # virtual force

        P = list()  # parameters
        g = list()  # list of constraint expressions
        J = list()  # list of cost function expressions

        self._trj = {
            'x': X,
            'u': U,
            'F': F,
            'P_mov_l': P_mov_l,
            'P_mov_r': P_mov_r,
            'DP_mov_l': DP_mov_l,
            'DP_mov_r': DP_mov_r
        }

        # iterate over knots starting from k = 0
        for k in range(self._knot_number):

            # slice indices for variables at knot k
            x_slice1 = k * dimx
            x_slice2 = (k + 1) * dimx
            u_slice0 = (k - 1) * dimu   # referring to previous knot
            u_slice1 = k * dimu
            u_slice2 = (k + 1) * dimu
            f_slice1 = k * dimf_tot
            f_slice2 = (k + 1) * dimf_tot

            # contact points
            p_k = sym_t.sym('p_' + str(k), ncontacts * dimc)
            P.append(p_k)

            # cost  function
            cost_function = 0.0
            cost_function += costs.penalize_horizontal_CoM_position(1e3, X[x_slice1:x_slice1 + 3], p_k)     # penalize CoM position
            cost_function += costs.penalize_vertical_CoM_position(1e3, X[x_slice1:x_slice1 + 3], p_k)
            cost_function += costs.penalize_xy_forces(1e-3, F[f_slice1:f_slice2])   # penalize xy forces
            cost_function += costs.penalize_quantity(1e-0, U[u_slice1:u_slice2], k, knot_number)    # penalize CoM control
            if k > 0:
                cost_function += costs.penalize_quantity(1e2, (DP_mov_l[u_slice1:u_slice2-1] - DP_mov_l[u_slice0:u_slice1-1]),
                                                         k, knot_number)    # penalize CoM control
                cost_function += costs.penalize_quantity(1e2, (DP_mov_r[u_slice1:u_slice2-1] - DP_mov_r[u_slice0:u_slice1-1]),
                                                         k, knot_number)    # penalize CoM control
            if k == self._knot_number - 1:
                default_lmov_contact = P_mov_l[u_slice1:u_slice2] - X[x_slice1:x_slice1 + 3] - [0.43, 0.179, 0.3]
                default_rmov_contact = P_mov_r[u_slice1:u_slice2] - X[x_slice1:x_slice1 + 3] - [0.43, -0.179, 0.3]
                cost_function += costs.penalize_quantity(1e3, default_lmov_contact, k, knot_number)
                cost_function += costs.penalize_quantity(1e3, default_rmov_contact, k, knot_number)
            J.append(cost_function)

            # newton - euler dynamic constraints
            newton_euler_constraint = constraints.newton_euler_constraint(
                X[x_slice1:x_slice2], mass, gravity, ncontacts, F[f_slice1:f_slice2],
                p_k, P_mov_l[u_slice1:u_slice2], P_mov_r[u_slice1:u_slice2], f_pay
            )
            g.append(newton_euler_constraint['newton'])
            g.append(newton_euler_constraint['euler'])

            # state constraint (triple integrator)
            if k > 0:
                x_old = X[(k - 1) * dimx: x_slice1]  # save previous state
                u_old = U[(k - 1) * dimu: u_slice1]  # prev control
                x_curr = X[x_slice1:x_slice2]
                state_constraint = constraints.state_constraint(
                    self._integrator(x0=x_old, u=u_old, delta_t=dt)['xf'], x_curr)
                g.append(state_constraint)

            # moving contact spline acceleration continuity
            if 0 < k < (self._knot_number - 1):
                left_mov_contact_spline_acc_constraint = constraints.spline_acc_constraint_3D(
                    P_mov_l[u_slice0:u_slice2 + 3], DP_mov_l[u_slice0:u_slice2 + 3], dt, k
                )

                right_mov_contact_spline_acc_constraint = constraints.spline_acc_constraint_3D(
                    P_mov_r[u_slice0:u_slice2 + 3], DP_mov_r[u_slice0:u_slice2 + 3], dt, k
                )

                g.append(left_mov_contact_spline_acc_constraint['x'])
                g.append(left_mov_contact_spline_acc_constraint['y'])
                g.append(left_mov_contact_spline_acc_constraint['z'])

                g.append(right_mov_contact_spline_acc_constraint['x'])
                g.append(right_mov_contact_spline_acc_constraint['y'])
                g.append(right_mov_contact_spline_acc_constraint['z'])

            # box constraint - moving contact
            left_mov_contact_box_constraint = constraints.moving_contact_box_constraint(
                P_mov_l[u_slice1:u_slice2], X[x_slice1:x_slice1 + 3])

            right_mov_contact_box_constraint = constraints.moving_contact_box_constraint(
                P_mov_r[u_slice1:u_slice2], X[x_slice1:x_slice1 + 3])

            g.append(left_mov_contact_box_constraint['x'])
            g.append(left_mov_contact_box_constraint['y'])
            g.append(left_mov_contact_box_constraint['z'])

            g.append(right_mov_contact_box_constraint['x'])
            g.append(right_mov_contact_box_constraint['y'])
            g.append(right_mov_contact_box_constraint['z'])

        # construct the solver
        self._nlp = {
            'x': cs.vertcat(X, U, F, P_mov_l, P_mov_r, DP_mov_l, DP_mov_r),
            'f': sum(J),
            'g': cs.vertcat(*g),
            'p': cs.vertcat(*P)
        }

        # save dimensions
        self._nvars = self._nlp['x'].size1()
        self._nconstr = self._nlp['g'].size1()
        self._nparams = self._nlp['p'].size1()

        solver_options = {
            'ipopt.linear_solver': 'ma57'
        }

        self._solver = cs.nlpsol('solver', 'ipopt', self._nlp, solver_options)

    def solve(self, x0, contacts, mov_contact_initial, swing_id, swing_tgt, swing_clearance, swing_t, min_f=50):
        """Solve the stepping problem

        Args:
            x0 ([type]): initial state (com position, velocity, acceleration)
            contacts ([type]): list of contact point positions
            mov_contact_initial: initial position and velocity of the moving contact
            swing_id ([type]): the index of the swing leg from 0 to 3
            swing_tgt ([type]): the target foothold for the swing leg
            swing_clearance: clearance achieved from the highest point between initial and target position
            swing_t ([type]): pair (t_lift, t_touch) in secs
            min_f: minimum threshold for forces in z direction
        """
        # grab moving contacts from the list
        lmov_contact_initial = mov_contact_initial[0]
        rmov_contact_initial = mov_contact_initial[1]

        # lists for assigning bounds

        # variables
        Xl = [0] * self._dimx * self._knot_number  # state lower bounds (for all knots)
        Xu = [0] * self._dimx * self._knot_number  # state upper bounds
        Ul = [0] * self._dimu * self._knot_number  # control lower bounds
        Uu = [0] * self._dimu * self._knot_number  # control upper bounds
        Fl = [0] * self._dimf_tot * self._knot_number # force lower bounds
        Fu = [0] * self._dimf_tot * self._knot_number  # force upper bounds
        Pl_movl = [0] * self._dimu * self._knot_number  # position of moving contact lower bounds
        Pl_movu = [0] * self._dimu * self._knot_number  # position of moving contact upper bounds
        DPl_movl = [0] * self._dimu * self._knot_number  # velocity of moving contact lower bounds
        DPl_movu = [0] * self._dimu * self._knot_number  # velocity of moving contact upper bounds
        # right
        Pr_movl = [0] * self._dimu * self._knot_number  # position of moving contact lower bounds
        Pr_movu = [0] * self._dimu * self._knot_number  # position of moving contact upper bounds
        DPr_movl = [0] * self._dimu * self._knot_number  # velocity of moving contact lower bounds
        DPr_movu = [0] * self._dimu * self._knot_number  # velocity of moving contact upper bounds

        # constraints
        gl = list()  # constraint lower bounds
        gu = list()  # constraint upper bounds

        # parameters
        P = list()  # parameter values

        # time that maximum clearance occurs
        clearance_time = 0.5 * (swing_t[0] + swing_t[1])  # not accurate

        # swing foot position at maximum clearance
        if contacts[swing_id][2] >= swing_tgt[2]:
            clearance_swing_position = contacts[swing_id][0:2].tolist() + [contacts[swing_id][2] + swing_clearance]
        else:
            clearance_swing_position = swing_tgt[0:2].tolist() + [swing_tgt[2] + swing_clearance]

        # iterate over knots starting from k = 0
        for k in range(self._knot_number):

            # slice indices for bounds at knot k
            x_slice1 = k * self._dimx
            x_slice2 = (k + 1) * self._dimx
            u_slice1 = k * self._dimu
            u_slice2 = (k + 1) * self._dimu
            f_slice1 = k * self._dimf_tot
            f_slice2 = (k + 1) * self._dimf_tot

            # state bounds
            state_bounds = constraints.bound_state_variables(x0, [np.full(9, -cs.inf), np.full(9, cs.inf)], k, self._knot_number)
            Xu[x_slice1:x_slice2] = state_bounds['max']
            Xl[x_slice1:x_slice2] = state_bounds['min']

            # ctrl bounds
            u_max = np.full(self._dimu, cs.inf)  # do not bound control
            u_min = - u_max

            Uu[u_slice1:u_slice2] = u_max
            Ul[u_slice1:u_slice2] = u_min

            # force bounds
            force_bounds = constraints.bound_force_variables(min_f, 1500, k, [swing_t], [swing_id], self._ncontacts, self._dt)
            Fu[f_slice1:f_slice2] = force_bounds['max']
            Fl[f_slice1:f_slice2] = force_bounds['min']

            # Moving contact position bounds - only initial condition
            left_mov_contact_bounds = constraints.bound_moving_contact_variables(
                lmov_contact_initial[0],
                lmov_contact_initial[1],
                [np.full(3, -cs.inf), np.full(3, cs.inf)],
                [np.full(3, -0.3), np.full(3, 0.3)],
                k, self._knot_number)
            Pl_movu[u_slice1:u_slice2] = left_mov_contact_bounds['p_mov_max']
            Pl_movl[u_slice1:u_slice2] = left_mov_contact_bounds['p_mov_min']
            DPl_movu[u_slice1:u_slice2] = left_mov_contact_bounds['dp_mov_max']
            DPl_movl[u_slice1:u_slice2] = left_mov_contact_bounds['dp_mov_min']

            right_mov_contact_bounds = constraints.bound_moving_contact_variables(
                rmov_contact_initial[0],
                rmov_contact_initial[1],
                [np.full(3, -cs.inf), np.full(3, cs.inf)],
                [np.full(3, -0.3), np.full(3, 0.3)],
                k, self._knot_number)
            Pr_movu[u_slice1:u_slice2] = right_mov_contact_bounds['p_mov_max']
            Pr_movl[u_slice1:u_slice2] = right_mov_contact_bounds['p_mov_min']
            DPr_movu[u_slice1:u_slice2] = right_mov_contact_bounds['dp_mov_max']
            DPr_movl[u_slice1:u_slice2] = right_mov_contact_bounds['dp_mov_min']

            # contact positions
            contact_params = constraints.set_contact_parameters(
                contacts, [swing_id], [swing_tgt], [clearance_time], [clearance_swing_position], k, self._dt
            )
            P.append(contact_params)

            # constraints' bounds
            gl.append(np.zeros(6))  # newton-euler
            gu.append(np.zeros(6))
            if k > 0:
                gl.append(np.zeros(self._dimx))     # state constraint
                gu.append(np.zeros(self._dimx))
            if 0 < k < self._knot_number - 1:
                gl.append(np.zeros(3))      # 2 moving contacts
                gu.append(np.zeros(3))

                gl.append(np.zeros(3))
                gu.append(np.zeros(3))

            # box constraint - moving contact bounds
            gl.append(np.array([0.35, 0.0, 0.25]))
            gu.append(np.array([0.48, 0.3, 0.35]))

            gl.append(np.array([0.35, -0.3, 0.25]))
            gu.append(np.array([0.48, 0.0, 0.35]))
        # final constraints
        Xl[-6:] = [0.0 for i in range(6)]  # zero velocity and acceleration
        Xu[-6:] = [0.0 for i in range(6)]

        DPl_movl[-3:] = [0.0 for i in range(3)]   # zero velocity of the moving contact
        DPl_movu[-3:] = [0.0 for i in range(3)]

        DPr_movl[-3:] = [0.0 for i in range(3)]  # zero velocity of the moving contact
        DPr_movu[-3:] = [0.0 for i in range(3)]

        # initial guess
        v0 = np.zeros(self._nvars)

        # format bounds and params according to solver
        lbv = cs.vertcat(Xl, Ul, Fl, Pl_movl, Pr_movl, DPl_movl, DPr_movl)
        ubv = cs.vertcat(Xu, Uu, Fu, Pl_movu, Pr_movu, DPl_movu, DPr_movu)
        lbg = cs.vertcat(*gl)
        ubg = cs.vertcat(*gu)
        params = cs.vertcat(*P)

        # compute solution-call solver
        sol = self._solver(x0=v0, lbx=lbv, ubx=ubv, lbg=lbg, ubg=ubg, p=params)

        # plot state, forces, control input, quantities to be computed by evaluate function
        x_trj = cs.horzcat(self._trj['x'])  # pack states in a desired matrix
        f_trj = cs.horzcat(self._trj['F'])  # pack forces in a desired matrix
        u_trj = cs.horzcat(self._trj['u'])  # pack control inputs in a desired matrix
        p_lmov_trj = cs.horzcat(self._trj['P_mov_l'])  # pack moving contact trj in a desired matrix
        dp_lmov_trj = cs.horzcat(self._trj['DP_mov_l'])  # pack moving contact trj in a desired matrix
        p_rmov_trj = cs.horzcat(self._trj['P_mov_r'])  # pack moving contact trj in a desired matrix
        dp_rmov_trj = cs.horzcat(self._trj['DP_mov_r'])  # pack moving contact trj in a desired matrix

        # return values of the quantities *_trj
        return {
            'x': self.evaluate(sol['x'], x_trj),
            'F': self.evaluate(sol['x'], f_trj),
            'u': self.evaluate(sol['x'], u_trj),
            'Pl_mov': self.evaluate(sol['x'], p_lmov_trj),
            'Pr_mov': self.evaluate(sol['x'], p_rmov_trj),
            'DPl_mov': self.evaluate(sol['x'], dp_lmov_trj),
            'DPr_mov': self.evaluate(sol['x'], dp_rmov_trj)
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
        expr_fun = cs.Function('expr_fun', [self._nlp['x']], [expr], ['v'], ['expr'])

        expr_value = expr_fun(v=solution)['expr'].toarray()

        # make it a flat list
        expr_value_flat = [i for sublist in expr_value for i in sublist]

        return expr_value_flat

    def interpolate(self, solution, sw_curr, sw_tgt, clearance, sw_t, resol):
        """ Interpolate the trajectories generated by the solution of the problem

        Args:
            solution: solution of the problem (numerical values) is a directory
                solution['x'] 9x30 --> optimized states
                solution['f'] 12x30 --> optimized forces
                solution['u'] 9x30 --> optimized control
                ...
            sw_curr: current position of the foot to be swinged
            sw_tgt: target position of the foot to be swinged
            clearance: swing clearance
            sw_t: (start, stop) period of foot swinging in a global manner wrt to optimization problem
            resol: interpolation resolution (points per second)

        Returns: a dictionary with:
            time list for interpolation times (in sec)
            list of list with state trajectory points
            list of lists with forces' trajectory points
            list of lists with the swinging foot's trajectory points
            the moving contact trajectory (cubic spline) and first and second derivative
        """

        # start and end times of optimization problem
        t_tot = [0.0, self._problem_duration]

        # state
        state_trajectory = self.state_interpolation(solution=solution, resolution=resol)

        # forces
        forces_trajectory = self.forces_interpolation(solution=solution)

        # moving contacts
        left_moving_contact_trajectory = self.moving_contact_interpolation(
            p_mov_optimal=solution['Pl_mov'], dp_mov_optimal=solution['DPl_mov'], resolution=resol
        )

        right_moving_contact_trajectory = self.moving_contact_interpolation(
            p_mov_optimal=solution['Pr_mov'], dp_mov_optimal=solution['DPr_mov'], resolution=resol
        )

        # swing leg trajectory planning and interpolation
        # swing trajectory with intemediate point
        sw_interpl = interpol.swing_trj_triangle(sw_curr, sw_tgt, clearance, sw_t, t_tot, resol)
        # swing trajectory with spline optimization for z coordinate
        # sw_interpl = interpol.swing_trj_optimal_spline(sw_curr, sw_tgt, clearance, sw_t, t_tot, resol)

        return {
            't': self._t,
            'x': state_trajectory,
            'f': forces_trajectory,
            'p_mov_l': [i['p'] for i in left_moving_contact_trajectory],
            'dp_mov_l': [i['dp'] for i in left_moving_contact_trajectory],
            'ddp_mov_l': [i['ddp'] for i in left_moving_contact_trajectory],
            'p_mov_r': [i['p'] for i in right_moving_contact_trajectory],
            'dp_mov_r': [i['dp'] for i in right_moving_contact_trajectory],
            'ddp_mov_r': [i['ddp'] for i in right_moving_contact_trajectory],
            'sw': sw_interpl
        }

    def state_interpolation(self, solution, resolution):
        """
        Interpolate the state vector of the problem using the state equation
        :param solution: optimal solution
        :param resolution: resolution of the trajectory, points per secong
        :return: list of list with state trajectory points
        """

        delta_t = 1.0 / resolution  # dt for interpolation

        # intermediate points between two knots --> time interval * resolution
        self._n = int(self._dt * resolution)

        x_old = solution['x'][0:9]  # initial state
        x_all = []  # list to append all states

        for ii in range(self._knot_number):  # loop for knots

            # control input to change in every knot
            u_old = solution['u'][self._dimu * ii:self._dimu * (ii + 1)]

            for j in range(self._n):  # loop for interpolation points

                x_all.append(x_old)  # storing state in the list 600x9

                x_next = self._integrator(x_old, u_old, delta_t)  # next state
                x_old = x_next  # refreshing the current state

        # initialize state and time lists to gather the data
        int_state = [[] for i in range(self._dimx)]  # primary dimension = number of state components
        self._t = [(ii * delta_t) for ii in range(self._Nseg * self._n)]

        for i in range(self._dimx):  # loop for every component of the state vector
            for j in range(self._Nseg * self._n):  # loop for every point of interpolation

                # append the value of x_i component on j point of interpolation
                # in the element i of the list int_state
                int_state[i].append(x_all[j][i])

        return int_state

    def forces_interpolation(self, solution):
        """
        Linear interpolation of the optimal forces
        :param solution: optimal solution
        :return: list of lists with the force trajectories
        """

        force_func = [[] for i in range(self._dimf_tot)]  # list to store the splines
        int_force = [[] for i in range(self._dimf_tot)]  # list to store lists of points

        for i in range(self._dimf_tot):  # loop for each component of the force vector

            # append the spline (by casadi) in the i element of the list force_func
            force_func[i].append(cs.interpolant('X_CONT', 'linear',
                                                [self._tjunctions],
                                                solution['F'][i::self._dimf_tot]))

            # store the interpolation points for each force component in the i element of the list int_force
            # primary dimension = number of force components
            int_force[i] = force_func[i][0](self._t)

        return int_force

    def moving_contact_interpolation(self, p_mov_optimal, dp_mov_optimal, resolution):
        """
        Interpolation of the moving contact trajectory as cubic spline.
        :param p_mov_optimal: optimal solution for mov. contact position
        :param dp_mov_optimal: optimal solution for mov. contact velocity
        :param resolution: resolution for the trajectory, points per sec
        :return: Trajectories generated from the cubic spline and its 1st and 2nd derivatives
        """
        mov_cont_splines = []
        mov_cont_polynomials = []
        mov_cont_points = []

        for i in range(3):
            mov_cont_splines.append(cubic_spline.CubicSpline(p_mov_optimal[i::self._dimu],
                                                             dp_mov_optimal[i::self._dimu],
                                                             self._tjunctions))

            mov_cont_polynomials.append(mov_cont_splines[i].get_poly_objects())
            mov_cont_points.append(mov_cont_splines[i].get_spline_trajectory(resolution))

        return mov_cont_points

    def print_trj(self, solution, results, resol, contacts, swing_id, t_exec=0):
        '''

        Args:
            t_exec: time that trj execution stopped (because of early contact or other)
            results: results from interpolation
            resol: interpolation resol
            contacts: contact points
            swing_id: the id of the leg to be swinged
        Returns: prints the nominal interpolated trajectories

        '''

        # Interpolated state plot
        state_labels = ['CoM Position', 'CoM Velocity', 'CoM Acceleration']
        plt.figure()
        for i, name in enumerate(state_labels):
            plt.subplot(3, 1, i + 1)
            for j in range(self._dimc):
                plt.plot(results['t'], results['x'][self._dimc * i + j], '-')
                # plt.plot(self._time, solution['x'][self._dimc * i + j::self._dimx], 'o')
            plt.grid()
            plt.legend(['x', 'y', 'z'])
            # plt.legend(['x', 'xopt', 'y', 'yopt', 'z', 'zopt'])
            plt.title(name)
        plt.xlabel('Time [s]')
        # plt.savefig('../plots/step_state_trj.png')

        feet_labels = ['front left', 'front right', 'hind left', 'hind right']

        # Interpolated force plot
        plt.figure()
        for i, name in enumerate(feet_labels):
            plt.subplot(2, 2, i + 1)
            for k in range(3):
                plt.plot(results['t'], results['f'][3 * i + k], '-')
            plt.grid()
            plt.title(name)
            plt.legend([str(name) + '_x', str(name) + '_y', str(name) + '_z'])
        plt.xlabel('Time [s]')
        # plt.savefig('../plots/step_forces.png')

        # Interpolated moving contact trajectory
        mov_contact_labels = ['p_mov_l', 'dp_mov_l', 'ddp_mov_l', 'p_mov_r', 'dp_mov_r', 'ddp_mov_r']
        plt.figure()
        for i, name in enumerate(mov_contact_labels):
            plt.subplot(2, 3, i+1)
            for k in range(3):
                plt.plot(results['t'], results[name][k], '-')
                plt.grid()
                plt.legend(['x', 'y', 'z'])
            plt.ylabel(name)
            plt.suptitle('Moving Contact trajectory')
        plt.xlabel('Time [s]')

        # plot swing trajectory
        # All points to be published
        N_total = int(self._problem_duration * resol)  # total points --> total time * frequency
        s = np.linspace(0, self._problem_duration, N_total)
        coord_labels = ['x', 'y', 'z']
        plt.figure()
        for i, name in enumerate(coord_labels):
            plt.subplot(3, 1, i + 1)
            plt.plot(s, results['sw'][name])  # nominal trj
            plt.plot(s[0:t_exec], results['sw'][name][0:t_exec])  # executed trj
            plt.grid()
            plt.legend(['nominal', 'real'])
            plt.title('Trajectory ' + name)
        plt.xlabel('Time [s]')
        # plt.savefig('../plots/step_swing.png')

        # plot swing trajectory in two dimensions Z- X
        plt.figure()
        plt.plot(results['sw']['x'], results['sw']['z'])  # nominal trj
        plt.plot(results['sw']['x'][0:t_exec], results['sw']['z'][0:t_exec])  # real trj
        plt.grid()
        plt.legend(['nominal', 'real'])
        plt.title('Trajectory Z- X')
        plt.xlabel('X [m]')
        plt.ylabel('Z [m]')

        # Support polygon and CoM motion in the plane
        SuP_x_coords = [contacts[k][1] for k in range(4) if k not in [swing_id]]
        SuP_x_coords.append(SuP_x_coords[0])
        SuP_y_coords = [contacts[k][0] for k in range(4) if k not in [swing_id]]
        SuP_y_coords.append(SuP_y_coords[0])
        plt.figure()
        plt.plot(results['x'][1], results['x'][0], '--', linewidth=3)
        plt.plot(SuP_x_coords, SuP_y_coords, 'ro-', linewidth=0.8)
        plt.grid()
        plt.title('Support polygon and CoM')
        plt.xlabel('Y [m]')
        plt.ylabel('X [m]')
        plt.xlim(0.5, -0.5)
        plt.show()


if __name__ == "__main__":
    start_time = time.time()

    w = Walking(mass=95, N=40, dt=0.2, payload_mass=5.0)

    # initial state =
    c0 = np.array([0.107729, 0.0000907, -0.02118])
    dc0 = np.zeros(3)
    ddc0 = np.zeros(3)
    x_init = np.hstack([c0, dc0, ddc0])

    foot_contacts = [
        np.array([0.35, 0.35, -0.7187]),  # fl
        np.array([0.35, -0.35, -0.7187]),  # fr
        np.array([-0.35, 0.35, -0.7187]),  # hl
        np.array([-0.35, -0.35, -0.7187])  # hr
    ]

    # mov contacts
    lmoving_contact = [
        np.array([0.53, 0.179, 0.3]),
        np.zeros(3),
    ]

    rmoving_contact = [
        np.array([0.53, -0.179, 0.3]),
        np.zeros(3),
    ]

    moving_contact = [lmoving_contact, rmoving_contact]

    # swing id from 0 to 3
    sw_id = 0

    step_clear = 0.05

    # swing_target = np.array([-0.35, -0.35, -0.719])
    dx = 0.0
    dy = 0.0
    dz = 0.0
    swing_target = np.array([foot_contacts[sw_id][0] + dx, foot_contacts[sw_id][1] + dy, foot_contacts[sw_id][2] + dz])

    # swing_time = (1.5, 3.0)
    swing_time = [2.0, 5.0]

    # sol is the directory returned by solve class function contains state, forces, control values
    sol = w.solve(x0=x_init, contacts=foot_contacts, mov_contact_initial=moving_contact,
                  swing_id=sw_id, swing_tgt=swing_target, swing_clearance=step_clear,
                  swing_t=swing_time, min_f=100)

    # check time needed
    end_time = time.time()
    print('Total time is:', (end_time - start_time) * 1000, 'ms')

    # debug
    print("X0 is:", x_init)
    print("contacts is:", foot_contacts)
    print("swing id is:", sw_id)
    print("swing target is:", swing_target)
    print("swing time:", swing_time)

    # interpolate the values, pass values and interpolation resolution
    res = 300
    interpl = w.interpolate(sol, foot_contacts[sw_id], swing_target, step_clear, swing_time, res)

    # check time needed
    end_time = time.time()
    print('Total time for nlp formulation, solution and interpolation:', (end_time - start_time) * 1000, 'ms')

    print("Solution is:")
    print("State is:", sol['x'])
    print("Control is:", sol['u'])
    print("Forces are:", sol['F'])
    #print("Moving contact is:", sol['P_mov'])
    #print("Moving contact velocity is:", sol['DP_mov'])
    # print the results
    w.print_trj(sol, interpl, res, foot_contacts, sw_id)


