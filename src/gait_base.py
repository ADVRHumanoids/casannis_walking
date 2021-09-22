import casadi as cs
import numpy as np
from matplotlib import pyplot as plt
import constraints
import parameters
import costs
import trj_interpolation as interpol

from scipy.spatial.transform import Rotation


class Gait:
    """
    Assumptions:
      1) mass concentrated at com
      1) mass concentrated at com
      2) zero angular momentum
      3) point contacts

    Dynamics:
      1) input is com jerk
      2) dynamics is a triple integrator of com jerk
      3) there must be contact forces that
        - realize the motion
        - fulfil contact constraints (i.e. unilateral constraint)
    """

    def __init__(self, mass, N, dt, slope_deg=0):
        """Gait class constructor

        Args:
            mass (float): robot mass
            N (int): horizon length
            dt (float): discretization step
        """

        # identify if there is payload
        robot_mass = 112.0
        if mass > robot_mass:
            payload = mass - robot_mass
            self._CoM_vert_offset = CoM_vert_offset = 0.068
            print(payload, ' kg payload detected. Check input for robot mass.')

        else:
            print('No payload')
            self._CoM_vert_offset = CoM_vert_offset = 0.0

        self._gravity = parameters.get_gravity_acc_vector(slope_deg)

        self._Nseg = N
        self._dt = dt  # dt used for optimization knots
        self._problem_duration = N * dt
        self._mass = mass

        self._knot_number = knot_number = N + 1  # number of knots is one more than segments number
        self._tjunctions = [(i * dt) for i in range(knot_number)]  # time junctions from first to last

        # define dimensions
        sym_t = cs.SX
        self._dimc = dimc = 3
        self._dimx = dimx = 3 * dimc
        self._dimu = dimu = dimc
        self._dimf = dimf = dimc
        self._ncontacts = ncontacts = 4
        self._dimf_tot = dimf_tot = ncontacts * dimf

        # list with constraint names
        self._g_string = []

        # define cs variables
        delta_t = sym_t.sym('delta_t', 1)  # symbolic in order to pass values for optimization/interpolation
        c = sym_t.sym('c', dimc)
        dc = sym_t.sym('dc', dimc)
        ddc = sym_t.sym('ddc', dimc)
        x = cs.vertcat(c, dc, ddc)
        u = sym_t.sym('u', dimc)

        # base orientation
        theta = sym_t.sym('theta', dimc)
        dtheta = sym_t.sym('dtheta', dimc)
        ddtheta = sym_t.sym('ddtheta', dimc)
        x_euler = cs.vertcat(theta, dtheta, ddtheta)
        u_euler = sym_t.sym('u_euler', dimc)

        # expression for the integrated state
        xf = cs.vertcat(
            c + dc * delta_t + 0.5 * ddc * delta_t ** 2 + 1.0 / 6.0 * u * delta_t ** 3,  # position
            dc + ddc * delta_t + 0.5 * u * delta_t ** 2,  # velocity
            ddc + u * delta_t  # acceleration
        )

        # expression for the integrated state
        xf_euler = cs.vertcat(
            theta + dtheta * delta_t + 0.5 * ddtheta * delta_t ** 2 + 1.0 / 6.0 * u_euler * delta_t ** 3,  # position
            dtheta + ddtheta * delta_t + 0.5 * u_euler * delta_t ** 2,  # velocity
            ddtheta + u_euler * delta_t  # acceleration
        )

        # wrap the expression into a function
        self._integrator = cs.Function('integrator', [x, u, delta_t], [xf], ['x0', 'u', 'delta_t'], ['xf'])
        self._integrator_euler = cs.Function('integrator_euler', [x_euler, u_euler, delta_t], [xf_euler],
                                             ['x0_euler', 'u_euler', 'delta_t'], ['xf_euler'])

        # construct the optimization problem (variables, cost, constraints, bounds)
        X = sym_t.sym('X', knot_number * dimx)  # state is an SX for all knots
        U = sym_t.sym('U', knot_number * dimu)  # for all knots

        X_euler = sym_t.sym('X_euler', knot_number * dimx)  # state is an SX for all knots
        U_euler = sym_t.sym('U_euler', knot_number * dimu)  # for all knots

        F = sym_t.sym('F', knot_number * (ncontacts * dimf))  # for all knots

        P = list()
        g = list()  # list of constraint expressions
        J = list()  # list of cost function expressions

        self._trj = {
            'x': X,
            'u': U,
            'x_euler': X_euler,
            'u_euler': U_euler,
            'F': F
        }

        # iterate over knots starting from k = 0
        for k in range(knot_number):

            x_slice1 = k * dimx
            x_slice2 = (k + 1) * dimx
            u_slice1 = k * dimu
            u_slice2 = (k + 1) * dimu
            f_slice1 = k * dimf_tot
            f_slice2 = (k + 1) * dimf_tot

            # contact points
            p_k = sym_t.sym('p_' + str(k), ncontacts * dimc)
            P.append(p_k)

            # cost  function
            cost_function = 0.0
            cost_function += costs.penalize_xy_forces(1e-3, F[f_slice1:f_slice2])  # penalize xy forces
            cost_function += costs.penalize_horizontal_CoM_position(1e3, X[x_slice1:x_slice1 + 3],
                                                                    p_k)  # penalize CoM position
            cost_function += costs.penalize_vertical_CoM_position(1e3, X[x_slice1:x_slice1 + 3],
                                                                  p_k, payload_offset_z=CoM_vert_offset)
            cost_function += costs.penalize_quantity(1e-0, U[u_slice1:u_slice2],
                                                     k, knot_number)  # penalize CoM jerk, that is the control

            # orientation
            cost_function += costs.penalize_quantity(1e1, X_euler[x_slice1:x_slice1 + 3],
                                                     k, knot_number)  # penalize orientation state
            cost_function += costs.penalize_quantity(1e-0, U_euler[u_slice1:u_slice2],
                                                     k, knot_number)  # penalize CoM jerk, that is the control
            J.append(cost_function)

            # newton - euler dynamic constraints
            # newton_euler_constraint = constraints.newton_euler_constraint(
            #     X[x_slice1:x_slice2], mass, self._gravity, ncontacts, F[f_slice1:f_slice2], p_k
            # )

            newton_euler_constraint = constraints.SRBD_dynamics_constraint(
                X[x_slice1:x_slice2], X_euler[x_slice1:x_slice2], mass, self._gravity, ncontacts, F[f_slice1:f_slice2], p_k
            )

            self._g_string.extend(newton_euler_constraint['name'])
            g.append(newton_euler_constraint['newton'])
            g.append(newton_euler_constraint['euler'])

            # friction pyramid
            friction_pyramid_constraint = constraints.friction_pyramid(F[f_slice1:f_slice2], 0.3)
            self._g_string.extend(friction_pyramid_constraint['name'])
            g.append(np.array(friction_pyramid_constraint['constraint']))

            # state constraint (triple integrator)
            if k > 0:
                x_old = X[(k - 1) * dimx: x_slice1]  # save previous state
                u_old = U[(k - 1) * dimu: u_slice1]  # prev control
                x_curr = X[x_slice1:x_slice2]
                state_constraint = constraints.state_constraint(
                    self._integrator(x0=x_old, u=u_old, delta_t=dt)['xf'], x_curr)
                self._g_string.extend(state_constraint['name'])
                g.append(state_constraint['constraint'])

                # euler
                x_euler_old = X_euler[(k - 1) * dimx: x_slice1]  # save previous state
                u_euler_old = U_euler[(k - 1) * dimu: u_slice1]  # prev control
                x_euler_curr = X_euler[x_slice1:x_slice2]
                state_euler_constraint = constraints.state_constraint(
                    self._integrator_euler(x0_euler=x_euler_old, u_euler=u_euler_old, delta_t=dt)['xf_euler'], x_euler_curr)
                self._g_string.extend(state_constraint['name'])
                g.append(state_euler_constraint['constraint'])

        # construct the solver
        self._nlp = {
            'x': cs.vertcat(X, U, X_euler, U_euler, F),
            'f': sum(J),
            'g': cs.vertcat(*g),
            'p': cs.vertcat(*P)
        }

        # save dimensions
        self._nvars = self._nlp['x'].size1()
        self._nconstr = self._nlp['g'].size1()
        self._nparams = self._nlp['p'].size1()

        solver_options = {
            'ipopt.linear_solver': 'ma57',
            'ipopt.print_level': 5
        }

        self._solver = cs.nlpsol('solver', 'ipopt', self._nlp, solver_options)

    def solve(self, x0, x0_euler, contacts, swing_id, swing_tgt, swing_clearance, swing_t, min_f=50):
        """Solve the stepping problem

        Args:
            x0 ([type]): initial state (com position, velocity, acceleration)
            contacts ([type]): list of contact point positions
            swing_id ([type]): the indexes of the legs to be swinged from 0 to 3
            swing_tgt ([type]): the target footholds for the swing legs
            swing_clearance: clearance achieved from the highest point between initial and target position
            swing_t ([type]): list of lists with swing times in secs
            min_f: minimum threshold for forces in z direction
        """

        # lists for assigning bounds
        Xl = [0] * self._dimx * self._knot_number  # state lower bounds (for all knots)
        Xu = [0] * self._dimx * self._knot_number  # state upper bounds
        Ul = [0] * self._dimu * self._knot_number  # control lower bounds
        Uu = [0] * self._dimu * self._knot_number  # control upper bounds
        X_eulerl = [0] * self._dimx * self._knot_number  # state lower bounds (for all knots)
        X_euleru = [0] * self._dimx * self._knot_number  # state upper bounds
        U_eulerl = [0] * self._dimu * self._knot_number  # control lower bounds
        U_euleru = [0] * self._dimu * self._knot_number  # control upper bounds
        Fl = [0] * self._dimf_tot * self._knot_number  # force lower bounds
        Fu = [0] * self._dimf_tot * self._knot_number  # force upper bounds
        gl = list()  # constraint lower bounds
        gu = list()  # constraint upper bounds
        self._P = P = list()  # parameter values

        # time that maximum clearance occurs
        clearance_times = [0.5 * (x[0] + x[1]) for x in swing_t]

        # number of steps
        step_num = len(swing_id)

        # swing feet positions at maximum clearance
        clearance_swing_position = []

        for i in range(step_num):
            if contacts[swing_id[i]][2] >= swing_tgt[i][2]:
                clearance_swing_position.append(contacts[swing_id[i]][0:2].tolist() +
                                                [contacts[swing_id[i]][2] + swing_clearance])
            else:
                clearance_swing_position.append(contacts[swing_id[i]][0:2].tolist() +
                                                [swing_tgt[i][2] + swing_clearance])

        # compute mean horizontal position of final contacts
        final_contacts = []
        for i in range(4):
            if i in swing_id:
                final_contacts.append(swing_tgt[swing_id.index(i)])
            else:
                final_contacts.append(contacts[i])

        final_state = constraints.get_nominal_CoM_bounds_from_contacts(final_contacts,
                                                                       offset_from_payload=self._CoM_vert_offset)

        final_state_euler = [np.zeros(9), np.zeros(9)]

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
            state_bounds = constraints.bound_state_variables(x0, [np.full(9, -cs.inf), np.full(9, cs.inf)], k,
                                                             self._knot_number,
                                                             final_state)
            Xu[x_slice1:x_slice2] = state_bounds['max']
            Xl[x_slice1:x_slice2] = state_bounds['min']

            # ctrl bounds
            u_max = np.full(self._dimu, cs.inf)  # do not bound control
            u_min = -u_max

            Uu[u_slice1:u_slice2] = u_max
            Ul[u_slice1:u_slice2] = u_min

            # state EULER bounds
            state_euler_bounds = constraints.bound_state_variables(x0_euler, [np.full(9, -cs.inf), np.full(9, cs.inf)], k,
                                                                   self._knot_number)
            X_euleru[x_slice1:x_slice2] = state_euler_bounds['max']
            X_eulerl[x_slice1:x_slice2] = state_euler_bounds['min']

            # ctrl bounds
            u_euler_max = np.full(self._dimu, cs.inf)  # do not bound control
            u_euler_min = -u_max

            U_euleru[u_slice1:u_slice2] = u_euler_max
            U_eulerl[u_slice1:u_slice2] = u_euler_min

            # force bounds
            force_bounds = constraints.bound_force_variables(min_f, 1500, k, swing_t, swing_id, self._ncontacts,
                                                             self._dt, step_num)
            Fu[f_slice1:f_slice2] = force_bounds['max']
            Fl[f_slice1:f_slice2] = force_bounds['min']

            # contact positions
            contact_params = constraints.set_contact_parameters(
                contacts, swing_id, swing_tgt, clearance_times, clearance_swing_position, k, self._dt, step_num
            )
            P.append(contact_params)

            # constraint bounds (newton-euler eq.)
            gl.append(np.zeros(6))
            gu.append(np.zeros(6))

            # friction pyramid
            gl.append(np.array([-cs.inf, 0.0, -cs.inf, 0.0] * self._ncontacts))
            gu.append(np.array([0.0, cs.inf, 0.0, cs.inf] * self._ncontacts))

            if k > 0:  # state constraint
                gl.append(np.zeros(self._dimx))
                gu.append(np.zeros(self._dimx))

                gl.append(np.zeros(self._dimx))     # euler
                gu.append(np.zeros(self._dimx))

        # final constraints
        Xl[-6:] = [0.0 for i in range(6)]  # zero velocity and acceleration
        Xu[-6:] = [0.0 for i in range(6)]

        X_eulerl[-6:] = [0.0 for i in range(6)]  # zero velocity and acceleration
        X_euleru[-6:] = [0.0 for i in range(6)]

        # initial guess
        v0 = np.zeros(self._nvars)

        # format bounds and params according to solver
        lbv = cs.vertcat(Xl, Ul, X_eulerl, U_eulerl, Fl)
        ubv = cs.vertcat(Xu, Uu, X_euleru, U_euleru, Fu)
        self._lbg = lbg = cs.vertcat(*gl)
        self._ubg = ubg = cs.vertcat(*gu)
        params = cs.vertcat(*P)

        # compute solution-call solver
        self._sol = sol = self._solver(x0=v0, lbx=lbv, ubx=ubv, lbg=lbg, ubg=ubg, p=params)

        # plot state, forces, control input, quantities to be computed by evaluate function
        x_trj = cs.horzcat(self._trj['x'])  # pack states in a desired matrix
        f_trj = cs.horzcat(self._trj['F'])  # pack forces in a desired matrix
        u_trj = cs.horzcat(self._trj['u'])  # pack control inputs in a desired matrix

        x_euler_trj = cs.horzcat(self._trj['x_euler'])  # pack states in a desired matrix
        u_euler_trj = cs.horzcat(self._trj['u_euler'])  # pack control inputs in a desired matrix

        # return values of the quantities *_trj
        return {
            'x': self.evaluate(sol['x'], x_trj),
            'F': self.evaluate(sol['x'], f_trj),
            'u': self.evaluate(sol['x'], u_trj),
            'x_euler': self.evaluate(sol['x'], x_euler_trj),
            'u_euler': self.evaluate(sol['x'], u_euler_trj)
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

    def evaluate_with_params(self, solution, parameters, expr):
        """ Evaluate a given expression

        Args:
            solution: given solution
            parameters: params of nlp
            expr: expression to be evaluated

        Returns:
            Numerical value of the given expression

        """

        # casadi function that symbolically maps the _nlp to the given expression
        expr_fun = cs.Function('expr_fun', [self._nlp['x'], self._nlp['p']], [expr], ['v', 'p'], ['expr'])

        expr_value = expr_fun(v=solution, p=parameters)['expr'].toarray()

        # make it a flat list
        expr_value_flat = [i for sublist in expr_value for i in sublist]

        return expr_value_flat

    def compute_constraint_violation(self, solution, parameters_list, constraints_list, lower_bounds, upper_bounds):
        '''
        Compute the constraint violation and check if it is accepted or not. Plot constraints and bounds
        :param solution: symbolical expression of the solution
        :param parameters_list: list of nlp params
        :param constraints_list: symbolical expression of constraints
        :param lower_bounds: list with lower constraint bounds
        :param upper_bounds: list with upper constraint bounds
        :return: The constraints that are violated and the constraints plots.
        '''

        # evaluate the constraints
        constraint_violation = self.evaluate_with_params(solution, parameters_list, constraints_list)

        # default tolerance of ipopt for the unscaled problem
        constr_viol_tolerance = 1e-4

        # loop over all constraints
        for i in range(self._nconstr):

            # check for violations
            if constraint_violation[i] < lower_bounds[i] - constr_viol_tolerance or \
                    constraint_violation[i] > upper_bounds[i] + constr_viol_tolerance:
                print('Violated constraint: ', self._g_string[i])

        # get names and sizes of constraints (in dictionary) to plot them
        constraint_names_sizes = parameters.get_constraint_names(formulation='gait')

        # loop over constraint names (e.g. 'dynamics')
        for i, name in enumerate(constraint_names_sizes):

            # indices for each cosntraint
            constraint_indices = np.where(np.isin(self._g_string, name))[0].tolist()

            # size of each constraint (#coordinates) to create subplots
            subplot_size = constraint_names_sizes[name]

            plt.figure()
            for ii in range(subplot_size):  # loop over coordinates
                plt.subplot(subplot_size, 1, ii + 1)
                plt.plot([constraint_violation[j] for j in constraint_indices[ii::subplot_size]], '.-', label=name)
                plt.plot([lower_bounds[j] - constr_viol_tolerance for j in constraint_indices[ii::subplot_size]],
                         'r', label='lower_bound')
                plt.plot([upper_bounds[j] + constr_viol_tolerance for j in constraint_indices[ii::subplot_size]],
                         'r', label='upper_bound')
            plt.suptitle(name)
            plt.legend()
        # plt.show()

    def interpolate(self, solution, sw_curr, sw_tgt, clearance, sw_t, resol):
        """ Interpolate the trajectories generated by the solution of the problem

        Args:
            solution: solution of the problem (numerical values) is a directory
                solution['x'] 9x30 --> optimized states
                solution['f'] 12x30 --> optimized forces
                solution['u'] 9x30 --> optimized control
            sw_curr: initial position of the feet to be swinged
            sw_tgt: target position of the feet to be swinged
            clearance: swing clearance
            sw_t: list of lists with swing times in secs e.g. [[a, b], [c, d]]
            resol: interpolation resolution (points per second)

        Returns: a dictionary with:
            time list for interpolation times (in sec)
            list of list with state trajectory points
            list of lists with forces' trajectory points
            list of lists with the swinging feet's trajectory points

        """

        # start and end times of optimization problem
        t_tot = [0.0, self._problem_duration]

        # state
        state_trajectory, state_euler_trajectory = self.state_interpolation(solution=solution, resolution=resol)

        # euler to quaternion
        state_quaternion_trajectory = self.get_quat_from_euler(state_euler_trajectory[0:3])

        # forces
        forces_trajectory = self.forces_interpolation(solution=solution)

        # swing leg trajectory planning & interpolation
        sw_interpl = []
        for i in range(len(sw_curr)):
            # swing trajectories with one intemediate point
            sw_interpl.append(interpol.swing_trj_triangle(sw_curr[i], sw_tgt[i], clearance, sw_t[i], t_tot, resol))
            # spline optimization
            # sw_interpl.append(interpol.swing_trj_optimal_spline(sw_curr[i], sw_tgt[i], clearance, sw_t[i], t_tot, resol))

        return {
            't': self._t,
            'x': state_trajectory,
            'x_euler': state_euler_trajectory,
            'orient_quaternion': state_quaternion_trajectory,
            'f': forces_trajectory,
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

        x_euler_old = solution['x_euler'][0:9]  # initial state
        x_euler_all = []  # list to append all states

        for ii in range(self._knot_number):  # loop for knots
            # control input to change in every knot
            u_old = solution['u'][self._dimu * ii:self._dimu * (ii + 1)]
            u_euler_old = solution['u_euler'][self._dimu * ii:self._dimu * (ii + 1)]

            for j in range(self._n):  # loop for interpolation points
                x_all.append(x_old)  # storing state in the list 600x9
                x_next = self._integrator(x_old, u_old, delta_t)  # next state
                x_old = x_next  # refreshing the current state

                x_euler_all.append(x_euler_old)  # storing state in the list 600x9
                x_euler_next = self._integrator_euler(x_euler_old, u_euler_old, delta_t)  # next state
                x_euler_old = x_euler_next  # refreshing the current state

        # initialize state and time lists to gather the data
        int_state = [[] for i in range(self._dimx)]  # primary dimension = number of state components
        int_state_euler = [[] for i in range(self._dimx)]  # primary dimension = number of state components
        self._t = [(ii * delta_t) for ii in range(self._Nseg * self._n)]

        for i in range(self._dimx):  # loop for every component of the state vector
            for j in range(self._Nseg * self._n):  # loop for every point of interpolation
                # append the value of x_i component on j point of interpolation
                # in the element i of the list int_state
                int_state[i].append(x_all[j][i])
                int_state_euler[i].append(x_euler_all[j][i])

        return int_state, int_state_euler

    def get_quat_from_euler(self, euler_orient_trj):

        quaternion_trj = []
        for j in range(self._Nseg * self._n):  # loop for every point of interpolation
            quaternion_trj.append(Rotation.from_euler('xyz', [coord[j] for coord in euler_orient_trj], degrees=False).as_quat())

        return quaternion_trj

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

    def print_trj(self, solution, results, resol, contacts, sw_id, t_exec=[0, 0, 0, 0]):
        '''
        Args:
            solution: optimized decision variables
            t_exec: list of last trj points that were executed (because of early contact or other)
            results: results from interpolation
            resol: interpolation resol
            publish_freq: publish frequency - applies only when trajectories are interfaced with rostopics,
                else publish_freq = resol

        Returns: prints the nominal interpolated trajectories

        '''

        # Interpolated state plot
        state_labels = ['CoM Position', 'CoM Velocity', 'CoM Acceleration']
        plt.figure()
        for i, name in enumerate(state_labels):
            plt.subplot(3, 1, i + 1)
            for j in range(self._dimc):
                plt.plot(results['t'], results['x'][self._dimc * i + j], '-')
                # plt.plot(self._tjunctions, solution['x'][self._dimc * i + j::self._dimx], 'o')
            plt.grid()
            plt.legend(['x', 'y', 'z'])
            plt.title(name)
        plt.xlabel('Time [s]')
        # plt.savefig('../plots/gait_state_trj.png')

        # Euler state plot
        euler_state_labels = ['Theta', 'Theta dot', 'Theta ddot']
        plt.figure()
        for i, name in enumerate(euler_state_labels):
            plt.subplot(3, 1, i + 1)
            for j in range(self._dimc):
                plt.plot(results['t'], results['x_euler'][self._dimc * i + j], '-')
                # plt.plot(self._tjunctions, solution['x_euler'][self._dimc * i + j::self._dimx], '.-')
            plt.grid()
            plt.legend(['x', 'y', 'z'])
            plt.title(name)
        plt.xlabel('Time [s]')

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
        # plt.savefig('../plots/gait_forces.png')

        # # plot swing trajectory
        # # All points to be published
        # N_total = int(self._problem_duration * resol)  # total points --> total time * frequency
        # s = np.linspace(0, self._problem_duration, N_total)
        # coord_labels = ['x', 'y', 'z']
        # for j in range(len(results['sw'])):
        #     plt.figure()
        #     for i, name in enumerate(coord_labels):
        #         plt.subplot(3, 1, i + 1)
        #         plt.plot(s, results['sw'][j][name])  # nominal trj
        #         plt.plot(s[0:t_exec[j]], results['sw'][j][name][0:t_exec[j]])  # executed trj
        #         plt.grid()
        #         plt.legend(['nominal', 'real'])
        #         plt.title('Trajectory ' + name)
        #     plt.xlabel('Time [s]')
        #     # plt.savefig('../plots/gait_swing.png')
        #
        # # plot swing trajectory in two dimensions Z - X
        # plt.figure()
        # for j in range(len(results['sw'])):
        #     plt.subplot(2, 2, j + 1)
        #     plt.plot(results['sw'][j]['x'], results['sw'][j]['z'])  # nominal trj
        #     plt.plot(results['sw'][j]['x'][0:t_exec[j]], results['sw'][j]['z'][0:t_exec[j]])  # real trj
        #     plt.grid()
        #     plt.legend(['nominal', 'real'])
        #     plt.title('Trajectory Z- X')
        #     plt.xlabel('X [m]')
        #     plt.ylabel('Z [m]')
        #     # plt.savefig('../plots/gait_swing_zx.png')

        # Support polygon and CoM motion in the plane
        color_labels = ['red', 'green', 'blue', 'yellow']
        line_labels = ['-', '--', '-.', ':']
        plt.figure()
        for i in range(len(sw_id)):
            SuP_x_coords = [contacts[k][1] for k in range(4) if k not in [sw_id[i]]]
            SuP_x_coords.append(SuP_x_coords[0])
            SuP_y_coords = [contacts[k][0] for k in range(4) if k not in [sw_id[i]]]
            SuP_y_coords.append(SuP_y_coords[0])
            plt.plot(SuP_x_coords, SuP_y_coords, line_labels[0], linewidth=2 - 0.4 * i, color=color_labels[i])
        plt.plot(results['x'][1], results['x'][0], '--', linewidth=3)  # robot links - based CoM
        plt.grid()
        plt.title('Support polygon and CoM')
        plt.xlabel('Y [m]')
        plt.ylabel('X [m]')
        plt.xlim(0.5, -0.5)

        # compute the violated constraints and plot
        self.compute_constraint_violation(self._sol['x'], np.array(self._P).flatten(),
                                          self._nlp['g'], self._lbg, self._ubg)

        plt.show()


if __name__ == "__main__":

    # initial state
    # c0 = np.array([-0.00629, -0.03317, 0.01687])
    # c0 = np.array([0.107729, 0.0000907, -0.02118])
    c0 = np.array([0.0922, 0.0009, -0.0222])
    # c0 = np.array([-0.03, -0.04, 0.01687])
    dc0 = np.zeros(3)
    ddc0 = np.zeros(3)
    x_init = np.hstack([c0, dc0, ddc0])

    # euler initial state
    theta0 = np.zeros(3)
    dtheta0 = np.zeros(3)
    ddtheta0 = np.zeros(3)
    x_euler_init = np.hstack([theta0, dtheta0, ddtheta0])

    foot_contacts = [
        np.array([0.35, 0.35, -0.7187]),  # fl
        np.array([0.35, -0.35, -0.7187]),  # fr
        np.array([-0.35, 0.35, -0.7187]),  # hl
        np.array([-0.35, -0.35, -0.7187])  # hr
    ]

    # swing id from 0 to 3
    # sw_id = 2
    sw_id = [0, 1, 2, 3]

    step_num = len(sw_id)

    # swing_target = np.array([-0.35, -0.35, -0.719])
    dx = 0.1
    dy = 0.0
    dz = 0.0
    swing_target = np.array(
        [[foot_contacts[sw_id[i]][0] + dx, foot_contacts[sw_id[i]][1] + dy, foot_contacts[sw_id[i]][2] + dz]
         for i in range(len(sw_id))])

    # swing_time
    # swing_time = [[1.0, 4.0], [5.0, 8.0]]
    swing_time = [[1.0, 2.5], [3.5, 5.0], [6.0, 7.5], [8.5, 10.0]]

    # swing_time = [[1.0, 4.0], [5.0, 8.0], [9.0, 12.0], [13.0, 16.0]]
    step_clear = 0.05

    w = Gait(mass=112, N=int((swing_time[-1][1] + 1.0) / 0.2), dt=0.2, slope_deg=0)

    # sol is the directory returned by solve class function contains state, forces, control values
    sol = w.solve(x0=x_init, x0_euler=x_euler_init, contacts=foot_contacts, swing_id=sw_id, swing_tgt=swing_target,
                  swing_clearance=step_clear, swing_t=swing_time, min_f=100)
    # debug
    print("X0 is:", x_init)
    print("contacts is:", foot_contacts)
    print("swing id is:", sw_id)
    print("swing target is:", swing_target)
    print("swing time:", swing_time)
    print("Solution:", sol)

    # interpolate the values, pass values and interpolation resolution
    res = 300

    swing_currents = []
    for i in range(step_num):
        swing_currents.append(foot_contacts[sw_id[i]])
    interpl = w.interpolate(sol, swing_currents, swing_target, step_clear, swing_time, res)

    # print the results
    w.print_trj(sol, interpl, res, foot_contacts, sw_id)



