import casadi as cs
import numpy as np
from matplotlib import pyplot as plt
import trj_interpolation as interpol
import matplotlib

import costs
import constraints
import parameters

import cubic_hermite_polynomial as cubic_spline
import time

class Gait(object):
    """
    TODO:
    1) Formulate box constraint that will exclude a central box of torso
    2) trade-off between CoM and arm movement
    3) create node to publish range of motion boxes in RViz


    3) formulate analytical cost - integrated residual minimization
    4) Study the dynamics of the problem and compare different possibilities
    5) Add pelvis orientation decision variables
    6) Try to optimize for footholds

    """

    def __init__(self, mass, N, dt, payload_masses, slope_deg=0, conservative_box=True):
        """Gait class constructor

        Args:
            mass (float): robot mass
            N (int): horizon length
            dt (float): discretization step
            payload_masses: mass of the payloads in left and right arm
            slope_deg: inclination of terrain in degrees (only in x direction)
            conservative_box: if true the arm EE box constraints are more conservative and far from robot chest
        """

        self._box_conservative = conservative_box

        self._gravity = parameters.get_gravity_acc_vector(slope_deg)
        self._Nseg = N
        self._dt = dt  # dt used for optimization knots
        self._problem_duration = N * dt
        self._mass = mass

        self._knot_number = knot_number = N + 1  # number of knots is one more than segments number
        self._tjunctions = [(i * dt) for i in range(knot_number)]  # time junctions from first to last

        # mass of the payload
        self._payload_mass_l = payload_masses[0]
        self._payload_mass_r = payload_masses[1]

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

        # wrap the expression into a function
        self._integrator = cs.Function('integrator', [x, u, delta_t], [xf], ['x0', 'u', 'delta_t'], ['xf'])

        # construct the optimization problem (variables, cost, constraints, bounds)
        X = sym_t.sym('X', knot_number * dimx)  # state is an SX for all knots
        U = sym_t.sym('U', knot_number * dimu)  # for all knots
        F = sym_t.sym('F', knot_number * (ncontacts * dimf))  # for all knots

        # moving contact
        P_mov_l = sym_t.sym('P_mov_l', knot_number * dimp_mov)  # position knots for the virtual contact
        P_mov_r = sym_t.sym('P_mov_r', knot_number * dimp_mov)  # position knots for the virtual contact
        DP_mov_l = sym_t.sym('DP_mov_l', knot_number * dimp_mov)  # velocity knots for the virtual contact
        DP_mov_r = sym_t.sym('DP_mov_r', knot_number * dimp_mov)  # velocity knots for the virtual contact
        f_pay_l = self._payload_mass_l * self._gravity  # virtual force
        f_pay_r = self._payload_mass_r * self._gravity  # virtual force

        P = list()
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

        # get default arm position for final penalty
        arms_default_pos = parameters.get_arm_default_pos('forward', self._box_conservative)

        # iterate over knots starting from k = 0
        for k in range(knot_number):

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
            cost_function += costs.penalize_xy_forces(1e-3, F[f_slice1:f_slice2])  # penalize xy forces
            # penalize CoM position
            cost_function += costs.penalize_horizontal_CoM_position(1e3, X[x_slice1:x_slice1 + 3], p_k)
            cost_function += costs.penalize_vertical_CoM_position(1e3, X[x_slice1:x_slice1 + 3], p_k)
            cost_function += costs.penalize_quantity(1e-0, U[u_slice1:u_slice2],
                                                     k, knot_number)  # penalize CoM jerk, that is the control

            # analytical costs for moving contact acceleration
            if k < knot_number - 1:
                weights = [1e1 for i in range(3)]
                l_acceleration_analytical_cost = costs.get_analytical_cost_3D(weights,
                                                                              P_mov_l[u_slice1:u_slice2 + 3],
                                                                              DP_mov_l[u_slice1:u_slice2 + 3],
                                                                              self._dt, k, 2)
                r_acceleration_analytical_cost = costs.get_analytical_cost_3D(weights,
                                                                              P_mov_r[u_slice1:u_slice2 + 3],
                                                                              DP_mov_r[u_slice1:u_slice2 + 3],
                                                                              self._dt, k, 2)
                cost_function += l_acceleration_analytical_cost['x'] + r_acceleration_analytical_cost['x']
                cost_function += l_acceleration_analytical_cost['y'] + r_acceleration_analytical_cost['y']
                cost_function += l_acceleration_analytical_cost['z'] + r_acceleration_analytical_cost['z']

            # hands penalization over whole trajectory and high final penalty
            cost_fraction = 0.0  # this determines the penalty for the trj except the final knot which is high
            cost_hands = cost_fraction * 1e3
            default_lmov_contact = P_mov_l[u_slice1:u_slice2] - X[x_slice1:x_slice1 + 3] - arms_default_pos['left']
            default_rmov_contact = P_mov_r[u_slice1:u_slice2] - X[x_slice1:x_slice1 + 3] - arms_default_pos['right']
            cost_function += costs.penalize_quantity(cost_hands, default_lmov_contact, k, knot_number, final_weight=1e3)
            cost_function += costs.penalize_quantity(cost_hands, default_rmov_contact, k, knot_number, final_weight=1e3)

            J.append(cost_function)

            # newton - euler dynamic constraints
            newton_euler_constraint = constraints.newton_euler_constraint(
                X[x_slice1:x_slice2], mass, self._gravity, ncontacts, F[f_slice1:f_slice2],
                p_k, P_mov_l[u_slice1:u_slice2], P_mov_r[u_slice1:u_slice2],
                f_pay_l, f_pay_r
            )
            g.append(newton_euler_constraint['newton'])
            g.append(newton_euler_constraint['euler'])

            # friction pyramid
            friction_pyramid_constraint = constraints.friction_pyramid(F[f_slice1:f_slice2], 0.3)
            g.append(np.array(friction_pyramid_constraint))

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
            swing_id ([type]): the indexes of the legs to be swinged from 0 to 3
            swing_tgt ([type]): the target footholds for the swing legs
            swing_clearance: clearance achieved from the highest point between initial and target position
            swing_t ([type]): list of lists with swing times in secs
            min_f: minimum threshold for forces in z direction
        """

        # grab moving contacts from the list
        lmov_contact_initial = mov_contact_initial[0]
        rmov_contact_initial = mov_contact_initial[1]

        # lists for assigning bounds
        Xl = [0] * self._dimx * self._knot_number  # state lower bounds (for all knots)
        Xu = [0] * self._dimx * self._knot_number  # state upper bounds
        Ul = [0] * self._dimu * self._knot_number  # control lower bounds
        Uu = [0] * self._dimu * self._knot_number  # control upper bounds
        Fl = [0] * self._dimf_tot * self._knot_number  # force lower bounds
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

        gl = list()  # constraint lower bounds
        gu = list()  # constraint upper bounds
        P = list()  # parameter values

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
        print("Final contacts are", final_contacts)

        final_state = constraints.get_nominal_CoM_bounds_from_contacts(final_contacts)

        # get bounds for arms' box constraints
        arm_bounds = parameters.get_arm_box_bounds('forward', self._box_conservative)

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
            state_bounds = constraints.bound_state_variables(x0, [np.full(9, -cs.inf), np.full(9, cs.inf)],
                                                             k, self._knot_number, final_state)
            Xu[x_slice1:x_slice2] = state_bounds['max']
            Xl[x_slice1:x_slice2] = state_bounds['min']

            # ctrl bounds
            u_max = np.full(self._dimu, cs.inf)  # do not bound control
            u_min = -u_max

            Uu[u_slice1:u_slice2] = u_max
            Ul[u_slice1:u_slice2] = u_min

            # force bounds
            force_bounds = constraints.bound_force_variables(min_f, 1500, k, swing_t, swing_id, self._ncontacts,
                                                             self._dt, step_num)
            Fu[f_slice1:f_slice2] = force_bounds['max']
            Fl[f_slice1:f_slice2] = force_bounds['min']

            # Moving contact position bounds - only initial condition
            left_mov_contact_bounds = constraints.bound_moving_contact_variables(
                lmov_contact_initial[0],
                lmov_contact_initial[1],
                [np.full(3, -cs.inf), np.full(3, cs.inf)],
                [np.full(3, -0.5), np.full(3, 0.5)],
                k, self._knot_number)
            Pl_movu[u_slice1:u_slice2] = left_mov_contact_bounds['p_mov_max']
            Pl_movl[u_slice1:u_slice2] = left_mov_contact_bounds['p_mov_min']
            DPl_movu[u_slice1:u_slice2] = left_mov_contact_bounds['dp_mov_max']
            DPl_movl[u_slice1:u_slice2] = left_mov_contact_bounds['dp_mov_min']

            right_mov_contact_bounds = constraints.bound_moving_contact_variables(
                rmov_contact_initial[0],
                rmov_contact_initial[1],
                [np.full(3, -cs.inf), np.full(3, cs.inf)],
                [np.full(3, -0.5), np.full(3, 0.5)],
                k, self._knot_number)
            Pr_movu[u_slice1:u_slice2] = right_mov_contact_bounds['p_mov_max']
            Pr_movl[u_slice1:u_slice2] = right_mov_contact_bounds['p_mov_min']
            DPr_movu[u_slice1:u_slice2] = right_mov_contact_bounds['dp_mov_max']
            DPr_movl[u_slice1:u_slice2] = right_mov_contact_bounds['dp_mov_min']

            # foothold positions
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

            if k > 0:       # state constraint
                gl.append(np.zeros(self._dimx))
                gu.append(np.zeros(self._dimx))

            if 0 < k < self._knot_number - 1:
                gl.append(np.zeros(6))  # 2 moving contacts spline acc continuity
                gu.append(np.zeros(6))

            # box constraint - moving contact bounds
            gl.append(arm_bounds['left_l'])
            gu.append(arm_bounds['left_u'])

            gl.append(arm_bounds['right_l'])
            gu.append(arm_bounds['right_u'])

        # final constraints
        Xl[-6:] = [0.0 for i in range(6)]  # zero velocity and acceleration
        Xu[-6:] = [0.0 for i in range(6)]

        DPl_movl[-3:] = [0.0 for i in range(3)]  # zero velocity of the moving contact
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

        # casadi function that symbolically maps the _nlp['x'] to the given expression
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
            if constraint_violation[i] < lower_bounds[i] - constr_viol_tolerance or\
                    constraint_violation[i] > upper_bounds[i] + constr_viol_tolerance:

                print('Violated constraint: ', self._g_string[i])

        # get names and sizes of constraints (in dictionary) to plot them
        constraint_names_sizes = parameters.get_constraint_names(formulation='payload')

        # loop over constraint names (e.g. 'dynamics')
        for i, name in enumerate(constraint_names_sizes):

            # indices for each cosntraint
            constraint_indices = np.where(np.isin(self._g_string, name))[0].tolist()

            # size of each constraint (#coordinates) to create subplots
            subplot_size = constraint_names_sizes[name]
            if subplot_size >= 2:
                column_size = 2.0 + (subplot_size%2)
            else:
                column_size = 1
            plt.figure()
            matplotlib.rc('xtick', labelsize=16)
            matplotlib.rc('ytick', labelsize=16)
            for ii in range(subplot_size):  # loop over coordinates
                plt.subplot(subplot_size/column_size, column_size, ii+1)
                plt.plot([constraint_violation[j] for j in constraint_indices[ii::subplot_size]], '.-', markersize=10, label=name, linewidth=3)
                plt.plot([lower_bounds[j] - constr_viol_tolerance for j in constraint_indices[ii::subplot_size]],
                         'r', label='lower_bound', linewidth=3)
                plt.plot([upper_bounds[j] + constr_viol_tolerance for j in constraint_indices[ii::subplot_size]],
                         'r', label='upper_bound', linewidth=3)
            plt.suptitle(name)
            plt.legend()
        # plt.show()

    def interpolate(self, solution, sw_curr, sw_tgt, clearance, sw_t, resol, feet_ee_swing_trj=None, shift_time=None,
                    swing_default_dur=2.0, skip_useless=False):

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
            feet_ee_swing_trj: previous swing trj (for receding horizon)
            shift_time: time to shift the previous swing trj (for receding horizon)
            swing_default_dur: durations of swing phases
            skip_useless: if True then skip quantities that are not passed to the WBC like CoM derivatives and forces

        Returns: a dictionary with:
            time list for interpolation times (in sec)
            list of list with state trajectory points
            list of lists with forces' trajectory points
            list of lists with the swinging feet's trajectory points, ordered according to the swing order

        """

        # start and end times of optimization problem
        t_tot = [0.0, self._problem_duration]

        if skip_useless is True:
            # state
            state_trajectory = self.state_interpolation(solution=solution, resolution=resol, active_components=3)
            # forces
            forces_trajectory = self.forces_interpolation(solution=solution, active_components=0)
        else:
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

        swing_phase_num = len(sw_curr)
        # if feet_ee_swing_trj is None:
        # swing leg trajectory planning & interpolation
        sw_interpl = []
        for i in range(swing_phase_num):

            # if whole swing phase is within the horizon
            current_swing_duration = sw_t[i][1] - sw_t[i][0]
            if round(current_swing_duration, 2) == swing_default_dur:
                # swing trajectories with one intemediate point
                sw_interpl.append(interpol.swing_trj_triangle(sw_curr[i], sw_tgt[i], clearance, sw_t[i], t_tot, resol))

            else:       # if part of swing phase is not in the horizon
                # if it is the first swing phase then shift previous swing trj and copy final trj point
                # for the points needed so as to have size of the horizon
                if i == 0:
                    elapsed_swing_time = swing_default_dur - (sw_t[i][1] - sw_t[i][0])
                    elapsed_trj_points = int(shift_time * resol)
                    partly_elapsed_swing = {
                        'x': feet_ee_swing_trj[i]['x'][elapsed_trj_points:] + [feet_ee_swing_trj[i]['x'][-1]] * elapsed_trj_points,
                        'y': feet_ee_swing_trj[i]['y'][elapsed_trj_points:] + [feet_ee_swing_trj[i]['y'][-1]] * elapsed_trj_points,
                        'z': feet_ee_swing_trj[i]['z'][elapsed_trj_points:] + [feet_ee_swing_trj[i]['z'][-1]] * elapsed_trj_points
                    }
                    sw_interpl.append(partly_elapsed_swing)

                else:       # else it is the last swing phase, interpolate as if it was the whole swing phase
                    whole_future_swing_t = [sw_t[-1][0], sw_t[-1][0] + swing_default_dur]
                    sw_interpl.append(
                        interpol.swing_trj_triangle(sw_curr[i], sw_tgt[i], clearance, whole_future_swing_t,
                                                    t_tot, resol, t_tot[1]))

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

    def state_interpolation(self, solution, resolution, active_components=None):
        """
        Interpolate the state vector of the problem using the state equation
        :param active_components: the number of first components of CoM state to interpolated
        :param solution: optimal solution
        :param resolution: resolution of the trajectory, points per secong
        :return: list of list with state trajectory points
        """

        if active_components is None:
            active_components = self._dimx

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
        int_state = []      # primary dimension = number of state components
        for i in range(active_components):  # loop for every component of the state vector
            int_state.append([])    # convert to 2d list
            self._t = []
            for j in range(self._Nseg * self._n):  # loop for every point of interpolation
                # append the value of x_i component on j point of interpolation
                # in the element i of the list int_state
                int_state[i].append(x_all[j][i])

                self._t.append(delta_t*j)

        return int_state

    def forces_interpolation(self, solution, active_components=None):
        """
        Linear interpolation of the optimal forces
        :param active_components: components of force vector that are to be interpolated
        :param solution: optimal solution
        :return: list of lists with the force trajectories
        """

        if active_components is None:
            active_components = self._dimf_tot

        force_func = []     # list to store the splines
        int_force = []      # list to store lists of points

        for i in range(active_components):  # loop for each component of the force vector

            force_func.append([])   # convert them to 2d lists
            int_force.append([])

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
            pp = p_mov_optimal[i::self._dimu]
            dppp = dp_mov_optimal[i::self._dimu]
            ttt = self._tjunctions
            mov_cont_splines.append(cubic_spline.CubicSpline(p_mov_optimal[i::self._dimu],
                                                             dp_mov_optimal[i::self._dimu],
                                                             self._tjunctions))

            mov_cont_polynomials.append(mov_cont_splines[i].get_poly_objects())
            mov_cont_points.append(mov_cont_splines[i].get_spline_trajectory(resolution))

        return mov_cont_points

    def compute_equivalent_CoM(self, masses, xyz_trajectories):
        '''
        Function that computes the equivalent CoM from a number of masses that follow trajectories.
        Used for printing.
        :param masses: list of masses that consist the system
        :param xyz_trajectories: [mass1, ... massi], where mass1 = [x, y, z] where x, y, z the trajectories
        :return: list of x, y, z trajectories of the equivalent CoM
        '''

        masses_number = len(masses)                 # number of masses
        trj_points = len(xyz_trajectories[0][0])    # number of trajectory points

        eq_CoM_trj = [[] for i in range(3)]         # list with 3 lists (x, y, z) of trajectory points

        for i in range(trj_points):                 # loop over trj points

            # lists of lists (for each mass) of 3 elements (x, y, z)
            CoM_contribution = [[0, 0, 0] for ii in range(masses_number)]

            for j in range(masses_number):      # loop over masses
                for k in range(3):              # loop over x, y, z coordinates

                    # save the term mass * mass_position / sum of masses
                    CoM_contribution[j][k] = masses[j] * xyz_trajectories[j][k][i] / sum(masses)

            # save the sum of the terms originating from each mass
            for j in range(3):                  # loop over cartesian coordinates
                eq_CoM_trj[j].append(sum([sublist[j] for sublist in CoM_contribution]))

        return eq_CoM_trj

    def print_trj(self, solution, results, resol, contacts, swing_id, t_exec=[0, 0, 0, 0]):
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
                # plt.plot([i * self._dt for i in range(self._N)], solution['x'][self._dimc * i + j], '.')
            plt.grid()
            plt.legend(['x', 'y', 'z'])
            plt.title(name)
        plt.xlabel('Time [s]')
        # plt.savefig('../plots/gait_state_trj.png')

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

        # Interpolated moving contact trajectory
        mov_contact_labels = ['p_mov_l', 'dp_mov_l', 'ddp_mov_l', 'p_mov_r', 'dp_mov_r', 'ddp_mov_r']
        plt.figure()
        for i, name in enumerate(mov_contact_labels):
            plt.subplot(2, 3, i + 1)
            for k in range(3):
                plt.plot(results['t'], results[name][k], '-')
                plt.grid()
                plt.legend(['x', 'y', 'z'])
            plt.ylabel(name)
            plt.suptitle('Moving Contact trajectory')
        plt.xlabel('Time [s]')

        # plot swing trajectory
        # All points to be published
        N_total = int(self._Nseg * self._dt * resol)  # total points --> total time * frequency
        s = np.linspace(0, self._dt * self._Nseg, N_total)
        coord_labels = ['x', 'y', 'z']
        for j in range(len(results['sw'])):
            plt.figure()
            for i, name in enumerate(coord_labels):
                plt.subplot(3, 1, i + 1)
                plt.plot(results['sw'][j][name])  # nominal trj
                # plt.plot(s[0:t_exec[j]], results['sw'][j][name][0:t_exec[j]])  # executed trj
                plt.grid()
                plt.legend(['nominal', 'real'])
                plt.title('Trajectory ' + name)
            plt.xlabel('Time [s]')
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

        # compute total support polygon including payload masses
        # Construct lists with the trajectory position from the solution of the TO problem
        CoM_optimal_position = []
        payload1_optimal_position = []
        payload2_optimal_position = []

        for i in range(3):      # loop over cartesian coordinates
            CoM_optimal_position.append(solution['x'][i::self._dimx])
            payload1_optimal_position.append(solution['Pl_mov'][i::self._dimp_mov])
            payload2_optimal_position.append(solution['Pr_mov'][i::self._dimp_mov])

        # call the function that computes the equivalent CoM
        total_CoM = self.compute_equivalent_CoM([self._mass, self._payload_mass_l, self._payload_mass_r],
                                                [CoM_optimal_position, payload1_optimal_position, payload2_optimal_position])

        # Support polygon and CoM motion in the plane
        color_labels = ['red', 'green', 'blue', 'yellow']
        line_labels = ['-', '--', '-.', ':']
        plt.figure()
        for i in range(len(swing_id)):
            SuP_x_coords = [contacts[k][1] for k in range(4) if k not in [swing_id[i]]]
            SuP_x_coords.append(SuP_x_coords[0])
            SuP_y_coords = [contacts[k][0] for k in range(4) if k not in [swing_id[i]]]
            SuP_y_coords.append(SuP_y_coords[0])
            plt.plot(SuP_x_coords, SuP_y_coords, line_labels[0], linewidth=2-0.4*i, color=color_labels[i])
        plt.plot(results['x'][1], results['x'][0], '--', linewidth=3)       # robot links - based CoM
        plt.plot(total_CoM[1], total_CoM[0])        # equivalent CoM
        plt.grid()
        plt.title('Support polygon and CoM')
        plt.xlabel('Y [m]')
        plt.ylabel('X [m]')
        plt.xlim(0.5, -0.5)
        plt.show()


class GaitNonlinear(Gait):
    """
    This is a trial for optimization of both virtual moving contact position and virtual force.
    Extra newton constraint is added for the dynamics of the mass of the payload.

    """

    def __init__(self, mass, N, dt, payload_masses, slope_deg, conservative_box=True):
        """Walking class constructor

        Args:
            mass (float): robot mass
            N (int): horizon length
            dt (float): discretization step
            payload_masses: masses attached to arms [left, right]
            slope_deg: inclination of terrain in degrees (only in x direction)
            conservative_box: if true the arm EE box constraints are more conservative and far from robot chest
        """

        self._box_conservative = conservative_box

        self._gravity = parameters.get_gravity_acc_vector(slope_deg)

        self._Nseg = N
        self._dt = dt  # dt used for optimization knots
        self._problem_duration = N * dt
        self._mass = mass

        self._knot_number = knot_number = N + 1  # number of knots is one more than segments number
        self._tjunctions = [(i * dt) for i in range(knot_number)]  # time junctions from first to last

        # mass of the payload of each arm
        self._payload_mass_l = payload_masses[0]
        self._payload_mass_r = payload_masses[1]

        # define dimensions
        sym_t = cs.SX
        self._dimc = dimc = 3
        self._dimx = dimx = 3 * dimc
        self._dimu = dimu = dimc
        self._dimf = dimf = dimc
        self._dimp_mov = dimp_mov = 3
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

        # expression for the integrated state
        xf = cs.vertcat(
            c + dc * delta_t + 0.5 * ddc * delta_t ** 2 + 1.0 / 6.0 * u * delta_t ** 3,  # position
            dc + ddc * delta_t + 0.5 * u * delta_t ** 2,  # velocity
            ddc + u * delta_t  # acceleration
        )

        # wrap the expression into a function
        self._integrator = cs.Function('integrator', [x, u, delta_t], [xf], ['x0', 'u', 'delta_t'], ['xf'])

        # construct the optimization problem (variables, cost, constraints, bounds)
        X = sym_t.sym('X', knot_number * dimx)  # state is an SX for all knots
        U = sym_t.sym('U', knot_number * dimu)  # for all knots
        F = sym_t.sym('F', knot_number * (ncontacts * dimf))  # for all knots

        # moving contact
        P_mov_l = sym_t.sym('P_mov_l', knot_number * dimp_mov)  # position knots for the virtual contact
        P_mov_r = sym_t.sym('P_mov_r', knot_number * dimp_mov)  # position knots for the virtual contact
        DP_mov_l = sym_t.sym('DP_mov_l', knot_number * dimp_mov)  # velocity knots for the virtual contact
        DP_mov_r = sym_t.sym('DP_mov_r', knot_number * dimp_mov)  # velocity knots for the virtual contact

        # virtual force
        F_virt_l = sym_t.sym('F_virt_l', dimu * knot_number)
        F_virt_r = sym_t.sym('F_virt_r', dimu * knot_number)

        P = list()
        g = list()  # list of constraint expressions
        J = list()  # list of cost function expressions

        self._trj = {
            'x': X,
            'u': U,
            'F': F,
            'P_mov_l': P_mov_l,
            'P_mov_r': P_mov_r,
            'DP_mov_l': DP_mov_l,
            'DP_mov_r': DP_mov_r,
            'F_virt_l': F_virt_l,
            'F_virt_r': F_virt_r
        }

        # get default arm position for final penalty
        arms_default_pos = parameters.get_arm_default_pos('forward', self._box_conservative)

        # iterate over knots starting from k = 0
        for k in range(knot_number):

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
            # penalize CoM position
            cost_function += costs.penalize_horizontal_CoM_position(1e3, X[x_slice1:x_slice1 + 3], p_k)
            cost_function += costs.penalize_vertical_CoM_position(1e3, X[x_slice1:x_slice1 + 3], p_k)
            cost_function += costs.penalize_xy_forces(1e-3, F[f_slice1:f_slice2])  # penalize xy forces
            cost_function += costs.penalize_quantity(1e-0, U[u_slice1:u_slice2],
                                                     k, knot_number)  # penalize CoM jerk, that is the control

            # analytical costs for moving contact acceleration
            if k < knot_number - 1:
                weights = [1e1 for i in range(3)]
                l_acceleration_analytical_cost = costs.get_analytical_cost_3D(weights,
                                                                              P_mov_l[u_slice1:u_slice2+3],
                                                                              DP_mov_l[u_slice1:u_slice2+3],
                                                                              self._dt, k, 2)
                r_acceleration_analytical_cost = costs.get_analytical_cost_3D(weights,
                                                                              P_mov_r[u_slice1:u_slice2 + 3],
                                                                              DP_mov_r[u_slice1:u_slice2 + 3],
                                                                              self._dt, k, 2)
                cost_function += l_acceleration_analytical_cost['x'] + r_acceleration_analytical_cost['x']
                cost_function += l_acceleration_analytical_cost['y'] + r_acceleration_analytical_cost['y']
                cost_function += l_acceleration_analytical_cost['z'] + r_acceleration_analytical_cost['z']

            # simple cost just for comparisons in the paper
            # if k > 0:
            #     cost_function += costs.penalize_quantity(5*1e1, DP_mov_l[u_slice1:u_slice2] - DP_mov_l[u_slice0:u_slice1],
            #                                              k, knot_number)
            #     cost_function += costs.penalize_quantity(5*1e1, DP_mov_r[u_slice1:u_slice2] - DP_mov_r[u_slice0:u_slice1],
            #                                              k, knot_number)

            # hands penalization over whole trajectory and high final penalty
            cost_fraction = 0.0     # this determines the penalty for the trj except the final knot which is high
            cost_hands = cost_fraction * 1e3
            default_lmov_contact = P_mov_l[u_slice1:u_slice2] - X[x_slice1:x_slice1 + 3] - arms_default_pos['left']
            default_rmov_contact = P_mov_r[u_slice1:u_slice2] - X[x_slice1:x_slice1 + 3] - arms_default_pos['right']
            cost_function += costs.penalize_quantity(cost_hands, default_lmov_contact, k, knot_number, final_weight=1e3)
            cost_function += costs.penalize_quantity(cost_hands, default_rmov_contact, k, knot_number, final_weight=1e3)

            J.append(cost_function)

            # newton - euler dynamic constraints
            newton_euler_constraint = constraints.newton_euler_constraint(
                X[x_slice1:x_slice2], mass, self._gravity, ncontacts, F[f_slice1:f_slice2],
                p_k, P_mov_l[u_slice1:u_slice2], P_mov_r[u_slice1:u_slice2],
                F_virt_l[u_slice1:u_slice2], F_virt_r[u_slice1:u_slice2]
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

                # newton constraint for the payload mass, virtual force has opposite sign
                payload_mass_constraint_l = constraints.newton_point_mass_constraint(P_mov_l[u_slice0:u_slice2],
                                                                                  DP_mov_l[u_slice0:u_slice2],
                                                                                  dt,
                                                                                  k,
                                                                                  self._payload_mass_l,
                                                                                  self._gravity,
                                                                                  -F_virt_l[u_slice0:u_slice1])
                self._g_string.extend(payload_mass_constraint_l['name'])
                g.append(payload_mass_constraint_l['constraint'])

                payload_mass_constraint_r = constraints.newton_point_mass_constraint(P_mov_r[u_slice0:u_slice2],
                                                                                  DP_mov_r[u_slice0:u_slice2],
                                                                                  dt,
                                                                                  k,
                                                                                  self._payload_mass_r,
                                                                                  self._gravity,
                                                                                  -F_virt_r[u_slice0:u_slice1])
                self._g_string.extend(payload_mass_constraint_r['name'])
                g.append(payload_mass_constraint_r['constraint'])

            # moving contact spline acceleration continuity
            if 0 < k < (self._knot_number - 1):
                left_mov_contact_spline_acc_constraint = constraints.spline_acc_constraint_3D(
                    P_mov_l[u_slice0:u_slice2 + 3], DP_mov_l[u_slice0:u_slice2 + 3], dt, k
                )

                right_mov_contact_spline_acc_constraint = constraints.spline_acc_constraint_3D(
                    P_mov_r[u_slice0:u_slice2 + 3], DP_mov_r[u_slice0:u_slice2 + 3], dt, k
                )

                self._g_string.extend(left_mov_contact_spline_acc_constraint['name'])
                self._g_string.extend(right_mov_contact_spline_acc_constraint['name'])

                g.append(left_mov_contact_spline_acc_constraint['constraint'])
                g.append(right_mov_contact_spline_acc_constraint['constraint'])

            # box constraint - moving contact
            left_mov_contact_box_constraint = constraints.moving_contact_box_constraint(
                P_mov_l[u_slice1:u_slice2], X[x_slice1:x_slice1 + 3])

            right_mov_contact_box_constraint = constraints.moving_contact_box_constraint(
                P_mov_r[u_slice1:u_slice2], X[x_slice1:x_slice1 + 3])

            self._g_string.extend(left_mov_contact_box_constraint['name'])
            self._g_string.extend(right_mov_contact_box_constraint['name'])

            g.append(left_mov_contact_box_constraint['constraint'])
            g.append(right_mov_contact_box_constraint['constraint'])

            # constraint for self collision avoidance of arm ee
            avoid_arm_collision_constraint = constraints.avoid_arm_self_collision_constraint(
                P_mov_l[u_slice1:u_slice2],
                P_mov_r[u_slice1:u_slice2]
            )

            self._g_string.extend(avoid_arm_collision_constraint['name'])
            g.append(avoid_arm_collision_constraint['constraint'])

        # construct the solver
        self._nlp = {
            'x': cs.vertcat(X, U, F, P_mov_l, P_mov_r, DP_mov_l, DP_mov_r, F_virt_l, F_virt_r),
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
            'ipopt.mu_strategy': 'adaptive',
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.print_level': 3
        }

        self._solver = cs.nlpsol('solver', 'ipopt', self._nlp, solver_options)

    def solve(self, x0, contacts, mov_contact_initial, swing_id, swing_tgt, swing_clearance, swing_t, min_f=50,
              init_guess=None, state_lamult=None, constr_lamult=None, nlp_params=None):
        # todo option to add final com constraint in the offline only case
        """Solve the stepping problem

        Args:
            x0 ([type]): initial state (com position, velocity, acceleration)
            contacts ([type]): list of contact point positions
            mov_contact_initial: initial position and velocity of the moving contact
            swing_id ([type]): the indexes of the legs to be swinged from 0 to 3
            swing_tgt ([type]): the target footholds for the swing legs
            swing_clearance: clearance achieved from the highest point between initial and target position
            swing_t ([type]): list of lists with swing times in secs
            min_f: minimum threshold for forces in z direction
            init_guess: initial guess of the solution
            nlp_params: parameters to assign to the nlp problem, here contact positions
        """

        # zero initial guess if not specified
        if init_guess is None:
            init_guess = np.zeros(self._nvars)

        # grab moving contacts from the list
        lmov_contact_initial = mov_contact_initial[0]
        rmov_contact_initial = mov_contact_initial[1]

        # lists for assigning bounds
        Xl = [0] * self._dimx * self._knot_number  # state lower bounds (for all knots)
        Xu = [0] * self._dimx * self._knot_number  # state upper bounds
        Ul = [0] * self._dimu * self._knot_number  # control lower bounds
        Uu = [0] * self._dimu * self._knot_number  # control upper bounds
        Fl = [0] * self._dimf_tot * self._knot_number  # force lower bounds
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

        # virtual forces
        F_virt_l_l = [0] * self._dimu * self._knot_number  # force lower bounds
        F_virt_l_u = [0] * self._dimu * self._knot_number  # force upper bounds
        F_virt_r_l = [0] * self._dimu * self._knot_number  # force lower bounds
        F_virt_r_u = [0] * self._dimu * self._knot_number  # force upper bounds

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
                i_index = swing_id.index(i)
                final_contacts.append(swing_tgt[swing_id.index(i)])
            else:
                final_contacts.append(contacts[i])
        # print("Final contacts are", final_contacts)

        # final_state = constraints.get_nominal_CoM_bounds_from_contacts(final_contacts)

        # get bounds for arms' box constraints
        arm_bounds = parameters.get_arm_box_bounds('forward', self._box_conservative)

        # get bounds for arm self collision avoidance
        arm_distance_bounds = parameters.get_distance_between_arms(self._box_conservative)

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
            state_bounds = constraints.bound_state_variables(x0, [np.full(9, -cs.inf), np.full(9, cs.inf)],
                                                             k, self._knot_number)#, final_state)
            Xu[x_slice1:x_slice2] = state_bounds['max']
            Xl[x_slice1:x_slice2] = state_bounds['min']

            # ctrl bounds
            u_max = np.full(self._dimu, cs.inf)  # do not bound control
            u_min = -u_max

            Uu[u_slice1:u_slice2] = u_max
            Ul[u_slice1:u_slice2] = u_min

            # force bounds
            force_bounds = constraints.bound_force_variables(min_f, 1500, k, swing_t, swing_id, self._ncontacts,
                                                             self._dt, step_num)
            Fu[f_slice1:f_slice2] = force_bounds['max']
            Fl[f_slice1:f_slice2] = force_bounds['min']

            # Moving contact position bounds - only initial condition
            left_mov_contact_bounds = constraints.bound_moving_contact_variables(
                lmov_contact_initial[0],
                lmov_contact_initial[1],
                [np.full(3, -cs.inf), np.full(3, cs.inf)],
                [np.full(3, -cs.inf), np.full(3, cs.inf)],
                k, self._knot_number,
                dp_mov_final_bound=[np.full(3, -cs.inf), np.full(3, cs.inf)])
            Pl_movu[u_slice1:u_slice2] = left_mov_contact_bounds['p_mov_max']
            Pl_movl[u_slice1:u_slice2] = left_mov_contact_bounds['p_mov_min']
            DPl_movu[u_slice1:u_slice2] = left_mov_contact_bounds['dp_mov_max']
            DPl_movl[u_slice1:u_slice2] = left_mov_contact_bounds['dp_mov_min']

            right_mov_contact_bounds = constraints.bound_moving_contact_variables(
                rmov_contact_initial[0],
                rmov_contact_initial[1],
                [np.full(3, -cs.inf), np.full(3, cs.inf)],
                [np.full(3, -cs.inf), np.full(3, cs.inf)],
                k, self._knot_number,
                dp_mov_final_bound=[np.full(3, -cs.inf), np.full(3, cs.inf)])
            Pr_movu[u_slice1:u_slice2] = right_mov_contact_bounds['p_mov_max']
            Pr_movl[u_slice1:u_slice2] = right_mov_contact_bounds['p_mov_min']
            DPr_movu[u_slice1:u_slice2] = right_mov_contact_bounds['dp_mov_max']
            DPr_movl[u_slice1:u_slice2] = right_mov_contact_bounds['dp_mov_min']

            # virtual force bounds
            f_magn_xy = 8.0 # cs.inf
            f_magn_z = 3.0  # cs.inf
            F_virt_l_l[u_slice1:u_slice2] = self._payload_mass_l * self._gravity + [-f_magn_xy, -f_magn_xy, - f_magn_z]
            F_virt_l_u[u_slice1:u_slice2] = self._payload_mass_l * self._gravity + [f_magn_xy, f_magn_xy, f_magn_z]

            F_virt_r_l[u_slice1:u_slice2] = self._payload_mass_r * self._gravity + [-f_magn_xy, -f_magn_xy, - f_magn_z]
            F_virt_r_u[u_slice1:u_slice2] = self._payload_mass_r * self._gravity + [f_magn_xy, f_magn_xy, f_magn_z]

            # foothold positions - parameters
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

            if k > 0:       # state constraint
                gl.append(np.zeros(self._dimx))
                gu.append(np.zeros(self._dimx))

                # newton constraint for payload mass
                gl.append(np.zeros(3))
                gu.append(np.zeros(3))

                gl.append(np.zeros(3))
                gu.append(np.zeros(3))

            if 0 < k < self._knot_number - 1:
                gl.append(np.zeros(3))  # 2 moving contacts spline acc continuity
                gu.append(np.zeros(3))

                gl.append(np.zeros(3))
                gu.append(np.zeros(3))

            # box constraint - moving contact bounds
            gl.append(arm_bounds['left_l'])
            gu.append(arm_bounds['left_u'])

            gl.append(arm_bounds['right_l'])
            gu.append(arm_bounds['right_u'])

            # arm self collision avoidance
            gl.append(arm_distance_bounds['lower'])
            gu.append(arm_distance_bounds['upper'])

        # in case the nlp params are already known then use them
        if nlp_params is not None:
            P = nlp_params

        # final constraints
        Xl[-6:] = [0.0 for i in range(6)]  # zero velocity and acceleration
        Xu[-6:] = [0.0 for i in range(6)]

        DPl_movl[-3:] = [0.0 for i in range(3)]  # zero velocity of the moving contact
        DPl_movu[-3:] = [0.0 for i in range(3)]

        DPr_movl[-3:] = [0.0 for i in range(3)]  # zero velocity of the moving contact
        DPr_movu[-3:] = [0.0 for i in range(3)]

        # format bounds and params according to solver
        lbv = cs.vertcat(Xl, Ul, Fl, Pl_movl, Pr_movl, DPl_movl, DPr_movl, F_virt_l_l, F_virt_r_l)
        ubv = cs.vertcat(Xu, Uu, Fu, Pl_movu, Pr_movu, DPl_movu, DPr_movu, F_virt_l_u, F_virt_r_u)
        self._lbg = lbg = cs.vertcat(*gl)
        self._ubg = ubg = cs.vertcat(*gu)
        params = cs.vertcat(*P)

        # compute solution-call solver
        if state_lamult is not None and constr_lamult is not None:
            self._sol = sol = self._solver(x0=init_guess, lbx=lbv, ubx=ubv, lbg=lbg, ubg=ubg, p=params,
                                           lam_x0=state_lamult, lam_g0=constr_lamult)
        else:
            self._sol = sol = self._solver(x0=init_guess, lbx=lbv, ubx=ubv, lbg=lbg, ubg=ubg, p=params)

        # plot state, forces, control input, quantities to be computed by evaluate function
        x_trj = cs.horzcat(self._trj['x'])  # pack states in a desired matrix
        f_trj = cs.horzcat(self._trj['F'])  # pack forces in a desired matrix
        u_trj = cs.horzcat(self._trj['u'])  # pack control inputs in a desired matrix
        p_lmov_trj = cs.horzcat(self._trj['P_mov_l'])  # pack moving contact trj in a desired matrix
        dp_lmov_trj = cs.horzcat(self._trj['DP_mov_l'])  # pack moving contact trj in a desired matrix
        p_rmov_trj = cs.horzcat(self._trj['P_mov_r'])  # pack moving contact trj in a desired matrix
        dp_rmov_trj = cs.horzcat(self._trj['DP_mov_r'])  # pack moving contact trj in a desired matrix

        # virtual force
        f_virt_l_trj = cs.horzcat(self._trj['F_virt_l'])  # pack moving contact trj in a desired matrix
        f_virt_r_trj = cs.horzcat(self._trj['F_virt_r'])  # pack moving contact trj in a desired matrix

        # return values of the quantities *_trj
        return {
            'x': self.evaluate(sol['x'], x_trj),
            'F': self.evaluate(sol['x'], f_trj),
            'u': self.evaluate(sol['x'], u_trj),
            'Pl_mov': self.evaluate(sol['x'], p_lmov_trj),
            'Pr_mov': self.evaluate(sol['x'], p_rmov_trj),
            'DPl_mov': self.evaluate(sol['x'], dp_lmov_trj),
            'DPr_mov': self.evaluate(sol['x'], dp_rmov_trj),
            'F_virt_l': self.evaluate(sol['x'], f_virt_l_trj),  # virtual force
            'F_virt_r': self.evaluate(sol['x'], f_virt_r_trj),  # virtual force
            'lam_x': sol['lam_x'],    # lagrange multipliers
            'lam_g': sol['lam_g'],
        }

    def print_trj(self, solution, results, resol, contacts, swing_id, t_exec=[0, 0, 0, 0]):
        # plot virtual force
        plt.figure()
        for k in range(3):
            plt.plot(self._tjunctions, solution['F_virt_l'][k::3], '-')
            plt.plot(self._tjunctions, solution['F_virt_r'][k::3], '-')
        plt.grid()
        plt.title("Virtual force in both moving contacts")
        plt.legend(['x_l', 'x_r', 'y_l', 'y_r', 'z_l', 'z_r'])
        plt.xlabel('Time [s]')

        # compute the violated constraints and plot
        self.compute_constraint_violation(self._sol['x'], np.array(self._P).flatten(),
                                          self._nlp['g'], self._lbg, self._ubg)

        super(GaitNonlinear, self).print_trj(solution, results, resol, contacts, swing_id, t_exec)


if __name__ == "__main__":

    # initial state
    # c0 = np.array([0.107729, 0.0000907, -0.02118])
    c0 = np.array([0.0822, 0.0009, -0.0222])    # open elbows and no ball hands
    # c0 = np.array([-0.03, -0.04, 0.01687])
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
        np.array([0.6336, 0.27945, 0.298]),
        np.zeros(3),
    ]

    rmoving_contact = [
        np.array([0.6336, -0.27945, 0.298]),
        np.zeros(3),
    ]

    moving_contact = [lmoving_contact, rmoving_contact]

    # swing id from 0 to 3
    # sw_id = 2
    sw_id = [2,3,0]

    step_num = len(sw_id)

    # swing_target = np.array([-0.35, -0.35, -0.719])
    dx = 0.1
    dy = 0.0
    dz = 0.0

    swing_target = []
    for i in range(step_num):
        swing_target.append([foot_contacts[sw_id[i]][0] + dx, foot_contacts[sw_id[i]][1] + dy, foot_contacts[sw_id[i]][2] + dz])

    swing_target = np.array(swing_target)

    # swing_time
    # swing_time = [[1.0, 4.0], [5.0, 8.0]]
    #swing_time = [[1.0, 2.5], [3.5, 5.0], [6.0, 7.5], [8.5, 10.0]]
    swing_time = [[1.0, 3.0], [4.0, 6.0], [7.0, 9.0], [10.0, 12.0]]

    step_clear = 0.05

    w = GaitNonlinear(mass=112, N=int((swing_time[0:step_num][-1][1] + 1.0)/0.2), dt=0.2,
                      payload_masses=[10.0, 10.0], slope_deg=0, conservative_box=True)

    # sol is the directory returned by solve class function contains state, forces, control values
    sol = w.solve(x0=x_init, contacts=foot_contacts, mov_contact_initial=moving_contact, swing_id=sw_id,
                  swing_tgt=swing_target, swing_clearance=step_clear, swing_t=swing_time[0:step_num], min_f=100)

    print(w._nlp['p'])
    # interpolate the values, pass values and interpolation resolution
    res = 300

    swing_currents = []
    for i in range(step_num):
        swing_currents.append(foot_contacts[sw_id[i]])
    interpl = w.interpolate(sol, swing_currents, swing_target, step_clear, swing_time[0:step_num], res)

    # print the results
    w.print_trj(sol, interpl, res, foot_contacts, sw_id)

