import casadi as cs
import numpy as np

import costs
import constraints
from gait_with_payload import Gait as ParentGait
from gait_with_payload import GaitNonlinear as GaitNonlinearForward

global_gravity = np.array([0.0, 0.0, -9.81])


class Gait(ParentGait):

    def __init__(self, mass, N, dt, payload_masses):
        """Gait class constructor

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
        f_pay_l = np.array([0, 0, self._payload_mass_l * global_gravity[2]])  # virtual force
        f_pay_r = np.array([0, 0, self._payload_mass_r * global_gravity[2]])  # virtual force

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
        arms_default_pos = constraints.get_arm_default_pos('backward')

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
                X[x_slice1:x_slice2], mass, ncontacts, F[f_slice1:f_slice2],
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
            'ipopt.linear_solver': 'ma57',
            'ipopt.print_level': 5
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

        # get box bounds for arms
        arm_bounds = constraints.get_arm_box_bounds('backward')

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


class GaitNonlinearBackward(GaitNonlinearForward):

    def __init__(self, mass, N, dt, payload_masses):
        """Gait class constructor

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
        arms_default_pos = constraints.get_arm_default_pos('backward')

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
                X[x_slice1:x_slice2], mass, ncontacts, F[f_slice1:f_slice2],
                p_k, P_mov_l[u_slice1:u_slice2], P_mov_r[u_slice1:u_slice2],
                F_virt_l[u_slice1:u_slice2], F_virt_r[u_slice1:u_slice2]
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

                # newton constraint for the payload mass, virtual force has opposite sign
                payload_mass_constraint_l = constraints.newton_payload_constraint(P_mov_l[u_slice0:u_slice2],
                                                                                  DP_mov_l[u_slice0:u_slice2],
                                                                                  dt,
                                                                                  k,
                                                                                  self._payload_mass_l,
                                                                                  -F_virt_l[u_slice0:u_slice1])
                g.append(payload_mass_constraint_l['x'])
                g.append(payload_mass_constraint_l['y'])
                g.append(payload_mass_constraint_l['z'])

                payload_mass_constraint_r = constraints.newton_payload_constraint(P_mov_r[u_slice0:u_slice2],
                                                                                  DP_mov_r[u_slice0:u_slice2],
                                                                                  dt,
                                                                                  k,
                                                                                  self._payload_mass_r,
                                                                                  -F_virt_r[u_slice0:u_slice1])
                g.append(payload_mass_constraint_r['x'])
                g.append(payload_mass_constraint_r['y'])
                g.append(payload_mass_constraint_r['z'])

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
            'ipopt.print_level': 5
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

        # virtual force
        F_virt_l_l = [0] * self._dimu * self._knot_number  # force lower bounds
        F_virt_l_u = [0] * self._dimu * self._knot_number  # force upper bounds
        F_virt_r_l = [0] * self._dimu * self._knot_number  # force lower bounds
        F_virt_r_u = [0] * self._dimu * self._knot_number  # force upper bounds

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

        # get box bounds for arms
        arm_bounds = constraints.get_arm_box_bounds('backward')

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

            # virtual force bounds
            F_virt_l_u[u_slice1:u_slice2] = np.array([10.0, 10.0, global_gravity[2] * self._payload_mass_l + 3])
            F_virt_l_l[u_slice1:u_slice2] = np.array([-10.0, -10.0, global_gravity[2] * self._payload_mass_l - 3])

            F_virt_r_u[u_slice1:u_slice2] = np.array([10.0, 10.0, global_gravity[2] * self._payload_mass_r + 3])
            F_virt_r_l[u_slice1:u_slice2] = np.array([-10.0, -10.0, global_gravity[2] * self._payload_mass_r - 3])

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

                # newton constraint for 2 payload masses
                gl.append(np.zeros(6))
                gu.append(np.zeros(6))

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
        lbv = cs.vertcat(Xl, Ul, Fl, Pl_movl, Pr_movl, DPl_movl, DPr_movl, F_virt_l_l, F_virt_r_l)
        ubv = cs.vertcat(Xu, Uu, Fu, Pl_movu, Pr_movu, DPl_movu, DPr_movu, F_virt_l_u, F_virt_r_u)
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
            'F_virt_r': self.evaluate(sol['x'], f_virt_r_trj)  # virtual force
        }


if __name__ == "__main__":

    # initial state
    c0 = np.array([0.06, 0.0, 0.002])
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
        np.array([-0.0347, 0.15, 0.417]),
        np.zeros(3),
    ]

    rmoving_contact = [
        np.array([-0.0347, -0.15, 0.417]),
        np.zeros(3),
    ]

    moving_contact = [lmoving_contact, rmoving_contact]

    # swing id from 0 to 3
    # sw_id = 2
    sw_id = [2, 3, 0, 1]

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
    #swing_time = [[1.0, 4.0], [5.0, 8.0], [9.0, 12.0], [13.0, 16.0]]
    swing_time = [[1.0, 2.5], [3.5, 5.0], [6.0, 7.5], [8.5, 9.0]]    # dynamic

    step_clear = 0.05

    w = GaitNonlinearBackward(mass=95, N=int((swing_time[-1][1] + 1.0) / 0.2), dt=0.2, payload_masses=[5.0, 5.0])

    # sol is the directory returned by solve class function contains state, forces, control values
    sol = w.solve(x0=x_init, contacts=foot_contacts, mov_contact_initial=moving_contact, swing_id=sw_id,
                  swing_tgt=swing_target, swing_clearance=step_clear, swing_t=swing_time, min_f=100)

    # interpolate the values, pass values and interpolation resolution
    res = 300

    swing_currents = []
    for i in range(step_num):
        swing_currents.append(foot_contacts[sw_id[i]])
    interpl = w.interpolate(sol, swing_currents, swing_target, step_clear, swing_time, res)

    # print the results
    w.print_trj(sol, interpl, res, foot_contacts, sw_id)


