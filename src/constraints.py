import numpy as np
import casadi as cs


def newton_euler_constraint(CoM_state, mass, accel_grav, contacts_num, forces, contact_positions,
                            lmoving_contact=cs.SX.zeros(3), rmoving_contact=cs.SX.zeros(3),
                            virtual_force_l=[0.0, 0.0, 0.0], virtual_force_r=[0.0, 0.0, 0.0]
                            ):
    # newton
    CoM_acc = CoM_state[6:9]
    newton_violation = mass * CoM_acc - mass * accel_grav

    # euler
    CoM_pos = CoM_state[0:3]
    euler_violation = 0.0

    for i in range(contacts_num):
        f_i_k = forces[3 * i:3 * (i + 1)]  # force of i-th contact
        newton_violation -= f_i_k

        p_i_k = contact_positions[3 * i:3 * (i + 1)]  # contact of i-th contact
        euler_violation -= cs.cross(f_i_k, CoM_pos - p_i_k)

    # additional for arm ee contacts
    newton_violation -= virtual_force_l
    newton_violation -= virtual_force_r
    euler_violation -= cs.cross(virtual_force_l, CoM_pos - lmoving_contact)
    euler_violation -= cs.cross(virtual_force_r, CoM_pos - rmoving_contact)

    return {
        'newton': newton_violation,
        'euler': euler_violation,
        'size': 6,
        'name': ['dynamics' for i in range(6)]
    }


def SRBD_dynamics_constraint(CoM_state, base_euler_state, mass, accel_grav, contacts_num, forces, contact_positions,
                             lmoving_contact=cs.SX.zeros(3), rmoving_contact=cs.SX.zeros(3),
                             virtual_force_l=[0.0, 0.0, 0.0], virtual_force_r=[0.0, 0.0, 0.0]
                             ):
    # CoM
    CoM_acc = CoM_state[6:9]
    CoM_pos = CoM_state[0:3]

    # base euler angles and rates
    theta = base_euler_state[0:3]
    d_theta = base_euler_state[3:6]
    dd_theta = base_euler_state[6:9]

    # angular velocity in world frame
    C, C_dot = get_mapper_from_euler_to_omega(theta, d_theta)
    omega = cs.mtimes(C, d_theta)

    # derivative of angular velocity
    omega_dot = cs.mtimes(C_dot, d_theta) + cs.mtimes(C, dd_theta)

    # inertia tensor at homing position wrt frame anchored at CoM and parallel to base frame
    I_b = cs.SX([[18.7613, 0.0278677, -3.27003],
                 [0.0278677, 21.2494, -0.0110752],
                 [-3.27003, -0.0110752, 15.9858]]
                )

    # inertia tensor expressed in world frame based on current body orientation
    w_R_b = get_rotation_matrix_from_euler(theta)
    I_w = cs.mtimes(w_R_b, cs.mtimes(I_b, w_R_b.T))

    # dynamics
    newton_violation = mass * CoM_acc - mass * accel_grav
    euler_violation = cs.mtimes(I_w, omega_dot) + cs.cross(omega, cs.mtimes(I_w, omega))

    for i in range(contacts_num):
        f_i_k = forces[3 * i:3 * (i + 1)]  # force of i-th contact
        newton_violation -= f_i_k

        p_i_k = contact_positions[3 * i:3 * (i + 1)]  # contact of i-th contact
        euler_violation -= cs.cross(f_i_k, CoM_pos - p_i_k)

    # additional for arm ee contacts
    newton_violation -= virtual_force_l
    newton_violation -= virtual_force_r
    euler_violation -= cs.cross(virtual_force_l, CoM_pos - lmoving_contact)
    euler_violation -= cs.cross(virtual_force_r, CoM_pos - rmoving_contact)

    return {
        'newton': newton_violation,
        'euler': euler_violation,
        'size': 6,
        'name': ['dynamics' for i in range(6)]
    }


def get_mapper_from_euler_to_omega(theta, theta_dot):

    # roll, pitch, yaw
    x = theta[0]
    y = theta[1]
    z = theta[2]
    xdot = theta_dot[0]
    ydot = theta_dot[1]
    zdot = theta_dot[2]

    M = cs.SX.zeros(3, 3)

    M[0, 0] = cs.cos(y) * cs.cos(z)
    M[0, 1] = - cs.sin(z)
    M[1, 0] = cs.cos(y) * cs.sin(z)
    M[1, 1] = cs.cos(z)
    M[2, 0] = -cs.sin(y)
    M[2, 2] = 1.0
    # mapping_matrix = np.array(np.array([np.cos(y)*np.cos(z), - np.sin(z), 0.0]),
    #                           np.array([np.cos(y)*np.sin(z), np.cos(z),   0.0]),
    #                           np.array([- np.sin(y),         0.0,         1.0])
    #                           )

    # derivative of mapping matrix
    Mdot = cs.SX.zeros(3, 3)

    Mdot[0, 0] = -cs.cos(z)*cs.sin(y)*ydot - cs.cos(y)*cs.sin(z)*zdot
    Mdot[0, 1] = -cs.cos(z)*zdot
    Mdot[1, 0] = cs.cos(y)*cs.cos(z)*zdot - cs.sin(y)*cs.sin(z)*ydot
    Mdot[1, 1] = -cs.sin(z)*zdot
    Mdot[2, 0] = -cs.cos(y)*ydot

    return M, Mdot


def get_rotation_matrix_from_euler(theta):

    # roll, pitch, yaw
    x = theta[0]
    y = theta[1]
    z = theta[2]

    R = cs.SX.zeros(3, 3)
    R[0, 0] = cs.cos(y) * cs.cos(z)
    R[0, 1] = - cs.sin(y) * cs.sin(z)
    R[0, 2] = cs.sin(y)
    R[1, 0] = cs.cos(x) * cs.sin(z) + cs.cos(z) * cs.sin(x) * cs.sin(y)
    R[1, 1] = cs.cos(x) * cs.cos(z) - cs.sin(x) * cs.sin(y) * cs.sin(z)
    R[1, 2] = - cs.cos(y) * cs.sin(x)
    R[2, 0] = cs.sin(x) * cs.sin(z) - cs.cos(x) * cs.cos(z) * cs.sin(y)
    R[2, 1] = cs.cos(z) * cs.sin(x) + cs.cos(x) * cs.sin(y) * cs.sin(z)
    R[2, 2] = cs.cos(x) * cs.cos(y)

    return R


def get_skew_symmetric(vector):
    skew_matrix = cs.SX.zeros(3, 3)
    skew_matrix[0, 1] = - vector[2]
    skew_matrix[0, 2] = vector[1]

    skew_matrix[1, 0] = vector[2]
    skew_matrix[1, 2] = - vector[0]

    skew_matrix[2, 0] = - vector[1]
    skew_matrix[2, 1] = vector[0]

    return skew_matrix


def newton_point_mass_constraint(p_mov_list, dp_mov_list, dt, junction_index, payload_mass, accel_grav, virtual_force):
    """
    This function is useful for imposing newton's 2nd law in point masses (payload).
    m * acc = Weight + Forces
    :param accel_grav: acceleration of gravity in the planning frame
    :param p_mov_list: list of values of the cubic polynomial junctions
    :param dp_mov_list: list of values of the first derivative of the polynomial at junction
    :param dt: time discretization of knots
    :param junction_index: index of current knot
    :param payload_mass: mass of the payload
    :param virtual_force: virtual force acting on the mass(3D vector)
    :return: violation of the newton constraint
    """
    dimensions = 3

    p_mov_previous = p_mov_list[0:dimensions]
    p_mov_current = p_mov_list[dimensions:2 * dimensions]

    dp_mov_previous = dp_mov_list[0:dimensions]
    dp_mov_current = dp_mov_list[dimensions:2 * dimensions]

    p10 = []
    p11 = []
    v10 = []
    v11 = []
    T1 = []
    t10 = []
    d1 = []
    c1 = []
    acceleration_at_start = []
    for i in range(3):
        polynomial = {
            'p_list': [p_mov_previous[i], p_mov_current[i]],
            'v_list': [dp_mov_previous[i], dp_mov_current[i]],
            'T': dt,
            't0': (junction_index - 1) * dt
        }

        # junction values and times for polynomial
        p10.append(polynomial['p_list'][0])
        p11.append(polynomial['p_list'][1])
        v10.append(polynomial['v_list'][0])
        v11.append(polynomial['v_list'][1])
        T1.append(polynomial['T'])
        t10.append(polynomial['t0'])

        # c, d coefficients of polynomial 1
        d1.append((2 * p10[i] - 2 * p11[i] + T1[i] * v10[i] + T1[i] * v11[i]) / T1[i] ** 3)
        c1.append(- (3 * p10[i] - 3 * p11[i] + 2 * T1[i] * v10[i] + T1[i] * v11[i]) / T1[i] ** 2 - 3 * d1[i] * t10[i])

        # acceleration = 2.0 * c1 + 6.0 * d1 * t
        acceleration_at_start.append(2.0 * c1[i])

    newton_violation = payload_mass * np.array(acceleration_at_start) - accel_grav * payload_mass - virtual_force

    return {
        'constraint': newton_violation,
        'size': 3,
        'name': ['point_mass_dynamics' for i in range(3)]
    }


def state_constraint(state_function, current_state):

    # state constraint (triple integrator)
    state_constraint_violation = state_function - current_state

    return {
        'constraint': state_constraint_violation,
        'size': 9,
        'name': ['state_constraint' for i in range(9)]
    }


def spline_acc_constraint(poly1, poly2, t):
    '''
    This constraint is used to compute the violation of the acceleration continuity
    of a cubic spline at the polynomial junctions
    :param poly1: current polynomial
    poly1 = {
        'p_list': p_list,
        'v_list': v_list,
        'T': T,
        't0': t0
    }
    :param poly2: next polynomial
    :param t: time to impose the constraint
    :return: constraint violation
    '''

    # junction values and times for polynomial 1
    p10 = poly1['p_list'][0]
    p11 = poly1['p_list'][1]
    v10 = poly1['v_list'][0]
    v11 = poly1['v_list'][1]
    T1 = poly1['T']
    t10 = poly1['t0']

    # junction values and times for polynomial 2
    p20 = poly2['p_list'][0]
    p21 = poly2['p_list'][1]
    v20 = poly2['v_list'][0]
    v21 = poly2['v_list'][1]
    T2 = poly2['T']
    t20 = poly2['t0']

    # c, d coefficients of polynomial 1
    d1 = (2 * p10 - 2 * p11 + T1 * v10 + T1 * v11) / T1 ** 3
    c1 = - (3 * p10 - 3 * p11 + 2 * T1 * v10 + T1 * v11) / T1 ** 2 - 3 * d1 * t10

    # c, d coefficients of polynomial 2
    d2 = (2 * p20 - 2 * p21 + T2 * v20 + T2 * v21) / T2 ** 3
    c2 = - (3 * p20 - 3 * p21 + 2 * T2 * v20 + T2 * v21) / T2 ** 2 - 3 * d2 * t20

    acceleration1 = 2.0 * c1 + 6.0 * d1 * t
    acceleration2 = 2.0 * c2 + 6.0 * d2 * t

    acc_continuity_violation = acceleration1 - acceleration2

    return {
        'constraint': acc_continuity_violation,
        'size': acc_continuity_violation.size1(),
        'name': ['acceleration_continuity' for i in range(acc_continuity_violation.size1())]
    }


def spline_acc_constraint_3D(p_mov_list, dp_mov_list, dt, junction_index):

    dimensions = 3
    t_current = junction_index * dt

    p_mov_previous = p_mov_list[0:dimensions]
    p_mov_current = p_mov_list[dimensions:2 * dimensions]
    p_mov_next = p_mov_list[2 * dimensions:3 * dimensions]

    dp_mov_previous = dp_mov_list[0:dimensions]
    dp_mov_current = dp_mov_list[dimensions:2 * dimensions]
    dp_mov_next = dp_mov_list[2 * dimensions:3 * dimensions]

    acc_continuity_violation = []
    for i in range(3):
        current_polynomial = {
            'p_list': [p_mov_previous[i], p_mov_current[i]],
            'v_list': [dp_mov_previous[i], dp_mov_current[i]],
            'T': dt,
            't0': (junction_index - 1) * dt
        }

        next_polynomial = {
            'p_list': [p_mov_current[i], p_mov_next[i]],
            'v_list': [dp_mov_current[i], dp_mov_next[i]],
            'T': dt,
            't0': junction_index * dt
        }

        acc_continuity_violation.append(spline_acc_constraint(current_polynomial, next_polynomial, t_current)['constraint'])

    return {
        'constraint': np.array(acc_continuity_violation),
        'size': 3,
        'name': ['acceleration_continuity' for i in range(3)]
    }


def chp_acc_constraint(poly1, t):
    '''
    Cubic Hermite polynomial acceleration constraint
    :param poly1: current polynomial
    poly1 = {
        'p_list': p_list,
        'v_list': v_list,
        'T': T,
        't0': t0
    }
    :param t: time to impose the constraint
    :return: constraint violation
    '''

    # junction values and times for polynomial 1
    p10 = poly1['p_list'][0]
    p11 = poly1['p_list'][1]
    v10 = poly1['v_list'][0]
    v11 = poly1['v_list'][1]
    T1 = poly1['T']
    t10 = poly1['t0']

    # c, d coefficients of polynomial 1
    d1 = (2 * p10 - 2 * p11 + T1 * v10 + T1 * v11) / T1 ** 3
    c1 = - (3 * p10 - 3 * p11 + 2 * T1 * v10 + T1 * v11) / T1 ** 2 - 3 * d1 * t10

    acceleration1 = 2.0 * c1 + 6.0 * d1 * t

    acc_violation = acceleration1

    return {
        'constraint': acc_violation,
        'size': acc_violation.size1(),
        'name': ['chp_acceleration' for i in range(acc_violation.size1())]
    }


def chp_acc_constraint_3D(p_mov_list, dp_mov_list, dt, junction_index):

    dimensions = 3
    t_current = junction_index * dt

    p_mov_previous = p_mov_list[0:dimensions]
    p_mov_current = p_mov_list[dimensions:]

    dp_mov_previous = dp_mov_list[0:dimensions]
    dp_mov_current = dp_mov_list[dimensions:]

    acc_violation = []
    for i in range(3):
        current_polynomial = {
            'p_list': [p_mov_previous[i], p_mov_current[i]],
            'v_list': [dp_mov_previous[i], dp_mov_current[i]],
            'T': dt,
            't0': (junction_index - 1) * dt
        }

        acc_violation.append(chp_acc_constraint(current_polynomial, t_current)['constraint'])

    return {
        'constraint': np.array(acc_violation),
        'size': 3,
        'name': ['chp_acceleration' for i in range(3)]
    }


def bound_force_variables(min_fz, max_f, knot, swing_time_integral, swing_id, ncontacts, dt, steps_number=1):
    """
    Assigns bounds for the force decision variables. Especially for the fz, it consists the unilateral constraint.
    For the x, y compoenents the bounds are overlapped by the fricion pyramid constraints, thus, they do not really
    affect the solver.
    :param min_fz: minimum positive magnitude of the fz component
    :param max_f: maximum magnitude of the fz component
    :param knot: current knot of the problem
    :param swing_time_integral: a list of lists with time integrals (list in a list for one step)
    :param swing_id: a list with the ids (list with one element for one step)
    :param ncontacts: number of contacts
    :param dt: time segment duration based on discretization of the nlp
    :param steps_number: number of steps
    :return:
    """
    # force bounds
    f_max = np.array([max_f, max_f, max_f] * ncontacts)
    f_min = np.array([-max_f, -max_f, min_fz] * ncontacts)  # bound only the z component

    # swing phases
    is_swing = []

    for i in range(steps_number):
        is_swing.append(swing_time_integral[i][0] / dt <= knot <= swing_time_integral[i][1] / dt)

        if is_swing[i]:
            # we are in swing phase
            f_max[3 * swing_id[i]:3 * (swing_id[i] + 1)] = np.zeros(3)  # overwrite forces for the swing leg
            f_min[3 * swing_id[i]:3 * (swing_id[i] + 1)] = np.zeros(3)
            break

    return {
        'min': f_min,
        'max': f_max
    }


def friction_pyramid(force_vector, friction_coeff, ncontacts=4):
    '''
    Friction pyramid constraint
    :param force_vector: vector of forces, consists 3 * number_contacts elements
    :param friction_coeff: friction coefficient
    :param ncontacts: number of contacts
    :return: the constraint violation
    Better need to be generalized for inclined terrains etc.
    '''
    friction_violation = []

    for i in range(ncontacts):
        force_x = force_vector[3*i]
        force_y = force_vector[3*i + 1]
        force_z = force_vector[3*i + 2]

        x_violation_neg = force_x - friction_coeff * force_z    # fx < mi * fz
        x_violation_pos = force_x + friction_coeff * force_z    # fx > -mi * fz
        y_violation_neg = force_y - friction_coeff * force_z
        y_violation_pos = force_y + friction_coeff * force_z

        friction_violation += [x_violation_neg, x_violation_pos, y_violation_neg, y_violation_pos]

    return {
        'constraint': friction_violation,
        'size': len(friction_violation),
        'name': ['friction_pyramid' for i in range(len(friction_violation))]
    }


def bound_moving_contact_variables(p_mov_initial, dp_mov_initial, p_mov_bound, dp_mov_bound,
                                   knot, knot_num, dp_mov_final_bound=np.zeros(3)):

    if knot == 0:
        p_mov_max = p_mov_initial
        p_mov_min = p_mov_initial

        dp_mov_max = dp_mov_initial
        dp_mov_min = dp_mov_initial

    elif knot == knot_num - 1:
        p_mov_min = p_mov_bound[0]
        p_mov_max = p_mov_bound[1]

        dp_mov_min = dp_mov_final_bound    # final velocity (default zero)
        dp_mov_max = dp_mov_final_bound

    else:
        p_mov_min = p_mov_bound[0]
        p_mov_max = p_mov_bound[1]

        dp_mov_min = dp_mov_bound[0]
        dp_mov_max = dp_mov_bound[1]

    return {
        'p_mov_min': p_mov_min,
        'p_mov_max': p_mov_max,
        'dp_mov_min': dp_mov_min,
        'dp_mov_max': dp_mov_max
    }


def moving_contact_box_constraint(p_mov, CoM_pos):

    constraint_violation = p_mov - CoM_pos

    return {
        'constraint': constraint_violation,
        'size': 3,
        'name': ['box_constraint' for i in range(3)]
    }


def avoid_arm_self_collision_constraint(ee_left, ee_right):

    constraint_violation = ee_left[1] - ee_right[1]

    return {
        'constraint': constraint_violation,
        'size': constraint_violation.size1(),
        'name': ['arm_self_collision' for i in range(constraint_violation.size1())]
    }


def bound_state_variables(initial_state, state_bound, knot, knot_num, final_state=None):

    # state bounds
    if knot == 0:
        x_max = initial_state
        x_min = initial_state

    elif knot == knot_num-1 and final_state is not None:
        x_max = final_state[1]
        x_min = final_state[0]

    else:
        x_max = state_bound[1]
        x_min = state_bound[0]

    return {
        'min': x_min,
        'max': x_max
    }


def set_contact_parameters(contacts, swing_id, swing_target, clearance_times, pos_at_max_clearance, knot, dt, steps_number=1):
    """
    Assign the footholds to the parameter vector of the optimization problem
    :param contacts:
    :param swing_id: list of swing ids (list of one integer for a single step)
    :param swing_target: list of swing targets (arrays) for the swing feet (list of one array for a single step)
    :param clearance_times: list of maximum clearance timings (list of one float for a single step)
    :param pos_at_max_clearance: list of lists for the position of the maximum clearance point
           (list of one list for a single step)
    :param knot:
    :param dt:
    :param steps_number:
    :return:
    """
    p_k = np.hstack(contacts)  # start with initial contacts (4x3)

    # for all swing legs overwrite with target positions
    for i in range(steps_number):
        # time region around max clearance time
        clearance_region = (clearance_times[i] / dt - 4 <= knot <= clearance_times[i] / dt + 4)

        if clearance_region:
            p_k[3 * swing_id[i]:3 * (swing_id[i] + 1)] = pos_at_max_clearance[i]

        elif knot > clearance_times[i] / dt + 4:
            # after the swing, the swing foot is now at swing_tgt
            p_k[3 * swing_id[i]:3 * (swing_id[i] + 1)] = swing_target[i]

    return p_k


def get_nominal_CoM_bounds_from_contacts(contacts, offset_from_payload=0):

    mean_hor_foothold = [0.25 * coordinate
                         for coordinate in [sum([sublist[0] for sublist in contacts]),
                                            sum([sublist[1] for sublist in contacts]),
                                            sum([sublist[2] for sublist in contacts])]
                         ]

    final_position_l = [mean_hor_foothold[0]] + [mean_hor_foothold[1]] + \
                       [mean_hor_foothold[2] + 0.65 + offset_from_payload]
    final_position_u = [mean_hor_foothold[0] + 0.05] + [mean_hor_foothold[1]] + \
                       [mean_hor_foothold[2] + 0.67 + offset_from_payload]

    final_state_bounds = [np.concatenate([final_position_l, np.zeros(6)]),
                          np.concatenate([final_position_u, np.zeros(6)])]

    return final_state_bounds