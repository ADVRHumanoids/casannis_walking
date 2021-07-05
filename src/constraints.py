import numpy as np
import casadi as cs

gravity = np.array([0, 0, -9.81])


def newton_euler_constraint(CoM_state, mass, contacts_num, forces, contact_positions,
                            moving_contact=cs.SX.zeros(3), virtual_force=[0.0, 0.0, 0.0]):
    # newton
    CoM_acc = CoM_state[6:9]
    newton_violation = mass * CoM_acc - mass * gravity

    for i in range(contacts_num):
        f_i_k = forces[3 * i:3 * (i + 1)]  # force of i-th contact
        newton_violation -= f_i_k
    newton_violation -= virtual_force

    # euler
    CoM_pos = CoM_state[0:3]
    euler_violation = 0.0

    for i in range(contacts_num):
        f_i_k = forces[3 * i:3 * (i + 1)]  # force of i-th contact
        p_i_k = contact_positions[3 * i:3 * (i + 1)]  # contact of i-th contact

        euler_violation += cs.cross(f_i_k, CoM_pos - p_i_k)
    euler_violation += cs.cross(virtual_force, CoM_pos - moving_contact)    # payload addition

    return {
        'newton': newton_violation,
        'euler': euler_violation
    }


def state_constraint(state_function, current_state):

    # state constraint (triple integrator)
    state_constraint_violation = state_function - current_state

    return state_constraint_violation


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

    return acc_continuity_violation


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

        acc_continuity_violation.append(spline_acc_constraint(current_polynomial, next_polynomial, t_current))

    return {
        'x': acc_continuity_violation[0],
        'y': acc_continuity_violation[1],
        'z': acc_continuity_violation[2]
    }


def bound_force_variables(min_fz, max_f, knot, swing_time_integral, swing_id, ncontacts, dt):

    # force bounds
    f_max = np.array([max_f, max_f, max_f] * ncontacts)
    f_min = np.array([-max_f, -max_f, min_fz] * ncontacts)  # bound only the z component

    # swing phase
    is_swing = swing_time_integral[0] / dt <= knot <= swing_time_integral[1] / dt
    if is_swing:
        # we are in swing phase
        f_max[3 * swing_id:3 * (swing_id + 1)] = np.zeros(3)  # overwrite forces for the swing leg
        f_min[3 * swing_id:3 * (swing_id + 1)] = np.zeros(3)

    return {
        'min': f_min,
        'max': f_max
    }


def bound_moving_contact_variables(p_mov_bound, dp_mov_bound, knot):

    p_mov_initial = np.array([0.53, 0.0, 0.3])
    dp_mov_initial = np.zeros(3)

    if knot == 0:
        p_mov_max = p_mov_initial
        p_mov_min = p_mov_initial

        dp_mov_max = dp_mov_initial
        dp_mov_min = dp_mov_initial
    else:
        p_mov_max = p_mov_bound[1]
        p_mov_min = p_mov_bound[0]

        dp_mov_max = dp_mov_bound[1]
        dp_mov_min = dp_mov_bound[0]

    return {
        'p_mov_min': p_mov_min,
        'p_mov_max': p_mov_max,
        'dp_mov_min': dp_mov_min,
        'dp_mov_max': dp_mov_max
    }


def bound_state_variables(initial_state, state_bound, knot):

    # state bounds
    if knot == 0:
        x_max = initial_state
        x_min = initial_state

    else:
        x_max = state_bound[1]
        x_min = state_bound[0]

    return {
        'min': x_min,
        'max': x_max
    }


def set_contact_parameters(contacts, swing_t, swing_id, swing_target, pos_at_max_clearance, knot, dt):

    # time that maximum clearance occurs
    clearance_time = 0.5 * (swing_t[0] + swing_t[1])  # not accurate

    # contact positions
    p_k = np.hstack(contacts)  # start with initial contacts (4x3), but not the moving one

    # time region around max clearance time
    clearance_region = (clearance_time / dt - 4 <= knot <= clearance_time / dt + 4)

    if clearance_region:
        p_k[3 * swing_id:3 * (swing_id + 1)] = pos_at_max_clearance

    elif knot > clearance_time / dt + 4:
        # after the swing, the swing foot is now at swing_tgt
        p_k[3 * swing_id:3 * (swing_id + 1)] = swing_target

    return p_k