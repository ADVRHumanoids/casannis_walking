import numpy as np
import math

import casadi as cs


def get_arm_box_bounds(side, conservative = True):
    '''
    Get the box constraint bounds for the arms' position.
    :param side: forward/backward/sideways
    :param conservative: true for conservative box bounds to avoid self collision
    :return: return the bounds as dictionary of np.array
    '''

    if side == 'forward' and conservative:

        left_lower_bound = np.array([0.40, -0.05, 0.32])
        left_upper_bound = np.array([0.58, 0.35, 0.42])

        right_lower_bound = np.array([0.40, -0.35, 0.32])
        right_upper_bound = np.array([0.58, 0.05, 0.42])

    elif side == 'forward' and not conservative:

        left_lower_bound = np.array([0.35, -0.1, 0.25])
        left_upper_bound = np.array([0.55, 0.35, 0.35])

        right_lower_bound = np.array([0.35, -0.35, 0.25])
        right_upper_bound = np.array([0.55, 0.1, 0.35])

    elif side == 'backward':
        left_lower_bound = np.array([-0.3, -0.1, 0.35])
        left_upper_bound = np.array([-0.08, 0.35, 0.45])

        right_lower_bound = np.array([-0.3, -0.35, 0.35])
        right_upper_bound = np.array([-0.08, 0.1, 0.45])

    elif side == 'sideways':
        # to be specified
        left_lower_bound = np.array([0.35, 0.0, 0.25])
        left_upper_bound = np.array([0.48, 0.3, 0.35])

        right_lower_bound = np.array([0.35, -0.3, 0.25])
        right_upper_bound = np.array([0.48, 0.0, 0.35])

    else:
        print('Wrong side specification')
        left_lower_bound = left_upper_bound = right_lower_bound = right_upper_bound = None

    return {
        'left_l': left_lower_bound,
        'left_u': left_upper_bound,
        'right_l': right_lower_bound,
        'right_u': right_upper_bound
    }


def get_constraint_names(formulation='gait'):
    '''
    A simple hardcoded function to return names and sizes of constraints in the form of a directory. Currently the
    formulation includes dynamics, point mass dynamics, state constraint, acceleration continuity, friction pyramid,
    box constraints, arm self collision.
    :param formulation: gait/payload
    :return: The directory of names and size of every constraint.
    '''

    leg_num = 4
    arm_num = 2

    if formulation == 'gait':

        names_sizes = {
            'dynamics': 6,
            'state_constraint': 9,
            'friction_pyramid': 4 * leg_num,
        }

    elif formulation == 'payload':

        names_sizes = {
            'dynamics': 6,
            'point_mass_dynamics': 3 * arm_num,
            'state_constraint': 9,
            'acceleration_continuity': 3 * arm_num,
            'friction_pyramid': 4 * leg_num,
            'box_constraint': 3 * arm_num,
            'arm_self_collision': 1
        }

    else:
        names_sizes = None
        print('Wrong type of formulation.')

    return names_sizes


def get_distance_between_arms(conservative = True):
    '''
    The returned distance will be the bound for the arm self collision avoidance constraint
    :param conservative: true for conservative box bounds to avoid self collision
    :return: return the bound
    '''

    if conservative:
        lower_bound = 0.3
        upper_bound = cs.inf

    elif not conservative:
        lower_bound = 0.2
        upper_bound = cs.inf
    else:
        print('Wrong side specification')
        lower_bound = upper_bound = None

    return {
        'lower': lower_bound,
        'upper': upper_bound
    }


def get_arm_default_pos(side, conservative=True):
    '''
    Get the default position of arms.
    :param side: forward/backward/sideways
    :return: return the bounds as dictionary of lists
    '''

    if side == 'forward' and conservative:

        left = [0.5, 0.2, 0.39]
        right = [0.5, -0.2, 0.39]

    elif side == 'forward' and not conservative:

        left = [0.43, 0.179, 0.3]
        right = [0.43, -0.179, 0.3]

    elif side == 'backward':
        left = [-0.0947, 0.15, 0.415]
        right = [-0.0947, -0.15, 0.415]

    elif side == 'sideways':
        # to be specified
        left = [0.43, 0.179, 0.3]
        right = [0.43, -0.179, 0.3]

    else:
        print('Wrong side specification')
        left = right = None

    return {
        'left': left,
        'right': right
    }


def get_gravity_acc_vector(inclination_deg=0):

    gravity = - 9.81

    g_x = gravity * math.sin(math.radians(inclination_deg))
    g_y = 0
    g_z = gravity * math.cos(math.radians(inclination_deg))

    gravity_acc_vector = np.array([g_x, g_y, g_z])

    return gravity_acc_vector


def get_distance_of_gravity_from_urdf_frame(link_name):

    mmTom = 0.001

    if link_name == 'ball':
        left_grav_urdf = right_grav_urdf = np.array([-1.0072498e-02*mmTom, 3.7590658e-02*mmTom, 1.9772332e+01*mmTom])

    elif link_name == 'rod_plate':
        left_grav_urdf = np.array([4.8407693e+01*mmTom, 2.0035723e+01*mmTom, 7.7533287e+01*mmTom])
        right_grav_urdf = np.array([4.8407693e+01 * mmTom, - 2.0035723e+01 * mmTom, 7.7533287e+01 * mmTom])

    else:
        left_grav_urdf = right_grav_urdf = None
        print('Wrong arm EE link name.')

    return {
        'left': left_grav_urdf,
        'right': right_grav_urdf
    }
