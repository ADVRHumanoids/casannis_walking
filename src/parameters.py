import numpy as np
import math

def get_arm_box_bounds(side, conservative = True):
    '''
    Get the box constraint bounds for the arms' position.
    :param side: forward/backward/sideways
    :param conservative: true for conservative box bounds to avoid self collision
    :return: return the bounds as dictionary of np.array
    '''

    if side == 'forward' and conservative:

        left_lower_bound = np.array([0.45, 0.1, 0.25])
        left_upper_bound = np.array([0.58, 0.35, 0.35])

        right_lower_bound = np.array([0.45, -0.35, 0.25])
        right_upper_bound = np.array([0.58, -0.1, 0.35])

    elif side == 'forward' and not conservative:

        left_lower_bound = np.array([0.35, 0.0, 0.25])
        left_upper_bound = np.array([0.48, 0.3, 0.35])

        right_lower_bound = np.array([0.35, -0.3, 0.25])
        right_upper_bound = np.array([0.48, 0.0, 0.35])

    elif side == 'backward':
        left_lower_bound = np.array([-0.3, 0.0, 0.35])
        left_upper_bound = np.array([-0.08, 0.35, 0.45])

        right_lower_bound = np.array([-0.3, -0.35, 0.35])
        right_upper_bound = np.array([-0.08, -0.0, 0.45])

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


def get_arm_default_pos(side, conservative=True):
    '''
    Get the default position of arms.
    :param side: forward/backward/sideways
    :return: return the bounds as dictionary of lists
    '''

    if side == 'forward' and conservative:

        left = [0.55, 0.279, 0.31]
        right = [0.55, -0.279, 0.31]

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
