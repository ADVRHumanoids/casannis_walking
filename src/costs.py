import casadi as cs
import numpy as np


def penalize_horizontal_CoM_position(weight, CoM_position, contact_positions, reference_position=None):

    if reference_position is None:

        contacts_horizontal_mean_position = 0.25 * (contact_positions[0:2] +
                                                    contact_positions[3:5] +
                                                    contact_positions[6:8] +
                                                    contact_positions[9:11]
                                                    )
        reference_position = contacts_horizontal_mean_position + np.array([0.05, 0.0])

    horizontal_dist = CoM_position[0:2] - reference_position

    return weight * cs.sumsqr(horizontal_dist)


def penalize_vertical_CoM_position(weight, CoM_position, contact_positions, reference_position=None):

    if reference_position is None:
        contacts_vertical_mean_position = 0.25 * (contact_positions[2] + contact_positions[5]
                                                  + contact_positions[8] + contact_positions[11])\
                                          + 0.66
        reference_position = contacts_vertical_mean_position

    vertical_dist = CoM_position[2] - reference_position

    return weight * cs.sumsqr(vertical_dist)


def penalize_xy_forces(weight, forces):

    cost_term = weight * cs.sumsqr(forces[0::3]) + weight * cs.sumsqr(forces[1::3])

    return cost_term


def penalize_quantity(weight, quantity, knot, knot_num, final_weight=None):

    if final_weight is not None and knot == knot_num - 1:
        cost = final_weight * cs.sumsqr(quantity)
    else:
        cost = weight * cs.sumsqr(quantity)

    return cost