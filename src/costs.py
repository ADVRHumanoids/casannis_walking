import casadi as cs
import numpy as np

from cubic_hermite_polynomial import CubicPolynomial


def penalize_horizontal_CoM_position(weight, CoM_position, contact_positions, reference_position=None):
    '''
    Penalize the distance of CoM from the mean of the sum of feet positions.
    One thing to be improved here is that the reference point is affected by the position at the max_clearance_point.
    This needs to be taken care more carefully.
    :param weight:
    :param CoM_position:
    :param contact_positions:
    :param reference_position: default is zero
    :return: penalty term
    '''
    if reference_position is None:

        contacts_horizontal_mean_position = 0.25 * (contact_positions[0:2] +
                                                    contact_positions[3:5] +
                                                    contact_positions[6:8] +
                                                    contact_positions[9:11]
                                                    )
        reference_position = contacts_horizontal_mean_position + np.array([0.05, 0.0])

    horizontal_dist = CoM_position[0:2] - reference_position

    return weight * cs.sumsqr(horizontal_dist)


def penalize_vertical_CoM_position(weight, CoM_position, contact_positions, payload_offset_z=0, reference_position=None):
    '''
    :param weight: weigtht of the cost
    :param CoM_position: position vector of the CoM
    :param contact_positions: contact points
    :param payload_offset_z: extra offset to compensate for payload or other effects (optional)
    :param reference_position: hardcode a reference position for the CoM (optional)
    :return:
    '''

    if reference_position is None:
        contacts_vertical_mean_position = 0.25 * (contact_positions[2] + contact_positions[5]
                                                  + contact_positions[8] + contact_positions[11])\
                                          + 0.66 + payload_offset_z
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


def get_analytical_cost(weight, polynomial_dict, desired_derivative=0):
    '''
    This function computes an analytical cost (integral of the square) of a given cubic polynomial or its derivative
    :param polynomial_dict:
            polynomial_dict = {
                               'p_list': p_list,
                               'v_list': v_list,
                               'T': T,
                               't0': t0
                              }
    :param desired_derivative: the derivative that we want
    :return: integral value of the polynomial
    '''

    # junction values and times for polynomial 1
    p0 = polynomial_dict['p_list'][0]
    p1 = polynomial_dict['p_list'][1]
    v0 = polynomial_dict['v_list'][0]
    v1 = polynomial_dict['v_list'][1]
    T = polynomial_dict['T']
    t0 = polynomial_dict['t0']

    cubic_poly_object = CubicPolynomial([p0, p1], [v0, v1], T, t0)
    analytical_cost = weight * cubic_poly_object.integrate_squared_deriv(desired_derivative)

    return analytical_cost


def get_analytical_cost_3D(weights, p_mov_list, dp_mov_list, dt, junction_index, desired_derivative=0):

    dimensions = 3

    p_mov_previous = p_mov_list[0:dimensions]
    p_mov_current = p_mov_list[dimensions:2 * dimensions]

    dp_mov_previous = dp_mov_list[0:dimensions]
    dp_mov_current = dp_mov_list[dimensions:2 * dimensions]

    analytical_cost_3D = []
    for i in range(3):
        polynomial_elements = {
            'p_list': [p_mov_previous[i], p_mov_current[i]],
            'v_list': [dp_mov_previous[i], dp_mov_current[i]],
            'T': dt,
            't0': (junction_index - 1) * dt
        }

        analytical_cost_3D.append(get_analytical_cost(weights[i], polynomial_elements, desired_derivative))

    return {
        'x': analytical_cost_3D[0],
        'y': analytical_cost_3D[1],
        'z': analytical_cost_3D[2]
    }


if __name__ == "__main__":

    # dense polynomial
    # poly_object = CubicPolynomial([0.0, 2.0], [0.0, 0.0], 1)
    # polynomial = poly_object.get_poly_from_coeffs()

    coeff_list = [3, 5, 3]
    polynomial = np.polynomial.polynomial.Polynomial(coeff_list)

    cost = integrate_numerically(polynomial, [2.0, 5.0])
    print(cost)