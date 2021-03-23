import casadi as cs
import numpy as np
import math


def swing_trj_triangle(sw_curr, sw_tgt, clear, sw_t, total_t, resol):
    '''
    Interpolates current, target foot position and a intermediate point with 5th order
    polynomials.
    Args:
        sw_curr: current foot position
        sw_tgt: target foot position
        clear: clearance (max height)
        sw_t: time interval of swing phase
        total_t: total time of optimization problem
        resol: interpolation resolution (points per sec)
        clear_loc: location of the achieved maximum clearance as a fraction of the dx an dy to be covered

    Returns:
        interpolated swing trajectory
    '''

    # intermediate point, clearance counted from the highest point
    if sw_curr[2] >= sw_tgt[2]:
        max_point = np.array([0.5 * (sw_curr[0] + sw_tgt[0]), 0.5 * (sw_curr[1] + sw_tgt[1]), sw_curr[2] + clear])
    else:
        max_point = np.array([0.5 * (sw_curr[0] + sw_tgt[0]), 0.5 * (sw_curr[1] + sw_tgt[1]), sw_tgt[2] + clear])

    # list of first and last point of swing phase
    sw_points = [sw_curr] + [max_point] + [sw_tgt]

    # list of the two points of swing phase for each coordinate
    sw_x = [sw_points[0][0], sw_points[1][0], sw_points[2][0]]
    sw_y = [sw_points[0][1], sw_points[1][1], sw_points[2][1]]
    sw_z = [sw_points[0][2], sw_points[1][2], sw_points[2][2]]

    # velocities x and y for the intermediate point
    vel_x = 3 * (sw_x[2] - sw_x[0]) / (sw_t[1] - sw_t[0])
    vel_y = 3 * (sw_y[2] - sw_y[0]) / (sw_t[1] - sw_t[0])

    # conditions, initial point of swing phase
    cond1_x = [sw_x[0], 0, 0]
    cond1_y = [sw_y[0], 0, 0]
    cond1_z = [sw_z[0], 0, 0]

    # conditions, second point of swing phase
    cond2_x = [sw_x[1], vel_x, 0]
    cond2_y = [sw_y[1], vel_y, 0]
    cond2_z = [sw_z[1], 0, -0.0]

    # conditions, third point of swing phase
    cond3_x = [sw_x[2], 0, 0]
    cond3_y = [sw_y[2], 0, 0]
    cond3_z = [sw_z[2], 0, 0]

    # divide time in two
    sw_t1 = [sw_t[0], 0.5 * (sw_t[0] + sw_t[1])]
    sw_t2 = [0.5 * (sw_t[0] + sw_t[1]), sw_t[1]]

    # save polynomial coefficients in one list for each coordinate
    sw_cx1 = splines(sw_t1, cond1_x, cond2_x)  # spline 1
    sw_cy1 = splines(sw_t1, cond1_y, cond2_y)
    sw_cz1 = splines(sw_t1, cond1_z, cond2_z)

    sw_cx2 = splines(sw_t2, cond2_x, cond3_x)  # spline 2
    sw_cy2 = splines(sw_t2, cond2_y, cond3_y)
    sw_cz2 = splines(sw_t2, cond2_z, cond3_z)

    # convert to polynomial functions
    sw_px1 = np.polynomial.polynomial.Polynomial(sw_cx1)  # spline 1
    sw_py1 = np.polynomial.polynomial.Polynomial(sw_cy1)
    sw_pz1 = np.polynomial.polynomial.Polynomial(sw_cz1)

    sw_px2 = np.polynomial.polynomial.Polynomial(sw_cx2)  # spline 2
    sw_py2 = np.polynomial.polynomial.Polynomial(sw_cy2)
    sw_pz2 = np.polynomial.polynomial.Polynomial(sw_cz2)

    # construct list of interpolated points according to specified resolution
    sw_dt = sw_t[1] - sw_t[0]
    sw_interpl_t1 = np.linspace(sw_t1[0], sw_t1[1], int(0.5 * resol * sw_dt))
    sw_interpl_t2 = np.linspace(sw_t2[0], sw_t2[1], int(0.5 * resol * sw_dt))

    sw_interpl_x = np.concatenate((sw_px1(sw_interpl_t1), sw_px2(sw_interpl_t2)))
    sw_interpl_y = np.concatenate((sw_py1(sw_interpl_t1), sw_py2(sw_interpl_t2)))
    sw_interpl_z = np.concatenate((sw_pz1(sw_interpl_t1), sw_pz2(sw_interpl_t2)))

    # number of interpolation points in non swing phases
    sw_n1 = int(resol * (sw_t[0] - total_t[0]))
    sw_n2 = int(resol * (total_t[1] - sw_t[1]))

    # append points for non swing phases
    sw_interpl_x = [sw_curr[0]] * sw_n1 + [sw_interpl_x[i] for i in range(len(sw_interpl_x))] + [sw_tgt[0]] * sw_n2
    sw_interpl_y = [sw_curr[1]] * sw_n1 + [sw_interpl_y[i] for i in range(len(sw_interpl_y))] + [sw_tgt[1]] * sw_n2
    sw_interpl_z = [sw_curr[2]] * sw_n1 + [sw_interpl_z[i] for i in range(len(sw_interpl_z))] + [sw_tgt[2]] * sw_n2

    # compute arc length of swing trj
    sw_interpl_s = 0.0
    for i in range(len(sw_interpl_x) - 1):
        sw_interpl_ds = math.sqrt(
            (sw_interpl_x[i + 1] - sw_interpl_x[i]) ** 2 + (sw_interpl_y[i + 1] - sw_interpl_y[i]) ** 2 + (
                    sw_interpl_z[i + 1] - sw_interpl_z[i]) ** 2)
        sw_interpl_s += sw_interpl_ds

    return {
        'x': sw_interpl_x,
        'y': sw_interpl_y,
        'z': sw_interpl_z,
        's': sw_interpl_s
    }


def cubic_splines(dt, init_cond, fin_cond):
    """
    This function computes the polynomial of 3rd order between two points with 4 given conditions
    Args:
        dt: time interval that the polynomial applies
        init_cond: list with initial position, velocity conditions
        fin_cond: list with final position, velocity conditions

    Returns:
        a list with the coefficient of the polynomial of the spline
    """

    # symbolic polynomial coefficients
    sym_t = cs.SX
    a0 = sym_t.sym('a0', 1)
    a1 = sym_t.sym('a1', 1)
    a2 = sym_t.sym('a2', 1)
    a3 = sym_t.sym('a3', 1)

    # time
    t = sym_t.sym('t', 1)

    # initial and final conditions
    p0 = init_cond[0]
    v0 = init_cond[1]
    p1 = fin_cond[0]
    v1 = fin_cond[1]
    # print('Initial and final conditions are:', p0, v0, ac0, p1, v1, ac1)

    # the 5th order polynomial expression
    spline = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3

    # wrap the polynomial expression in a function
    p = cs.Function('p', [t, a0, a1, a2, a3], [spline], ['t', 'a0', 'a1', 'a2', 'a3'], ['spline'])

    # symbolic velocity - 1st derivative
    first_der = cs.jacobian(spline, t)
    dp = cs.Function('dp', [t, a1, a2, a3], [first_der], ['t', 'a1', 'a2', 'a3'], ['first_der'])

    # symbolic acceleration - 2nd derivative
    sec_der = cs.jacobian(first_der, t)
    ddp = cs.Function('ddp', [t, a2, a3], [sec_der], ['t', 'a2', 'a3'], ['sec_der'])

    # construct the system of equations Ax=B, with x the list of coefficients to be computed
    A = np.array([[p(dt[0], 1, 0, 0, 0), p(dt[0], 0, 1, 0, 0), p(dt[0], 0, 0, 1, 0), p(dt[0], 0, 0, 0, 1)],\
                  [0, dp(dt[0], 1, 0, 0), dp(dt[0], 0, 1, 0), dp(dt[0], 0, 0, 1)],\
                  [p(dt[1], 1, 0, 0, 0), p(dt[1], 0, 1, 0, 0), p(dt[1], 0, 0, 1, 0), p(dt[1], 0, 0, 0, 1)],\
                  [0, dp(dt[1], 1, 0, 0), dp(dt[1], 0, 1, 0), dp(dt[1], 0, 0, 1)]])

    B = np.array([p0, v0, p1, v1])

    # coefficients
    coeffs = np.linalg.inv(A).dot(B)

    return coeffs

def splines(dt, init_cond, fin_cond):
    """
    This function computes the polynomial of 5th order between two points with 6 given conditions
    Args:
        dt: time interval that the polynomial applies
        init_cond: list with initial position, velocity and acceleration conditions
        fin_cond: list with final position, velocity and acceleration conditions

    Returns:
        a list with the coefficient of the polynomial of the spline
    """

    # symbolic polynomial coefficients
    sym_t = cs.SX
    a0 = sym_t.sym('a0', 1)
    a1 = sym_t.sym('a1', 1)
    a2 = sym_t.sym('a2', 1)
    a3 = sym_t.sym('a3', 1)
    a4 = sym_t.sym('a4', 1)
    a5 = sym_t.sym('a5', 1)

    # time
    t = sym_t.sym('t', 1)

    # initial and final conditions
    p0 = init_cond[0]
    v0 = init_cond[1]
    ac0 = init_cond[2]
    p1 = fin_cond[0]
    v1 = fin_cond[1]
    ac1 = fin_cond[2]
    # print('Initial and final conditions are:', p0, v0, ac0, p1, v1, ac1)

    # the 5th order polynomial expression
    spline = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5

    # wrap the polynomial expression in a function
    p = cs.Function('p', [t, a0, a1, a2, a3, a4, a5], [spline], ['t', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5'],
                    ['spline'])

    # symbolic velocity - 1st derivative
    first_der = cs.jacobian(spline, t)
    dp = cs.Function('dp', [t, a1, a2, a3, a4, a5], [first_der], ['t', 'a1', 'a2', 'a3', 'a4', 'a5'], ['first_der'])

    # symbolic acceleration - 2nd derivative
    sec_der = cs.jacobian(first_der, t)
    ddp = cs.Function('ddp', [t, a2, a3, a4, a5], [sec_der], ['t', 'a2', 'a3', 'a4', 'a5'], ['sec_der'])

    # construct the system of equations Ax=B, with x the list of coefficients to be computed
    A = np.array([[p(dt[0], 1, 0, 0, 0, 0, 0), p(dt[0], 0, 1, 0, 0, 0, 0), p(dt[0], 0, 0, 1, 0, 0, 0),
                   p(dt[0], 0, 0, 0, 1, 0, 0), p(dt[0], 0, 0, 0, 0, 1, 0), p(dt[0], 0, 0, 0, 0, 0, 1)], \
                  [0, dp(dt[0], 1, 0, 0, 0, 0), dp(dt[0], 0, 1, 0, 0, 0), dp(dt[0], 0, 0, 1, 0, 0),
                   dp(dt[0], 0, 0, 0, 1, 0), dp(dt[0], 0, 0, 0, 0, 1)], \
                  [0, 0, ddp(dt[0], 1, 0, 0, 0), ddp(dt[0], 0, 1, 0, 0), ddp(dt[0], 0, 0, 1, 0),
                   ddp(dt[0], 0, 0, 0, 1)], \
                  [p(dt[1], 1, 0, 0, 0, 0, 0), p(dt[1], 0, 1, 0, 0, 0, 0), p(dt[1], 0, 0, 1, 0, 0, 0),
                   p(dt[1], 0, 0, 0, 1, 0, 0), p(dt[1], 0, 0, 0, 0, 1, 0), p(dt[1], 0, 0, 0, 0, 0, 1)], \
                  [0, dp(dt[1], 1, 0, 0, 0, 0), dp(dt[1], 0, 1, 0, 0, 0), dp(dt[1], 0, 0, 1, 0, 0),
                   dp(dt[1], 0, 0, 0, 1, 0), dp(dt[1], 0, 0, 0, 0, 1)], \
                  [0, 0, ddp(dt[1], 1, 0, 0, 0), ddp(dt[1], 0, 1, 0, 0), ddp(dt[1], 0, 0, 1, 0),
                   ddp(dt[1], 0, 0, 0, 1)]])

    B = np.array([p0, v0, ac0, p1, v1, ac1])

    # coefficients
    coeffs = np.linalg.inv(A).dot(B)

    return coeffs