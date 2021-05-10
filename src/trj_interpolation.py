import casadi as cs
import numpy as np
import math
import time
import spline_optimization as splinopt
from scipy.stats import norm
from operator import add


def swing_trj_triangle(sw_curr, sw_tgt, clear, sw_t, total_t, resol, spline_order=5):
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

    Returns:
        interpolated swing trajectory
    '''

    # intermediate point, clearance counted from the highest point
    if sw_curr[2] >= sw_tgt[2]:
        max_point = np.array([0.5 * (sw_curr[0] + sw_tgt[0]), 0.5 * (sw_curr[1] + sw_tgt[1]), sw_curr[2] + clear])
    else:
        max_point = np.array([0.5 * (sw_curr[0] + sw_tgt[0]), 0.5 * (sw_curr[1] + sw_tgt[1]), sw_tgt[2] + clear])

    # list of three points swing phase
    sw_points = [sw_curr] + [max_point] + [sw_tgt]

    # list of the three points of swing phase for each coordinate
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

    # fractions of distances between the three points to assign corresponding time
    dist1 = math.sqrt((sw_x[1] - sw_x[0]) ** 2 + (sw_y[1] - sw_y[0]) ** 2 + (sw_z[1] - sw_z[0]) ** 2)
    dist2 = math.sqrt((sw_x[2] - sw_x[1]) ** 2 + (sw_y[2] - sw_y[1]) ** 2 + (sw_z[2] - sw_z[1]) ** 2)

    try:    # deal with division with zero
        dist_fraction1 = dist1 / (dist1 + dist2)
        dist_fraction2 = dist2 / (dist1 + dist2)
    except:
        dist_fraction1 = 0.5
        dist_fraction2 = 0.5

    # assign spline time according to distances
    t_middle = sw_t[0] + dist_fraction1 * (sw_t[1] - sw_t[0])
    sw_t1 = [sw_t[0], t_middle]
    sw_t2 = [t_middle, sw_t[1]]

    # save polynomial coefficients in one list for each coordinate
    if spline_order == 3:   # cubic polynomials
        sw_cx1 = cubic_splines(sw_t1, cond1_x[0:2], cond2_x[0:2])  # spline 1
        sw_cy1 = cubic_splines(sw_t1, cond1_y[0:2], cond2_y[0:2])
        sw_cz1 = cubic_splines(sw_t1, cond1_z[0:2], cond2_z[0:2])

        sw_cx2 = cubic_splines(sw_t2, cond2_x[0:2], cond3_x[0:2])  # spline 2
        sw_cy2 = cubic_splines(sw_t2, cond2_y[0:2], cond3_y[0:2])
        sw_cz2 = cubic_splines(sw_t2, cond2_z[0:2], cond3_z[0:2])

    else:   # 5th order default
        sw_cx1 = quintic_splines(sw_t1, cond1_x, cond2_x)  # spline 1
        sw_cy1 = quintic_splines(sw_t1, cond1_y, cond2_y)
        sw_cz1 = quintic_splines(sw_t1, cond1_z, cond2_z)

        sw_cx2 = quintic_splines(sw_t2, cond2_x, cond3_x)  # spline 2
        sw_cy2 = quintic_splines(sw_t2, cond2_y, cond3_y)
        sw_cz2 = quintic_splines(sw_t2, cond2_z, cond3_z)

    # convert to polynomial functions
    sw_px1 = np.polynomial.polynomial.Polynomial(sw_cx1)  # spline 1
    sw_py1 = np.polynomial.polynomial.Polynomial(sw_cy1)
    sw_pz1 = np.polynomial.polynomial.Polynomial(sw_cz1)

    sw_px2 = np.polynomial.polynomial.Polynomial(sw_cx2)  # spline 2
    sw_py2 = np.polynomial.polynomial.Polynomial(sw_cy2)
    sw_pz2 = np.polynomial.polynomial.Polynomial(sw_cz2)

    # construct list of interpolated points according to specified resolution
    sw_dt = sw_t[1] - sw_t[0]
    sw_interpl_t1 = np.linspace(sw_t1[0], sw_t1[1], round(dist_fraction1 * resol * sw_dt))
    sw_interpl_t2 = np.linspace(sw_t2[0], sw_t2[1], round(dist_fraction2 * resol * sw_dt))

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


def swing_trj_optimal_spline(sw_curr, sw_tgt, clear, sw_t, total_t, resol, terrain_conf=0.04, z_knots=9):
    '''
    Interpolates current, target foot position and a intermediate point with 3rd order
    polynomials. Z coordinate is derived from spline optimization.
    Args:
        z_knots:
        terrain_conf:
        sw_curr: current foot position
        sw_tgt: target foot position
        clear: clearance (max height)
        sw_t: time interval of swing phase
        total_t: total time of optimization problem
        resol: interpolation resolution (points per sec)

    Returns:
        interpolated swing trajectory
    '''

    # number of interpolation points in non swing phases
    sw_n1 = int(resol * (sw_t[0] - total_t[0]))
    sw_n2 = int(resol * (total_t[1] - sw_t[1]))

    # Z coordinate - spline optimization
    ramp_points = 3  # including initial
    obstacle_points = 3

    # initial knot plan
    z_plan = splinopt.original_plan(num=z_knots, initial=sw_curr[2], target=sw_tgt[2], height_conf=terrain_conf,
                                    swing_t=sw_t, clearance=clear, ramp_num=ramp_points, obstacle_num=obstacle_points)

    start = time.time()

    # define optim. problem
    z_qp = splinopt.Spline_optimization_z(z_knots, z_plan['wayp_dt'])

    # call solver
    z_qpsol = z_qp.solver(z_plan['waypoints'], z_plan['midpoints'], ramp_points, obstacle_points)

    # get splines
    z_splines = z_qp.get_splines()

    # get trj points
    z_trj_points = z_qp.interpolate_trj(z_splines, z_plan['wayp_times'])

    # print
    #z_qp.print_results(z_plan, z_qpsol, z_splines)

    end = time.time()
    print('Time for spline optimization:', 1e3 * (end - start), 'msec')

    # complete trj for non swing phase
    sw_interpl_z = sw_n1 * [sw_curr[2]] +\
                   z_trj_points +\
                   sw_n2 * [sw_tgt[2]]

    # XY coordinates - 2 splines
    # intermediate point
    max_point = np.array([0.5 * (sw_curr[0] + sw_tgt[0]), 0.5 * (sw_curr[1] + sw_tgt[1])])

    # list of three points swing phase
    sw_points = [sw_curr[:2]] + [max_point] + [sw_tgt[:2]]

    # list of the three points of swing phase for each coordinate
    sw_x = [sw_points[0][0], sw_points[1][0], sw_points[2][0]]
    sw_y = [sw_points[0][1], sw_points[1][1], sw_points[2][1]]

    # velocities x and y for the intermediate point
    vel_x = 3 * (sw_x[2] - sw_x[0]) / (sw_t[1] - sw_t[0])
    vel_y = 3 * (sw_y[2] - sw_y[0]) / (sw_t[1] - sw_t[0])

    # conditions, initial point of swing phase
    cond1_x = [sw_x[0], 0]
    cond1_y = [sw_y[0], 0]

    # conditions, second point of swing phase
    cond2_x = [sw_x[1], vel_x]
    cond2_y = [sw_y[1], vel_y]

    # conditions, third point of swing phase
    cond3_x = [sw_x[2], 0]
    cond3_y = [sw_y[2], 0]

    # fractions of distances between the three points to assign corresponding time
    dist1 = math.sqrt((sw_x[1] - sw_x[0]) ** 2 + (sw_y[1] - sw_y[0]) ** 2)
    dist2 = math.sqrt((sw_x[2] - sw_x[1]) ** 2 + (sw_y[2] - sw_y[1]) ** 2)

    dist_fraction1 = dist1 / (dist1 + dist2)
    dist_fraction2 = dist2 / (dist1 + dist2)

    # assign spline time according to distances
    t_middle = sw_t[0] + dist_fraction1 * (sw_t[1] - sw_t[0])
    sw_t1 = [sw_t[0], t_middle]
    sw_t2 = [t_middle, sw_t[1]]

    # save polynomial coefficients in one list for each coordinate
    sw_cx1 = cubic_splines(sw_t1, cond1_x, cond2_x)  # spline 1
    sw_cy1 = cubic_splines(sw_t1, cond1_y, cond2_y)

    sw_cx2 = cubic_splines(sw_t2, cond2_x, cond3_x)  # spline 2
    sw_cy2 = cubic_splines(sw_t2, cond2_y, cond3_y)

    # convert to polynomial functions
    sw_px1 = np.polynomial.polynomial.Polynomial(sw_cx1)  # spline 1
    sw_py1 = np.polynomial.polynomial.Polynomial(sw_cy1)

    sw_px2 = np.polynomial.polynomial.Polynomial(sw_cx2)  # spline 2
    sw_py2 = np.polynomial.polynomial.Polynomial(sw_cy2)

    # construct list of interpolated points according to specified resolution
    sw_dt = sw_t[1] - sw_t[0]
    sw_interpl_t1 = np.linspace(sw_t1[0], sw_t1[1], round(dist_fraction1 * resol * sw_dt))
    sw_interpl_t2 = np.linspace(sw_t2[0], sw_t2[1], round(dist_fraction2 * resol * sw_dt))

    sw_interpl_x = np.concatenate((sw_px1(sw_interpl_t1), sw_px2(sw_interpl_t2)))
    sw_interpl_y = np.concatenate((sw_py1(sw_interpl_t1), sw_py2(sw_interpl_t2)))

    # append points for non swing phases
    sw_interpl_x = [sw_curr[0]] * sw_n1 + [sw_interpl_x[i] for i in range(len(sw_interpl_x))] + [sw_tgt[0]] * sw_n2
    sw_interpl_y = [sw_curr[1]] * sw_n1 + [sw_interpl_y[i] for i in range(len(sw_interpl_y))] + [sw_tgt[1]] * sw_n2

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


def swing_trj_gaussian(sw_curr, sw_tgt, sw_t, total_t, resol):
    '''
    Interpolates current and target foot position with a 5th order
    polynomial and superimposes a gaussian bell function to achieve clearance
    Args:
        sw_curr: current swing foot position
        sw_tgt: target swing foot position
        sw_t: swing phase time interval
        total_t: total time of optim. problem
        resol: interpolation resolution

    Returns:
        interpolated swing trajectory
    '''

    # current and target points
    sw_points = [sw_curr] + [sw_tgt]

    # list of the two points of swing phase for each coordinate
    sw_x = [sw_points[0][0], sw_points[1][0]]
    sw_y = [sw_points[0][1], sw_points[1][1]]
    sw_z = [sw_points[0][2], sw_points[1][2]]

    # conditions, initial point of swing phase
    init_x = [sw_x[0], 0, 0]
    init_y = [sw_y[0], 0, 0]
    init_z = [sw_z[0], 0, 0]

    # conditions, final point of swing phase
    fin_x = [sw_x[1], 0, 0]
    fin_y = [sw_y[1], 0, 0]
    fin_z = [sw_z[1], 0, 0]

    # save polynomial coefficients in one list for each coordinate
    sw_cx = quintic_splines(sw_t, init_x, fin_x)
    sw_cy = quintic_splines(sw_t, init_y, fin_y)
    sw_cz = quintic_splines(sw_t, init_z, fin_z)

    # convert to polynomial functions
    sw_px = np.polynomial.polynomial.Polynomial(sw_cx)
    sw_py = np.polynomial.polynomial.Polynomial(sw_cy)
    sw_pz = np.polynomial.polynomial.Polynomial(sw_cz)

    # construct list of interpolated points according to specified resolution
    sw_dt = sw_t[1] - sw_t[0]
    sw_interpl_t = np.linspace(sw_t[0], sw_t[1], int(resol * sw_dt))
    sw_interpl_x = sw_px(sw_interpl_t)
    sw_interpl_y = sw_py(sw_interpl_t)
    sw_interpl_z = sw_pz(sw_interpl_t)

    # number of interpolation points in non swing phases
    sw_n1 = int(resol * (sw_t[0] - total_t[0]))
    sw_n2 = int(resol * (total_t[1] - sw_t[1]))

    # append points for non swing phases
    sw_interpl_x = [sw_curr[0]] * sw_n1 + [sw_interpl_x[i] for i in range(len(sw_interpl_x))] + [sw_tgt[0]] * sw_n2
    sw_interpl_y = [sw_curr[1]] * sw_n1 + [sw_interpl_y[i] for i in range(len(sw_interpl_y))] + [sw_tgt[1]] * sw_n2
    sw_interpl_z = [sw_curr[2]] * sw_n1 + [sw_interpl_z[i] for i in range(len(sw_interpl_z))] + [sw_tgt[2]] * sw_n2

    # define bell function to adjust trajectory of z coordinate
    mean = 0.48 * (sw_t[0] + sw_t[1])  # center of normal distribution
    std = 0.14 * (sw_t[1] - sw_t[0])  # standard deviation
    sw_bellz = norm(loc=mean, scale=std)

    # weight of the bell function
    bell_w = 0.16   #0.08

    # list of points generated by the bell
    bell_trj = [0] * sw_n1 +\
               [bell_w * sw_bellz.pdf(sw_interpl_t[i]) for i in range(round((sw_t[1] - sw_t[0]) * resol))] +\
               [0] * sw_n2

    # sum bell with z trajectory
    sw_interpl_z = list(map(add, bell_trj, sw_interpl_z))

    # compute arc length of swing trj
    sw_interpl_s = 0.0
    for i in range(len(sw_interpl_x) - 1):
        sw_interpl_ds = math.sqrt(
            (sw_interpl_x[i + 1] - sw_interpl_x[i]) ** 2 + (sw_interpl_y[i + 1] - sw_interpl_y[i]) ** 2 +
            (sw_interpl_z[i + 1] - sw_interpl_z[i]) ** 2)
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

    # construct the system of equations Ax=B, with x the list of coefficients to be computed
    A = np.array([[p(dt[0], 1, 0, 0, 0), p(dt[0], 0, 1, 0, 0), p(dt[0], 0, 0, 1, 0), p(dt[0], 0, 0, 0, 1)],\
                  [0, dp(dt[0], 1, 0, 0), dp(dt[0], 0, 1, 0), dp(dt[0], 0, 0, 1)],\
                  [p(dt[1], 1, 0, 0, 0), p(dt[1], 0, 1, 0, 0), p(dt[1], 0, 0, 1, 0), p(dt[1], 0, 0, 0, 1)],\
                  [0, dp(dt[1], 1, 0, 0), dp(dt[1], 0, 1, 0), dp(dt[1], 0, 0, 1)]])

    B = np.array([p0, v0, p1, v1])

    # coefficients
    coeffs = np.linalg.inv(A).dot(B)

    return coeffs


def quintic_splines(dt, init_cond, fin_cond):
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