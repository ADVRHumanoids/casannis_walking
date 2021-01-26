import numpy as np
from matplotlib import pyplot as plt
import casadi as cs

def swing_leg(pos_curr, pos_tgt, swing_t):
    '''

    Args:
        pos_curr: current position of the foot
        pos_tgt: target position of the foot
        dz: height difference between the target position and the highest point of the trajectory
        swing_t: (start, stop) period of swinging in a global manner wrt to optimization problem

    Returns:
        a dictionary with a list of polynomials (5th order splines) for each coordinate

    '''

    # list of first and last point of swing phase
    points = [pos_curr] + [pos_tgt]

    # list of the two points for each coordinate
    x = [points[0][0], points[1][0]]
    y = [points[0][1], points[1][1]]
    z = [points[0][2], points[1][1]]

    # conditions, initial point of swing phase
    init_x = [x[0], 0, 0]
    init_y = [y[0], 0, 0]
    init_z = [z[0], 0, 0]

    # conditions, final point of swing phase
    fin_x = [x[1], 0, 0]
    fin_y = [y[1], 0, 0]
    fin_z = [z[1], 0, 0]

    # save polynomial coefficients in one list for each coordinate
    # inverse the elements as [0, 1, 1] is translated into 1 + x from polynomial function
    coeff_x = splines(swing_t, init_x, fin_x)[::-1]
    coeff_y = splines(swing_t, init_y, fin_y)[::-1]
    coeff_z = splines(swing_t, init_z, fin_z)[::-1]

    # convert to polynomial functions
    poly_x = np.poly1d(coeff_x)
    poly_y = np.poly1d(coeff_y)
    poly_z = np.poly1d(coeff_z)

    return {
        'x': poly_x,
        'y': poly_y,
        'z': poly_z
    }


# function for symbolic solution of a parameterization with 5th order splines
def splines(dt, init_cond, fin_cond):
    '''
    This function computes the polynomial of 5th order between two points with 6 given conditions
    Args:
        dt: time interval that the polynomial applies
        init_cond: list with initial position, velocity and acceleration conditions
        fin_cond: list with final position, velocity and acceleration conditions

    Returns:
        a list with the coefficient of the polynomial of the spline
    '''

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
    print('Initial and final conditions are:', p0, v0, ac0, p1, v1, ac1)

    # the 5th order polynomial expression
    spline = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5

    # wrap the polynomial expression in a function
    p = cs.Function('p', [t, a0, a1, a2, a3, a4, a5], [spline], ['t', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5'], ['spline'])

    # symbolic velocity - 1st derivative
    first_der = cs.jacobian(spline, t)
    dp = cs.Function('dp', [t, a1, a2, a3, a4, a5], [first_der], ['t', 'a1', 'a2', 'a3', 'a4', 'a5'], ['first_der'])

    # symbolic acceleration - 2nd derivative
    sec_der = cs.jacobian(first_der, t)
    ddp = cs.Function('ddp', [t, a2, a3, a4, a5], [sec_der], ['t', 'a2', 'a3', 'a4', 'a5'], ['sec_der'])

    # construct the system of equations Ax=B, with x the list of coefficients to be computed
    A = np.array([[p(dt[0], 1, 0, 0, 0, 0, 0), p(dt[0], 0, 1, 0, 0, 0, 0), p(dt[0], 0, 0, 1, 0, 0, 0), p(dt[0], 0, 0, 0, 1, 0, 0), p(dt[0], 0, 0, 0, 0, 1, 0), p(dt[0], 0, 0, 0, 0, 0, 1)],\
                  [0, dp(dt[0], 1, 0, 0, 0, 0), dp(dt[0], 0, 1, 0, 0, 0), dp(dt[0], 0, 0, 1, 0, 0), dp(dt[0], 0, 0, 0, 1, 0), dp(dt[0], 0, 0, 0, 0, 1)],\
        [0, 0, ddp(dt[0], 1, 0, 0, 0), ddp(dt[0], 0, 1, 0, 0), ddp(dt[0], 0, 0, 1, 0), ddp(dt[0], 0, 0, 0, 1)],\
        [p(dt[1], 1, 0, 0, 0, 0, 0), p(dt[1], 0, 1, 0, 0, 0, 0), p(dt[1], 0, 0, 1, 0, 0, 0), p(dt[1], 0, 0, 0, 1, 0, 0), p(dt[1], 0, 0, 0, 0, 1, 0), p(dt[1], 0, 0, 0, 0, 0, 1)],\
        [0, dp(dt[1], 1, 0, 0, 0, 0), dp(dt[1], 0, 1, 0, 0, 0), dp(dt[1], 0, 0, 1, 0, 0), dp(dt[1], 0, 0, 0, 1, 0), dp(dt[1], 0, 0, 0, 0, 1)],\
        [0, 0, ddp(dt[1], 1, 0, 0, 0), ddp(dt[1], 0, 1, 0, 0), ddp(dt[1], 0, 0, 1, 0), ddp(dt[1], 0, 0, 0, 1)]])

    B = np.array([p0, v0, ac0, p1, v1, ac1])

    # coefficients
    coeffs = np.linalg.inv(A).dot(B)
    print(coeffs)

    return coeffs


if __name__ == "__main__":

    position = [0.35, 0.35, -0.71]
    target = [0.45, 0.35, -0.61]
    height = 0.1

    period = (0.5, 2.5)

    trj = swing_leg(pos_curr=position, pos_tgt=target, swing_t=period)

    #print(trj['x'](0)) #debug

    # plot all splines in one graph
    s = np.linspace(period[0], period[1], 100)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(s, trj['x'](s))
    plt.grid()
    plt.title("Trajectory X")
    plt.subplot(3, 1, 2)
    plt.plot(s, trj['y'](s))
    plt.grid()
    plt.title("Trajectory Y")
    plt.subplot(3, 1, 3)
    plt.plot(s, trj['z'](s))
    plt.grid()
    plt.title("Trajectory Z")
    plt.xlabel("Time [s]")
    plt.show()