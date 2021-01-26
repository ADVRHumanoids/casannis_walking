import numpy as np
from matplotlib import pyplot as plt
import casadi as cs

def swing_leg(pos_curr, pos_tgt, dz, swing_t):
    '''

    Args:
        pos_curr: current position of the foot
        pos_tgt: target position of the foot
        dz: height difference between the target position and the highest point of the trajectory
        swing_t: (start, stop) period of swinging in a global manner wrt to optimization problem

    Returns:
        a dictionary with a list of polynomials (5th order splines) for each coordinate

    '''

    N = 6  # number of points in swing phase

    # d --> difference between first and last point
    # dd --> elementary difference to vary every point from the previous
    d = []
    dd = []
    for i in range(3):  # 3 dimensions
        d.append(pos_tgt[i] - pos_curr[i])
        dd.append((pos_tgt[i] - pos_curr[i])/(N-1))

    # height distance that the foot covers during the swing
    s = d[2] + 2 * dz
    ds = s / (N-1)  # elementary height distance

    # intermediate points
    mid_point = [[] for j in range(N-2)]
    for i in range(N-2):
        mid_point[i].append(pos_curr[0] + (i+1) * dd[0])
        mid_point[i].append(pos_curr[1] + (i+1) * dd[1])
        mid_point[i].append(pos_curr[2] + (i+1) * ds)

    # list of all points of swing phase
    points = [pos_curr] + mid_point + [pos_tgt]

    # elementary time difference
    dt = (swing_t[1] - swing_t[0]) / (N-1)

    # time list corresponding to point list
    time = [(swing_t[0] + i * dt) for i in range(N)] # for points in swing phase

    # list of total points for each coordinate
    x = [points[i][0] for i in range(N)]
    y = [points[i][1] for i in range(N)]
    z = [points[i][2] for i in range(N)]

    # coefficients' list for each spline
    coeff_x = []
    coeff_y = []
    coeff_z = []

    # polynomials
    poly_x = [[] for i in range(N - 1)]
    poly_y = [[] for i in range(N - 1)]
    poly_z = [[] for i in range(N - 1)]

    # compute the splines by calling splines function
    for i in range(N-1):    # loop for every spline
        dt = [time[i], time[i+1]]   # use time period for each spline

        # first point of spline conditions
        init_x = [x[i], 0.01, 0.0]
        init_y = [y[i], 0.0, 0.0]
        init_z = [z[i], 0.1, 0.0]

        # final point of spline conditions
        fin_x = [x[i + 1], 0.05, 0.0]
        fin_y = [y[i + 1], 0.0, 0.0]
        fin_z = [z[i + 1], 0.1, 0.0]

        # initial point of swing phase
        if i == 0:
            init_x = [x[i], 0, 0]
            init_y = [y[i], 0, 0]
            init_z = [z[i], 0, 0]

        # final point of swing phase
        if i == N - 2:
            fin_x = [x[i + 1], 0, 0]
            fin_y = [y[i + 1], 0, 0]
            fin_z = [z[i + 1], 0, 0]

        # save spline coefficients in one list for each coordinate
        # inverse the elements as [0, 1, 1] is translated into 1 + x from polynomial function
        coeff_x.append(splines(dt, init_x, fin_x)[::-1])
        coeff_y.append(splines(dt, init_y, fin_y)[::-1])
        coeff_z.append(splines(dt, init_z, fin_z)[::-1])

        # convert to polynomial functions
        poly_x[i] = np.poly1d(coeff_x[i])
        poly_y[i] = np.poly1d(coeff_y[i])
        poly_z[i] = np.poly1d(coeff_z[i])

    # Splines interpolation with order 5
    '''(tx, cx, kx) = intrpl.splrep(time, x, k=5)
    (ty, cy, ky) = intrpl.splrep(time, y, k=5)
    (tz, cz, kz) = intrpl.splrep(time, z, k=5)

    traj_x = intrpl.BSpline(tx, cx, kx)
    traj_y = intrpl.BSpline(ty, cy, ky)
    traj_z = intrpl.BSpline(tz, cz, kz)'''

    # Polynomial interpolation
    '''traj_x = np.polyfit(time, x, 5)
    traj_y = np.polyfit(time, y, 5)
    traj_z = np.polyfit(time, z, 5)
    return {
        'x': traj_x,
        'y': traj_y,
        'z': traj_z
    }'''

    return {
        't': time,
        'x': poly_x,
        'y': poly_y,
        'z': poly_z
    }


# function for symbolic solution of a parameterization with 5th order splines
def splines(dt, init_cond, fin_cond):
    '''
    This function computes the spline polynomial of 5th order between two points with 6 given conditions
    Args:
        dt: time interval that the spline applies
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

    trj = swing_leg(pos_curr=position, pos_tgt=target, dz=height, swing_t=period)

    print(trj['x'][0](0))

    # plot all splines in one graph
    s = np.linspace(period[0], period[1], 100)
    plt.figure()
    plt.subplot(3, 1, 1)
    for i in range(len(trj['x'])):
        plt.plot(s[20*i:20*(i+1)], trj['x'][i](s[20*i:20*(i+1)]))
    plt.grid()
    plt.title("Trajectory X")
    plt.subplot(3, 1, 2)
    for i in range(len(trj['y'])):
        plt.plot(s[20 * i:20 * (i + 1)], trj['y'][i](s[20*i:20*(i+1)]))
    plt.grid()
    plt.title("Trajectory Y")
    plt.subplot(3, 1, 3)
    for i in range(len(trj['z'])):
        plt.plot(s[20 * i:20 * (i + 1)], trj['z'][i](s[20*i:20*(i+1)]))
    plt.grid()
    plt.title("Trajectory Z")
    plt.xlabel("Time [s]")
    plt.show()