import numpy as np
import scipy.interpolate as intrpl
from matplotlib import pyplot as plt

def swing_leg(pos_curr, pos_tgt, dz, dt):
    '''

    Args:
        pos_curr: current position of the foot
        pos_tgt: target position of the foot
        dz: height difference between the target position and the highest point of the trajectory

    Returns:
        a dictionary with one spline for each coordinate

    '''

    # d --> target position - current position
    d = []
    for i in range(3):
        d.append(pos_tgt[i] - pos_curr[i])

    # The first intermediate point
    mid_point1 = []
    mid_point1.append(pos_curr[0] + d[0] / 3)   # current x + 1/3 * dx
    mid_point1.append(pos_curr[1] + d[1] / 3)   # current y + 1/3 * dy
    mid_point1.append(pos_curr[2] + (pos_tgt[2] + 2 * dz) / 3)  # current z + (z_target+dz)/3

    # The second intermediate point
    mid_point2 = []
    mid_point2.append(pos_curr[0] + 2 * d[0] / 3)   # current x + 2/3 * dx
    mid_point2.append(pos_curr[1] + 2 * d[1] / 3)   # current x + 2/3 * dx
    mid_point2.append(pos_curr[2] + pos_tgt[2] + (pos_tgt[2] + 2 * dz) / 3) # current z + target_z + (z_target+dz)/3

    time = [0, dt/3, 2*dt/3, dt]    # split time in 3 segments

    # interpolate separately for each coordinate
    x = [pos_curr[0], mid_point1[0], mid_point2[0], pos_tgt[0]]
    y = [pos_curr[1], mid_point1[1], mid_point2[1], pos_tgt[1]]
    z = [pos_curr[2], mid_point1[2], mid_point2[2], pos_tgt[2]]

    # use splines of order 3
    traj_x = intrpl.InterpolatedUnivariateSpline(time, x)
    traj_y = intrpl.InterpolatedUnivariateSpline(time, y)
    traj_z = intrpl.InterpolatedUnivariateSpline(time, z)

    return {
        'x': traj_x,
        'y': traj_y,
        'z': traj_z
    }


if __name__ == "__main__":
    position = [0.3, 0.3, 0.0]
    target = [0.4, 0.3, 0.1]
    height = 0.1
    time = 3

    trj = swing_leg(position, target, height, time)


    # plot splines
    s = np.linspace(0, time, 100)
    coord_labels = ['x', 'y', 'z']
    plt.figure()
    for i, name in enumerate(coord_labels):
        plt.subplot(3, 1, i+1)
        plt.plot(s, trj[name](s), '-')
        plt.grid()
        plt.title('Trajectory '+ name)
    plt.xlabel('Time [s]')
    plt.show()
