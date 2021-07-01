import casadi as cs
import numpy as np
from matplotlib import pyplot as plt
from walking import Walking as nominal
from step_with_payload import Walking as adaptable


def compare_print(nom_results, payl_results, contacts, swing_id):

    cartesian_dim = 3

    # Interpolated state plot
    state_labels = ['CoM Position', 'CoM Velocity', 'CoM Acceleration']
    plt.figure()
    for i, name in enumerate(state_labels):
        plt.subplot(3, 1, i + 1)
        for j in range(cartesian_dim):
            plt.plot(nom_results['t'], nom_results['x'][cartesian_dim * i + j], '-')
            plt.plot(payl_results['t'], payl_results['x'][cartesian_dim * i + j], '-')
        plt.grid()
        plt.legend(['x_nom', 'x', 'y_nom', 'y', 'z_nom', 'z'])
        plt.title(name)
    plt.xlabel('Time [s]')

    feet_labels = ['fl', 'fr', 'hl', 'hr']

    # Interpolated force plot
    plt.figure()
    for i, name in enumerate(feet_labels):
        plt.subplot(2, 2, i + 1)
        for k in range(3):
            plt.plot(nom_results['t'], nom_results['f'][3 * i + k], '-')
            plt.plot(payl_results['t'], payl_results['f'][3 * i + k], '-')
        plt.grid()
        plt.title(name)
        plt.legend([str(name) + '_x_nom', str(name) + '_x',
                    str(name) + '_y_nom', str(name) + '_y',
                    str(name) + '_z_nom', str(name) + '_z'])
    plt.xlabel('Time [s]')

    # Interpolated moving contact trajectory
    plt.figure()
    for k in range(3):
        plt.plot(payl_results['t'], payl_results['p_mov'][k], '-')
    plt.grid()
    plt.title('Moving Contact trajectory')
    plt.legend(['x', 'y', 'z'])
    plt.xlabel('Time [s]')

    # Support polygon and CoM motion in the plane
    SuP_x_coords = [contacts[k][1] for k in range(4) if k not in [swing_id]]
    SuP_x_coords.append(SuP_x_coords[0])
    SuP_y_coords = [contacts[k][0] for k in range(4) if k not in [swing_id]]
    SuP_y_coords.append(SuP_y_coords[0])
    plt.figure()
    plt.plot(nom_results['x'][1], nom_results['x'][0], '-')
    plt.plot(payl_results['x'][1], payl_results['x'][0], '-')
    plt.plot(SuP_x_coords, SuP_y_coords, 'ro-')
    plt.grid()
    plt.title('Support polygon and CoM')
    plt.xlabel('Y [m]')
    plt.ylabel('X [m]')
    plt.xlim(0.5, -0.5)
    plt.legend(['nominal', 'adaptable'])
    plt.show()


if __name__ == "__main__":

    # initial state =
    c0 = np.array([0.107729, 0.0000907, -0.02118])
    dc0 = np.zeros(3)
    ddc0 = np.zeros(3)
    x_init = np.hstack([c0, dc0, ddc0])

    foot_contacts = [
        np.array([0.35, 0.35, -0.7187]),  # fl
        np.array([0.35, -0.35, -0.7187]),  # fr
        np.array([-0.35, 0.35, -0.7187]),  # hl
        np.array([-0.35, -0.35, -0.7187])  # hr
    ]

    # swing id from 0 to 3
    sw_id = 1

    step_clear = 0.05

    # swing_target = np.array([-0.35, -0.35, -0.719])
    dx = 0.1
    dy = 0.0
    dz = -0.05
    swing_target = np.array([foot_contacts[sw_id][0] + dx, foot_contacts[sw_id][1] + dy, foot_contacts[sw_id][2] + dz])

    # swing_time = (1.5, 3.0)
    swing_time = [2.0, 5.0]

    w1 = nominal(mass=95, N=40, dt=0.2)

    # sol is the directory returned by solve class function contains state, forces, control values
    sol1 = w1.solve(x0=x_init, contacts=foot_contacts, swing_id=sw_id, swing_tgt=swing_target,
                  swing_clearance=step_clear, swing_t=swing_time, min_f=100)


    # interpolate the values, pass values and interpolation resolution
    res = 300
    interpl1 = w1.interpolate(sol1, foot_contacts[sw_id], swing_target, step_clear, swing_time, res)

    # print the results
    #w1.print_trj(sol1, interpl1, res, foot_contacts, sw_id)

    # adaptable
    w2 = adaptable(mass=95, N=40, dt=0.2)

    # sol is the directory returned by solve class function contains state, forces, control values
    sol2 = w2.solve(x0=x_init, contacts=foot_contacts, swing_id=sw_id, swing_tgt=swing_target,
                  swing_clearance=step_clear, swing_t=swing_time, min_f=100)


    # interpolate the values, pass values and interpolation resolution
    interpl2 = w2.interpolate(sol2, foot_contacts[sw_id], swing_target, step_clear, swing_time, res)

    # print the results
    #w1.print_trj(sol1, interpl1, res, foot_contacts, sw_id)

    compare_print(interpl1, interpl2, foot_contacts, sw_id)
