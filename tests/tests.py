import numpy as np
from matplotlib import pyplot as plt
import walking as nominal_step
import step_with_payload as adapted_step


def compare_print(nominal, adaptable, contacts, swing_id):
    '''

    :param swing_id: the id of the foot to be swinged
    :param nominal: results after interpolation of the nominal case
    :param adaptable: results after interpolation of the adaptable case
    :param contacts: initial contact points
    :return: print comparison between the two cases, especially for CoM
    '''

    cartesian_dim = 3

    # Interpolated state plot
    state_labels = ['CoM Position', 'CoM Velocity', 'CoM Acceleration']
    plt.figure()
    for i, name in enumerate(state_labels):
        plt.subplot(3, 1, i + 1)
        for j in range(cartesian_dim):
            plt.plot(nominal['t'], nominal['x'][cartesian_dim * i + j], '-')
            plt.plot(adaptable['t'], adaptable['x'][cartesian_dim * i + j], '-')
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
            plt.plot(nominal['t'], nominal['f'][3 * i + k], '-')
            plt.plot(adaptable['t'], adaptable['f'][3 * i + k], '-')
        plt.grid()
        plt.title(name)
        plt.legend([str(name) + '_x_nom', str(name) + '_x',
                    str(name) + '_y_nom', str(name) + '_x',
                    str(name) + '_z_nom', str(name) + '_x'])
    plt.xlabel('Time [s]')

    # Interpolated moving contact trajectory
    plt.figure()
    for k in range(3):
        plt.plot(adaptable['t'], adaptable['p_mov'][k], '-')
    plt.grid()
    plt.title('Moving Contact trajectory')
    plt.legend(['x', 'y', 'z'])
    plt.xlabel('Time [s]')
    # plt.savefig('../plots/mov_contact.png')

    # Support polygon and CoM motion in the plane
    SuP_x_coords = [contacts[k][1] for k in range(4) if k not in [swing_id]]
    SuP_x_coords.append(SuP_x_coords[0])
    SuP_y_coords = [contacts[k][0] for k in range(4) if k not in [swing_id]]
    SuP_y_coords.append(SuP_y_coords[0])
    plt.figure()
    plt.plot(nominal['x'][1], nominal['x'][0], '-')
    plt.plot(adaptable['x'][1], adaptable['x'][0], '-')
    plt.plot(SuP_x_coords, SuP_y_coords, 'ro-')
    plt.grid()
    plt.title('Support polygon and CoM')
    plt.xlabel('Y [m]')
    plt.ylabel('X [m]')
    plt.xlim(0.5, -0.5)
    plt.legend(['nominal', 'payload'])
    plt.show()


if __name__ == "__main__":

    # initial state
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

    # nominal solution
    w1 = nominal_step.Walking(mass=95, N=40, dt=0.2)
    sol1 = w1.solve(x0=x_init, contacts=foot_contacts, swing_id=sw_id, swing_tgt=swing_target,
                  swing_clearance=step_clear, swing_t=swing_time, min_f=100)

    res = 300   # interpolate the values, pass values and interpolation resolution
    interpl1 = w1.interpolate(sol1, foot_contacts[sw_id], swing_target, step_clear, swing_time, res)
    #w1.print_trj(sol1, interpl, res, foot_contacts, sw_id)    # print the results

    # adaptable solution for payload
    w2 = adapted_step.Walking(mass=95, N=40, dt=0.2)
    sol2 = w2.solve(x0=x_init, contacts=foot_contacts, swing_id=sw_id, swing_tgt=swing_target,
                  swing_clearance=step_clear, swing_t=swing_time, min_f=100)

    interpl2 = w2.interpolate(sol2, foot_contacts[sw_id], swing_target, step_clear, swing_time, res)
    #w2.print_trj(sol1, interpl, res, foot_contacts, sw_id)    # print the results

    # print and compare
    compare_print(interpl1, interpl2, foot_contacts, sw_id)