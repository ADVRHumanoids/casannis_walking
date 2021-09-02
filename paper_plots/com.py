import casadi as cs
import numpy as np
from matplotlib import pyplot as plt
from gait import Gait as Nominal
from gait_with_payload import GaitNonlinear as Payload


def compare_print(nom_results, payl_results, contacts, swing_id, swing_periods):

    cartesian_dim = 3
    cartesian_labels = ['X', 'Y', 'Z']

    # Interpolated state plot
    state_labels = ['CoM Position', 'CoM Velocity', 'CoM Acceleration']
    plt.figure()
    for i, name in enumerate(state_labels):
        plt.subplot(3, 1, i + 1)
        # shade swing periods
        for k in range(len(swing_id)):
            plt.axvspan(swing_periods[k][0], swing_periods[k][1], alpha=0.2)

        # plot state
        for j in range(cartesian_dim):
            plt.plot(nom_results['t'], nom_results['x'][cartesian_dim * i + j], '-')
            plt.plot(payl_results['t'], payl_results['x'][cartesian_dim * i + j], '-')
        #plt.plot(2 * [k[1] for k in swing_periods], [-1,-1,-1,-1,1,1,1,1], '.')

        plt.grid()
        plt.legend(['x_nom', 'x', 'y_nom', 'y', 'z_nom', 'z'])
        plt.title(name)
    plt.xlabel('Time [s]')

    feet_labels = ['FL', 'FR', 'HL', 'HR']

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

    try:
        # Interpolated moving contact trajectory
        plt.figure()
        for k, name in enumerate(cartesian_labels):
            plt.plot(nom_results['t'], nom_results['p_mov_l'][k], '-')
            plt.plot(payl_results['t'], payl_results['p_mov_l'][k], '-')
        plt.legend([str(cartesian_labels[0]) + '_nom', str(cartesian_labels[0]),
                    str(cartesian_labels[1]) + '_nom', str(cartesian_labels[1]),
                    str(cartesian_labels[2]) + '_nom', str(cartesian_labels[2])])
        plt.grid()
        plt.title('Moving Contact trajectory')
        plt.xlabel('Time [s]')

    except:
        print("Cannot plot moving contact trajectory")

    # Support polygon and CoM motion in the plane
    SuP_x_coords = [contacts[k][1] for k in range(4) if k not in [swing_id]]
    SuP_x_coords.append(SuP_x_coords[0])
    SuP_y_coords = [contacts[k][0] for k in range(4) if k not in [swing_id]]
    SuP_y_coords.append(SuP_y_coords[0])
    plt.figure()
    plt.plot(nom_results['x'][0], nom_results['x'][1], '-')
    plt.plot(payl_results['x'][0], payl_results['x'][1], '-')
    plt.plot(SuP_y_coords, SuP_x_coords, 'ro-')
    plt.grid()
    plt.title('Support polygon and CoM')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    #plt.xlim(0.5, -0.5)
    plt.legend(['nominal', 'payload'])
    plt.show()


if __name__ == "__main__":

    # initial state
    c0 = np.array([0.107729, 0.0000907, -0.02118])
    # c0 = np.array([-0.03, -0.04, 0.01687])
    dc0 = np.zeros(3)
    ddc0 = np.zeros(3)
    x_init = np.hstack([c0, dc0, ddc0])

    foot_contacts = [
        np.array([0.35, 0.35, -0.7187]),  # fl
        np.array([0.35, -0.35, -0.7187]),  # fr
        np.array([-0.35, 0.35, -0.7187]),  # hl
        np.array([-0.35, -0.35, -0.7187])  # hr
    ]

    # mov contacts
    lmoving_contact = [
        np.array([0.53, 0.179, 0.3]),
        np.zeros(3),
    ]

    rmoving_contact = [
        np.array([0.53, -0.179, 0.3]),
        np.zeros(3),
    ]

    moving_contact = [lmoving_contact, rmoving_contact]

    # swing id from 0 to 3
    # sw_id = 2
    sw_id = [2, 3, 0, 1]

    step_num = len(sw_id)

    # swing_target = np.array([-0.35, -0.35, -0.719])
    dx = 0.2
    dy = 0.0
    dz = 0.0

    swing_target = []
    for i in range(step_num):
        swing_target.append(
            [foot_contacts[sw_id[i]][0] + dx, foot_contacts[sw_id[i]][1] + dy, foot_contacts[sw_id[i]][2] + dz])

    swing_target = np.array(swing_target)

    # swing_time
    # swing_time = [[1.0, 4.0], [5.0, 8.0]]
    swing_time = [[1.0, 2.5], [3.5, 5.0], [6.0, 7.5], [8.5, 10.0]]

    step_clear = 0.05

    w_nominal = Nominal(mass=95, N=int((swing_time[0:step_num][-1][1] + 1.0) / 0.2), dt=0.2)

    # sol is the directory returned by solve class function contains state, forces, control values
    sol1 = w_nominal.solve(x0=x_init, contacts=foot_contacts, swing_id=sw_id, swing_tgt=swing_target,
                  swing_clearance=step_clear, swing_t=swing_time, min_f=100)


    # interpolate the values, pass values and interpolation resolution
    res = 300

    swing_currents = []
    for i in range(step_num):
        swing_currents.append(foot_contacts[sw_id[i]])

    interpl1 = w_nominal.interpolate(sol1, swing_currents, swing_target, step_clear, swing_time, res)

    # print the results
    #w1.print_trj(sol1, interpl1, res, foot_contacts, sw_id)

    # adaptable
    w_payload = Payload(mass=95, N=int((swing_time[0:step_num][-1][1] + 1.0) / 0.2), dt=0.2, payload_masses=[5.0, 5.0])

    # sol is the directory returned by solve class function contains state, forces, control values
    sol2 = w_payload.solve(x0=x_init, contacts=foot_contacts, mov_contact_initial=moving_contact,
                    swing_id=sw_id, swing_tgt=swing_target, swing_clearance=step_clear,
                    swing_t=swing_time, min_f=100)


    # interpolate the values, pass values and interpolation resolution
    interpl2 = w_payload.interpolate(sol2, swing_currents, swing_target, step_clear, swing_time, res)

    # print the results
    #w1.print_trj(sol1, interpl1, res, foot_contacts, sw_id)

    compare_print(interpl1, interpl2, foot_contacts, sw_id, swing_time[0:step_num])
