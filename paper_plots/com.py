import numpy as np
from matplotlib import pyplot as plt
from gait import Gait as Nominal
from gait_with_payload import GaitNonlinear as Payload

import matplotlib


def compare_print(nom_results, payl_results, contacts, swing_id, swing_periods, steps, interpol_freq):

    step_num = len(swing_id)

    cartesian_dim = 3
    cartesian_labels = ['X', 'Y', 'Z']
    linestyles = ['-', '--', ':']

    # Interpolated state plot
    state_labels = ['Position [$m$]', 'Velocity [$m/s$]', 'Acceleration [$m/s^2$]']
    plt.figure()
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)
    for i, name in enumerate(state_labels):
        axes = plt.subplot(3, 1, i + 1)
        # shade swing periods
        for k in range(step_num):
            plt.axvspan(swing_periods[k][0], swing_periods[k][1], alpha=0.2)

        # plot state
        for j, style_name in enumerate(linestyles):
            plt.plot(nom_results['t'], nom_results['x'][cartesian_dim * i + j],
                     style_name, linewidth=4, color='g', label=cartesian_labels[j])
            plt.plot(payl_results['t'], payl_results['x'][cartesian_dim * i + j],
                     style_name, linewidth=4, color='r')
            # if i == 0:
            #     plt.legend(cartesian_labels)
        plt.grid()
        plt.ylabel(name, fontsize=20)

    # handles, labels = plt.gca().get_legend_handles_labels()
    # plt.legend(handles, labels, loc='upper center')

    legend1 = plt.legend(prop={'size': 25})
    axes.add_artist(legend1)
    plt.xlabel('Time [$s$]', fontsize=20)

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
        matplotlib.rc('xtick', labelsize=16)
        matplotlib.rc('ytick', labelsize=16)
        # shade swing periods
        for k in range(step_num):
            plt.axvspan(swing_periods[k][0], swing_periods[k][1], alpha=0.2)
        for k, (name, style_name) in enumerate(zip(cartesian_labels, linestyles)):
            plt.plot(payl_results['t'], payl_results['p_mov_l'][k], style_name, linewidth=4, color='g')
            plt.plot(payl_results['t'], payl_results['p_mov_r'][k], style_name, linewidth=4, color='r')
        #plt.legend([str(cartesian_labels[0]), str(cartesian_labels[1]), str(cartesian_labels[2])])
        plt.grid()
        #plt.title('Arm end-effectors trajectory', fontsize=20)
        plt.ylabel('Position [$m$]', fontsize=20)
        plt.xlabel('Time [$s$]', fontsize=20)

    except:
        print("Cannot plot moving contact trajectory")

    # Support polygon and CoM motion in the plane
    SuP_x_coords = [contacts[k][1] for k in range(4) if k not in [swing_id]]
    SuP_x_coords.append(SuP_x_coords[0])
    SuP_y_coords = [contacts[k][0] for k in range(4) if k not in [swing_id]]
    SuP_y_coords.append(SuP_y_coords[0])

    plt.figure()
    plt.plot(nom_results['x'][0], nom_results['x'][1], '-')
    plt.plot(payl_results['x'][0], payl_results['x'][1], '--')
    plt.plot(SuP_y_coords, SuP_x_coords, 'ro-')
    plt.grid()
    plt.title('Support polygon and CoM')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    #plt.xlim(0.5, -0.5)
    plt.legend(['nominal', 'payload'])


    # polygons = [[k[0:2].tolist() for k in contacts]]
    # polygon_current = [k[0:2].tolist() for k in contacts]
    #
    # xss, yss = zip(*polygon_current)  # create lists of x and y values
    #
    # plt.figure()
    # plt.plot(xss, yss, '.')
    # plt.fill(xss, yss, alpha=0.3)
    # for i in range(step_num):
    #
    #     polygon_current[swing_id[i]] = list(map(add, polygon_current[swing_id[i]], steps[0:2]))
    #     polygons.append(polygon_current)
    #
    #
    #
    #
    # polygon_times = [0] + [k[1] for k in swing_periods]
    # polygons_x = []
    # polygons_y = []
    #
    # # Support polygon and CoM motion in the plane
    # for i in range(step_num + 1):
    #     # SuP_x_coords = [contacts[k][1] for k in range(4)]
    #     SuP_x_coords = [k['x'][int(interpol_freq * polygon_times[i])] for k in nom_results['sw']]
    #     #SuP_x_coords.append(SuP_x_coords[0])
    #     SuP_y_coords = [k['y'][int(interpol_freq * polygon_times[i])] for k in nom_results['sw']]
    #     #SuP_y_coords.append(SuP_y_coords[0])
    #
    #     polygons_x.append(SuP_x_coords)
    #     polygons_y.append(SuP_y_coords)
    #
    #     polygons = []
    #     for ii in range(4):
    #         polygons.append([SuP_x_coords[ii], SuP_y_coords[ii]])
    #     polygons.append(polygons[0])
    #
    #     xs, ys = zip(*polygons)  # create lists of x and y values
    #
    #     plt.figure()
    #     plt.plot(xs, ys, '.')
    #     plt.fill(xs, ys, alpha=0.3)
    #
    # plt.figure()
    # plt.plot(nom_results['x'][0], nom_results['x'][1], '-')
    # plt.plot(payl_results['x'][0], payl_results['x'][1], '--')
    # plt.plot(polygons_y[0], polygons_x[0], 'ro-')
    # plt.grid()
    # plt.title('Support polygon and CoM')
    # plt.xlabel('X [m]')
    # plt.ylabel('Y [m]')
    # # plt.xlim(0.5, -0.5)
    # plt.legend(['nominal', 'payload'])
    plt.show()


def single_comparison(sw_id, steps, step_clear, swing_time,
                      robot_mass, dt, min_force, slope, payloads=[10.0, 10.0]):

    dx = steps[0]
    dy = steps[1]
    dz = steps[2]

    step_num = len(sw_id)

    # initial state
    c0 = np.array([0.0822, 0.0009, -0.0222])
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
        np.array([0.6336, 0.27945, 0.298]),
        np.zeros(3),
    ]

    rmoving_contact = [
        np.array([0.6336, -0.27945, 0.298]),
        np.zeros(3),
    ]

    moving_contact = [lmoving_contact, rmoving_contact]

    swing_target = []
    for i in range(step_num):
        swing_target.append(
            [foot_contacts[sw_id[i]][0] + dx, foot_contacts[sw_id[i]][1] + dy, foot_contacts[sw_id[i]][2] + dz])

    swing_target = np.array(swing_target)

    w_nominal = Nominal(mass=robot_mass, N=int((swing_time[0:step_num][-1][1] + 1.0) / dt), dt=dt, slope_deg=slope)

    # w_nominal = Nominal(mass=robot_mass+payloads[0]+payloads[1], N=int((swing_time[0:step_num][-1][1] + 1.0) / dt), dt=dt)
    c_nom0 = np.array([0.1422, 0.0009, -0.0222])
    x_nom_init = np.hstack([c_nom0, dc0, ddc0])

    # sol is the directory returned by solve class function contains state, forces, control values
    sol1 = w_nominal.solve(x0=x_nom_init, contacts=foot_contacts, swing_id=sw_id, swing_tgt=swing_target,
                  swing_clearance=step_clear, swing_t=swing_time, min_f=min_force)


    # interpolate the values, pass values and interpolation resolution
    res = 300

    swing_currents = []
    for i in range(step_num):
        swing_currents.append(foot_contacts[sw_id[i]])

    interpl1 = w_nominal.interpolate(sol1, swing_currents, swing_target, step_clear, swing_time, res)

    # print the results
    #w1.print_trj(sol1, interpl1, res, foot_contacts, sw_id)

    # adaptable
    w_payload = Payload(mass=robot_mass, N=int((swing_time[0:step_num][-1][1] + 1.0) / dt), dt=dt,
                        slope_deg=slope, payload_masses=payloads)

    # sol is the directory returned by solve class function contains state, forces, control values
    sol2 = w_payload.solve(x0=x_init, contacts=foot_contacts, mov_contact_initial=moving_contact,
                    swing_id=sw_id, swing_tgt=swing_target, swing_clearance=step_clear,
                    swing_t=swing_time, min_f=min_force)


    # interpolate the values, pass values and interpolation resolution
    interpl2 = w_payload.interpolate(sol2, swing_currents, swing_target, step_clear, swing_time, res)

    # print the results
    #w1.print_trj(sol1, interpl1, res, foot_contacts, sw_id)

    # compute deviation
    CoM_deviation = compute_CoM_deviation(interpl1, interpl2)

    compare_print(interpl1, interpl2, foot_contacts, sw_id, swing_time[0:step_num], [dx, dy, dz], res)

    return CoM_deviation


def compute_CoM_deviation(nominal_trj, payload_trj):

    point_num = len(nominal_trj['x'][0])

    deviation = []
    for i in range(point_num):
        current_dev = np.sqrt((nominal_trj['x'][0][i] - payload_trj['x'][0][i])**2 +
                              (nominal_trj['x'][0][i] - payload_trj['x'][0][i])**2 +
                              (nominal_trj['x'][0][i] - payload_trj['x'][0][i])**2)
        deviation.append(current_dev)

    # plt.figure()
    # plt.plot(nominal_trj['t'], deviation)
    # plt.grid()
    # plt.xlabel('Time [s]')
    # plt.ylabel('CoM deviation [m]')
    # plt.show()

    return deviation


if __name__ == "__main__":

    dt_nlp = 0.2
    force_z = 50
    feet_id = [2, 0, 3, 1]
    clearance = 0.05
    swing_periods = [[1.0, 3.0], [4.0, 6.0], [7.0, 9.0], [10.0, 12.0]]
    centauro_mass = 112
    zero_slope = 0

    # 0.3 stepping
    scenario1_CoM = single_comparison(sw_id=feet_id, steps=[0.3, 0.0, 0.0], step_clear=clearance,
                                      swing_time=swing_periods,
                                      robot_mass=centauro_mass, dt=dt_nlp, min_force=force_z, slope=zero_slope)

    scenario2_CoM = single_comparison(sw_id=feet_id, steps=[0.3, 0.0, 0.0], step_clear=clearance,
                                      swing_time=swing_periods,
                                      robot_mass=centauro_mass, dt=dt_nlp, min_force=force_z, slope=zero_slope,
                                      payloads=[5.0, 10.0])

    # 2 step-ups on 20 cm platform
    scenario3_CoM = single_comparison(sw_id=[0, 1], steps=[0.2, 0.0, 0.2], step_clear=clearance,
                                      swing_time=swing_periods,
                                      robot_mass=centauro_mass, dt=dt_nlp, min_force=force_z, slope=zero_slope)

    # -10 deg inclined terrain
    scenario4_CoM = single_comparison(sw_id=feet_id, steps=[0.1, 0.0, 0.0], step_clear=clearance,
                                      swing_time=swing_periods,
                                      robot_mass=centauro_mass, dt=dt_nlp, min_force=force_z, slope=-10.0,
                                      payloads=[10.0, 10.0])

    # ------------------------------- 2x5 kg--------------------------------------------------
    # 0.3 stepping
    scenario2_1_CoM = single_comparison(sw_id=feet_id, steps=[0.3, 0.0, 0.0], step_clear=clearance,
                                        swing_time=swing_periods,
                                        robot_mass=centauro_mass, dt=dt_nlp, min_force=force_z, slope=zero_slope,
                                        payloads=[5.0, 5.0])

    # 2 step-ups on 20 cm platform
    scenario2_2_CoM = single_comparison(sw_id=[0, 1], steps=[0.2, 0.0, 0.2], step_clear=clearance,
                                        swing_time=swing_periods,
                                        robot_mass=centauro_mass, dt=dt_nlp, min_force=force_z, slope=zero_slope,
                                        payloads=[5.0, 5.0])

    # -10 deg inclined terrain
    scenario2_3_CoM = single_comparison(sw_id=feet_id, steps=[0.1, 0.0, 0.0], step_clear=clearance,
                                        swing_time=swing_periods,
                                        robot_mass=centauro_mass, dt=dt_nlp, min_force=force_z, slope=-10.0,
                                        payloads=[5.0, 5.0])

    # ------------------------------- 5 + 10 kg--------------------------------------------------
    # 0.3 stepping
    scenario3_1_CoM = single_comparison(sw_id=feet_id, steps=[0.3, 0.0, 0.0], step_clear=clearance,
                                        swing_time=swing_periods,
                                        robot_mass=centauro_mass, dt=dt_nlp, min_force=force_z, slope=zero_slope,
                                        payloads=[5.0, 10.0])

    # 2 step-ups on 20 cm platform
    scenario3_2_CoM = single_comparison(sw_id=[0, 1], steps=[0.2, 0.0, 0.2], step_clear=clearance,
                                        swing_time=swing_periods,
                                        robot_mass=centauro_mass, dt=dt_nlp, min_force=force_z, slope=zero_slope,
                                        payloads=[5.0, 10.0])

    # -10 deg inclined terrain
    scenario3_3_CoM = single_comparison(sw_id=feet_id, steps=[0.1, 0.0, 0.0], step_clear=clearance,
                                        swing_time=swing_periods,
                                        robot_mass=centauro_mass, dt=dt_nlp, min_force=force_z, slope=-10,
                                        payloads=[5.0, 10.0])


    # team of boxplots
    #boxplots_dict = {'Sc. 1': scenario1_CoM,
    #                 'Sc. 2': scenario2_CoM,
    #                 'Sc. 3': scenario3_CoM,
    #                 'Sc. 4': scenario4_CoM,
    #                 'sc2_1': scenario2_1_CoM,
    #                 'sc2_2': scenario2_2_CoM,
    #                 'sc2_3': scenario2_3_CoM,
    #                 'sc3_1': scenario3_1_CoM,
    #                 'sc3_2': scenario3_2_CoM,
    #                 'sc3_3': scenario3_3_CoM
    #                 }

    boxplots_dict = {' 0.3 m steps': scenario1_CoM + scenario2_1_CoM + scenario3_1_CoM,
                     ' Platform step-up': scenario3_CoM + scenario2_2_CoM + scenario3_2_CoM,
                     ' -10 deg. inclined terrain': scenario4_CoM + scenario2_3_CoM + scenario3_3_CoM
                     }

    fig, ax = plt.subplots()
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)

    boxprops = dict(linestyle='-', linewidth=4, color='black')
    flierprops = dict(marker='o', markerfacecolor='green', markersize=12,
                      markeredgecolor='none')
    medianprops = dict(linestyle='-', linewidth=3, color='red')
    meanpointprops = dict(marker='D', markeredgecolor='black',
                          markerfacecolor='firebrick')
    meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')

    ax.boxplot(boxplots_dict.values(), boxprops=boxprops, medianprops=medianprops)
    ax.set_xticklabels(boxplots_dict.keys(), fontsize=20)
    plt.grid()
    plt.ylabel('CoM position deviation [$m$]', fontsize=20)
    plt.show()