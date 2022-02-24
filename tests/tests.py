import numpy as np
import Receding_horizon as rh
from matplotlib import pyplot as plt
from gait_with_payload import GaitNonlinear as Gait
from Receding_horizon import Receding_hz_handler as Receding

if __name__ == '__main__':

    # object class of the optimization problem
    nlp_discr = 0.2
    optim_horizon = 4.0
    payload_m = [10.0, 10.0]
    inclination_deg = 0
    arm_box_conservative = False

    walk = Gait(mass=112, N=int(optim_horizon / nlp_discr), dt=nlp_discr, payload_masses=payload_m,
                        slope_deg=inclination_deg, conservative_box=arm_box_conservative)

    variables_dim = {
        'x': walk._dimx,
        'u': walk._dimu,
        'Pl_mov': walk._dimp_mov,
        'Pr_mov': walk._dimp_mov,
        'DPl_mov': walk._dimp_mov,
        'DPr_mov': walk._dimp_mov,
        'F': walk._dimf_tot,
        'F_virt_l': walk._dimf,
        'F_virt_r': walk._dimf
    }

    x0 = [0.08440614,  0.00099207, - 0.02779854, 0., 0., 0., 0., 0., 0.]
    contacts = [np.array([0.34942143, 0.34977262, -0.71884984]),
                np.array([0.34942143, -0.34977262, -0.71884984]),
                np.array([-0.3494216, 0.34977278, -0.71884984]),
                np.array([-0.3494216, -0.34977278, -0.71884984])]
    all_contacts = contacts
    moving_contact = [[np.array([0.52584379, 0.18904212, 0.28303459]), np.array([0., 0., 0.])],
                      [np.array([0.52582343, -0.18897632, 0.28300443]), np.array([0., 0., 0.])]]
    swing_id = [2]
    swing_tgt = [[-0.24942160289010637, 0.34977278149106616, -0.718849844313593]]
    swing_clear = 0.050
    swing_t = [[1.0, 3.0]]

    swing_contacts = [np.array([-0.3494216, 0.34977278, -0.71884984])]

    minimum_force, int_freq = 100, 300
    # call the solver of the optimization problem
    sol_previous = walk.solve(x0=x0, contacts=contacts, mov_contact_initial=moving_contact, swing_id=swing_id,
                     swing_tgt=swing_tgt, swing_clearance=swing_clear, swing_t=swing_t, min_f=minimum_force)
    interpl_previous = walk.interpolate(sol_previous, swing_contacts, swing_tgt, swing_clear, swing_t, int_freq)

    # receding time of the horizon
    knots_shift = 3
    horizon_shift = knots_shift * nlp_discr
    stride = [0.1, 0.0, 0.0]

    # sol = sol_previous
    # interpl = interpl_previous

    # handler
    mpc = Receding(horizon=optim_horizon, knots_toshift=knots_shift, nlp_dt=nlp_discr, desired_gait=[2, 0, 3, 1],
                   swing_dur=2.0, stance_dur=1.0, interpolation_freq=int_freq)

    mpc.set_current_contacts(contacts)
    mpc.set_current_swing_tgt(swing_tgt)
    mpc.set_previous_solution(sol_previous)
    mpc.set_previous_interpolated_solution(interpl_previous)
    mpc.set_swing_durations(swing_t, swing_id)
    mpc.count_optimizations(1)

    # for i in range(1):
    while True:

        # get shifted com and arm ee positions
        shifted_com_state = mpc.get_variable_after_knots_toshift(key_var='x', dimension_var=9)
        shifted_arm_ee = [
            [
                np.array(mpc.get_variable_after_knots_toshift(pos, 3)),
                np.array(mpc.get_variable_after_knots_toshift(vel, 3))
            ] for (pos, vel) in zip(['Pl_mov', 'Pr_mov'], ['DPl_mov', 'DPr_mov'])
        ]

        # new swing_t and swing_id for next optimization
        swing_t, swing_id, another_step = mpc.get_next_swing_durations(stride)

        # get initial guess
        shifted_guess = mpc.get_shifted_solution(variables_dim)

        # get target positions fot the swing legs
        # swing_tgt = mpc.get_swing_targets(contacts, [tgt_dx, tgt_dy, tgt_dz])

        old_nlp_params = walk._P
        new_nlp_params = mpc.get_updated_nlp_params(walk._P, swing_clear)

        # # update contacts
        # # contacts = [np.array(new_nlp_params[knots_shift][3*i:3*(i+1)]) for i in range(4)]
        nlp_params_extension = new_nlp_params[-3:]

        # access previous solution
        sol_previous = mpc.get_previous_solution()
        interpl_previous = mpc.get_previous_interpolated_solution()

        # debug some stuff
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('@@@@@@@@@ prev_swing_t', mpc._prev_swing_t)
        print('@@@@@@@@@ new swing_t', mpc._swing_t)
        print('@@@@@@@@@ prev_swing_id', mpc._prev_swing_id)
        print('@@@@@@@@@ new swing_id', mpc._swing_id)
        print('@@@@@@@@@ Another step:', another_step)
        print('@@@@@@@@@ Contacts:', mpc._contacts)
        print('@@@@@@@@@ Swing tgt:', mpc._swing_tgt)
        # for i in range(knots_shift):
        #     print('@@@@@@@@@ New nlp params:', nlp_params_extension[i])
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

        shift_lamultx = mpc.get_shifted_variable(sol_previous['lam_x'], int(walk._nvars / mpc._knot_number))

        sol = walk.solve(x0=shifted_com_state, contacts=mpc._contacts, mov_contact_initial=shifted_arm_ee, swing_id=mpc._swing_id,
                         swing_tgt=mpc._swing_tgt, swing_clearance=swing_clear, swing_t=mpc._swing_t, min_f=minimum_force,
                         init_guess=shifted_guess, state_lamult=sol_previous['lam_x'], constr_lamult=sol_previous['lam_g'],
                         nlp_params=new_nlp_params)

        # print(next_swing_leg_pos)
        interpl = walk.interpolate(sol, [mpc._contacts[ii] for ii in mpc._swing_id], mpc._swing_tgt, swing_clear,
                                   mpc._swing_t, int_freq, feet_ee_swing_trj=interpl_previous['sw'],
                                   shift_time=mpc._time_shifting, skip_useless=True)

        # walk.print_trj(sol1, interpl1, int_freq, contacts, swing_id)
        print(interpl_previous['sw'][0]['z'][180], interpl['sw'][0]['z'][0])

        # debug
        if old_nlp_params[3:] == new_nlp_params[:-3]:
            print('Params shifted correctly')
        else:
            print('Attention. Params not shifted correctly')
        # print('Last part of params:', new_nlp_params[-3:])

        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # ^^^^^^^^^^^^^^^^^^^ Print plots ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # Interpolated state plot
        # shifted_time = [0.6 + i for i in walk._tjunctions]#walk._tjunctions[3:] + [(walk._tjunctions[-1] + walk._dt*i) for i in range(1, 4)]
        # shifted_interpol_time = [0.6 + i for i in interpl['t']]
        #
        # state_labels = ['CoM Position', 'CoM Velocity', 'CoM Acceleration']
        # colors = ['r', 'g', 'b']
        # plt.figure()
        # for i, name in enumerate(state_labels):
        #     plt.subplot(3, 1, i + 1)
        #     for j in range(walk._dimc):
        #         # plt.plot(interpl_previous['t'], interpl_previous['x'][walk._dimc * i + j], '-')
        #         # plt.plot(shifted_interpol_time, interpl['x'][walk._dimc * i + j], '-')
        #
        #         plt.plot(walk._tjunctions, sol_previous['x'][walk._dimc * i + j::9], '.-')
        #         plt.plot(shifted_time, sol['x'][walk._dimc * i + j::9], '.--')
        #     plt.grid()
        #     plt.legend(['x', 'x1', 'y', 'y1', 'z', 'z1'])
        #     plt.title(name)
        # plt.xlabel('Time [s]')
        #
        # feet_labels = ['front left', 'front right', 'hind left', 'hind right']
        # # Interpolated force plot
        # plt.figure()
        # for i, name in enumerate(feet_labels):
        #     plt.subplot(2, 2, i + 1)
        #     for k in range(3):
        #         plt.plot(interpl_previous['t'], interpl_previous['f'][3 * i + k], '-')
        #         plt.plot(interpl['t'], interpl['f'][3 * i + k], '-')
        #         plt.plot(walk._tjunctions, sol['F'][3 * i + k::walk._dimf_tot], '.')
        #     plt.grid()
        #     plt.title(name)
        #     plt.legend([str(name) + '_x', str(name) + '_x1',
        #                 str(name) + '_y', str(name) + '_y2',
        #                 str(name) + '_z', str(name) + '_z1'])
        # plt.xlabel('Time [s]')
        # # plt.savefig('../plots/gait_forces.png')
        #
        # # Interpolated moving contact trajectory
        # mov_contact_labels = ['p_mov_l', 'dp_mov_l', 'ddp_mov_l', 'p_mov_r', 'dp_mov_r', 'ddp_mov_r']
        # plt.figure()
        # for i, name in enumerate(mov_contact_labels):
        #     plt.subplot(2, 3, i + 1)
        #     for k in range(3):
        #         plt.plot(shifted_interpol_time, interpl[name][k], '.--')
        #         plt.plot(interpl_previous['t'], interpl_previous[name][k], '.-')
        #         plt.grid()
        #         plt.legend(['x', 'x1', 'y', 'y1', 'z', 'z1'])
        #     plt.ylabel(name)
        #     plt.suptitle('Moving Contact trajectory')
        # plt.xlabel('Time [s]')

        # plot swing trajectory
        # All points to be published
        # N_total = int(walk._Nseg * walk._dt * int_freq)  # total points --> total time * frequency
        # s = np.linspace(0, walk._dt * walk._Nseg, N_total)
        # coord_labels = ['x', 'y', 'z']
        # for j in range(min(len(interpl_previous['sw']), len(interpl['sw']))):
        #     plt.figure()
        #     for i, name in enumerate(coord_labels):
        #         plt.subplot(3, 1, i + 1)
        #         plt.plot(interpl_previous['sw'][j][name])  # nominal trj
        #         plt.plot([None]*180 + interpl['sw'][j][name])  # nominal trj
        #
        #         # plt.plot(s[0:t_exec[j]], results['sw'][j][name][0:t_exec[j]])  # executed trj
        #         plt.grid()
        #         plt.legend(['previous', 'current'])
        #         plt.title('Trajectory ' + name)
        #     plt.xlabel('Time [s]')
        #
        # plt.show()

        # set to general variables
        mpc.count_optimizations(1)      # solutions_counter += 1
        mpc.set_previous_solution(sol)
        mpc.set_previous_interpolated_solution(interpl)
        # mpc.set_previous_swing_durations(swing_t, swing_id)
        # sol_previous = sol
        # interpl_previous = interpl



