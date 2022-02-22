import numpy as np
import Receding_horizon as rh
from matplotlib import pyplot as plt
from gait_with_payload import GaitNonlinear as Gait
from gait import Gait as SimpleGait

if __name__ == '__main__':

    # swing_dur = [[0.0, 0.6], [1.6, 3.6], [4.6, 7.0]]
    # swing_id = [2,0,3]
    # for i in range(50):
    # # while True:
    #     swing_dur, swing_id, k = get_swing_durations(swing_dur, swing_id, [3, 1, 4, 2], 0.6, 7.0)
    #
    #     print('Loop: ', ': ', len(swing_dur), len(swing_id))
    #     # print('gait pattern: ', len(swing_id))

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
    # sol = sol_previous
    # interpl = interpl_previous

    solutions_counter = 1   # counter of solutions acquired
    # for i in range(1):
    while True:

        # start of the next horizon wrt to initial time
        start_of_next_horizon = solutions_counter * horizon_shift

        difference = start_of_next_horizon - optim_horizon
        if difference > 0.0:
            solutions_counter = 1
            start_of_next_horizon = solutions_counter * horizon_shift

        # print('________', knots_shift*walk._dimx , (knots_shift + 1)*walk._dimx)
        # update arguments of solve function
        x0 = sol_previous['x'][knots_shift*walk._dimx : (knots_shift + 1)*walk._dimx]
        moving_contact = [[np.array(sol_previous['Pl_mov'][knots_shift*walk._dimp_mov : (knots_shift + 1)*walk._dimp_mov]),
                           np.array(sol_previous['DPl_mov'][knots_shift*walk._dimp_mov : (knots_shift + 1)*walk._dimp_mov])],
                          [np.array(sol_previous['Pr_mov'][knots_shift*walk._dimp_mov : (knots_shift + 1)*walk._dimp_mov]),
                           np.array(sol_previous['DPr_mov'][knots_shift*walk._dimp_mov : (knots_shift + 1)*walk._dimp_mov])]]

        # swing contacts based on previous plan at the desired time (start of next planning horizon)
        # prev_swing_leg_pos = rh.get_current_leg_pos(interpl['sw'], swing_id, start_of_next_horizon, 300)
        # for i in swing_id:
        #     contacts[i] = prev_swing_leg_pos[swing_id.index(i)]

        prev_swing_t = swing_t      # save old swing_t and swing_id
        prev_swing_id = swing_id

        # debug some stuff
        # print('**Initial state:', x0)
        # print('**Moving contact:', moving_contact)
        # print('**Initial contacts:', contacts)
        # print('**All contacts:', all_contacts)

        print('@@@@@@@@@ prev_swing_t', prev_swing_t)
        print('@@@@@@@@@ prev_swing_id', prev_swing_id)

        # new swing_t and swing_id for next optimization
        swing_t, swing_id, another_step = rh.get_swing_durations(prev_swing_t, prev_swing_id, [2, 0, 3, 1],
                                                                 horizon_shift, optim_horizon)
        # debug some stuff
        print('======', prev_swing_id)
        print('====== New Swing timings:', swing_t)
        print('====== New Swing id:', swing_id)
        print('====== Another step:', another_step)

        # # form position of swing legs for next optimization
        # if another_step[0] is True:     # new swing phase added at the end of the horizon
        #     next_swing_leg_pos = prev_swing_leg_pos + [np.array(contacts[swing_id[-1]])]
        # else:     # no new swing phase added at the end of the horizon
        #     next_swing_leg_pos = prev_swing_leg_pos
        #
        # if another_step[1] is True:     # first swing phase removed because it has passed
        #     next_swing_leg_pos = next_swing_leg_pos[1:]

        # # debug some stuff
        # print('!!!!Prev swing leg pos:', prev_swing_leg_pos)
        # print('!!!!Next swing leg pos:', next_swing_leg_pos)

        # get initial guess
        shifted_guess = rh.shift_solution(sol_previous, knots_shift, variables_dim)

        # update tgt_dx heuristically
        new_step_num = len(swing_id)
        tgt_dx = [0.1] * new_step_num
        tgt_dy = [0.0] * new_step_num
        tgt_dz = [0.0] * new_step_num
        print('!!!!', tgt_dx, tgt_dy, tgt_dz)

        # get target positions fot the swing legs
        swing_tgt = rh.get_swing_targets(swing_id, contacts, [tgt_dx, tgt_dy, tgt_dz])

        # debug some stuff
        print('================================================')
        print('================ Solver inputs =====================')
        print('================================================')
        print('**Initial state:', x0)
        print('**All contacts:', contacts)
        print('**Moving contact:', moving_contact)
        print('==Swing id:', swing_id)
        print('==Swing tgt:', swing_tgt)
        print('==Swing clear:', swing_clear)
        print('==Swing_t:', swing_t)
        print('================================================')
        print('================================================')

        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print('^^^^^^^^^^^^^^^^ Targets ^^^^^^^^^^^^^^^^^^^')
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print('Swing tgt:', swing_tgt)
        print('Contacts:', contacts)
        # print('Next swing leg position', next_swing_leg_pos)
        print('PPPPPPPPP previous:', walk._P[0][:6])
        old_nlp_params = walk._P
        new_nlp_params = rh.get_updated_nlp_params(walk._P, knots_shift, another_step, swing_id, swing_t,
                                                   swing_tgt, contacts, swing_clear)
        # update contacts
        contacts = [np.array(new_nlp_params[knots_shift][3*i:3*(i+1)]) for i in range(4)]
        nlp_params_extension = new_nlp_params[-3:]
        if old_nlp_params[3:] == new_nlp_params[:-3]:
            print('Params shifted correctly')
        else:
            print('Attention. Params not shifted correctly')
        print('Last part of params:', new_nlp_params[-3:])

        # print('PPPPPPPPP new:', new_nlp_params[0][:6])
        sol = walk.solve(x0=x0, contacts=contacts, mov_contact_initial=moving_contact, swing_id=swing_id,
                         swing_tgt=swing_tgt, swing_clearance=swing_clear, swing_t=swing_t, min_f=minimum_force,
                         init_guess=shifted_guess, state_lamult=sol_previous['lam_x'], constr_lamult=sol_previous['lam_g'],
                         nlp_params=new_nlp_params)
        # # debug force plot
        # tt = np.linspace(0.0, (swing_t[-1][1] + 1.0), walk._knot_number)
        # plt.figure()
        # for i, name in enumerate(['fl', 'fr', 'hl', 'hr']):
        #     plt.subplot(2, 2, i + 1)
        #     for k in range(3):
        #         plt.plot(tt, sol['F'][3 * i + k::12], '.-')
        #     plt.grid()
        #     plt.title(name)
        #     plt.legend([str(name) + '_x', str(name) + '_y', str(name) + '_z'])
        # plt.xlabel('Time [s]')
        # plt.show()

        # print(next_swing_leg_pos)
        interpl = walk.interpolate(sol, [contacts[ii] for ii in swing_id], swing_tgt, swing_clear, swing_t, int_freq,
                                   feet_ee_swing_trj=interpl_previous['sw'])

        # walk.print_trj(sol1, interpl1, int_freq, contacts, swing_id)

        solutions_counter += 1
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # ^^^^^^^^^^^^^^^^^^^ Print plots ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        # # Interpolated state plot
        # shifted_time = [0.6 + i for i in walk._tjunctions]#walk._tjunctions[3:] + [(walk._tjunctions[-1] + walk._dt*i) for i in range(1, 4)]
        # shifted_interpol_time = [0.6 + i for i in interpl['t']]
        #
        # state_labels = ['CoM Position', 'CoM Velocity', 'CoM Acceleration']
        # colors = ['r', 'g', 'b']
        # plt.figure()
        # for i, name in enumerate(state_labels):
        #     plt.subplot(3, 1, i + 1)
        #     for j in range(walk._dimc):
        #         plt.plot(interpl_previous['t'], interpl_previous['x'][walk._dimc * i + j], '-')
        #         plt.plot(shifted_interpol_time, interpl['x'][walk._dimc * i + j], '-')
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

        # # plot swing trajectory
        # # All points to be published
        # N_total = int(walk._Nseg * walk._dt * int_freq)  # total points --> total time * frequency
        # s = np.linspace(0, walk._dt * walk._Nseg, N_total)
        # coord_labels = ['x', 'y', 'z']
        # for j in range(len(interpl_previous['sw'])):
        #     plt.figure()
        #     for i, name in enumerate(coord_labels):
        #         plt.subplot(3, 1, i + 1)
        #         plt.plot(interpl_previous['t'], interpl_previous['sw'][j][name])  # nominal trj
        #         plt.plot(interpl['t'], interpl['sw'][j][name])  # nominal trj
        #
        #         # plt.plot(s[0:t_exec[j]], results['sw'][j][name][0:t_exec[j]])  # executed trj
        #         plt.grid()
        #         plt.legend(['nominal', 'real'])
        #         plt.title('Trajectory ' + name)
        #     plt.xlabel('Time [s]')

        # plt.show()

        # set to general variables
        sol_previous = sol
        interpl_previous = interpl



