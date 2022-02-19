import numpy as np
import copy
import constraints


def shift_solution(solution, knots_toshift, dims):
    '''
    Generate the new initial guess by shifting the previous solution.
    :param solution: previous solution
    :param knots_toshift: knots to shift the new solution
    :param dims: list with dimensions of the different variables
    :return: the shifted solution which can be used as initial guess
    '''

    new_values = [0]
    # solution_keys = solution.keys()
    solution_based_ordered_keys = ['x', 'u', 'F', 'Pl_mov', 'Pr_mov', 'DPl_mov', 'DPr_mov', 'F_virt_l', 'F_virt_r']

    # shifted_sol = {}
    shifted_sol_array = []
    for keyname in solution_based_ordered_keys:
        shifted_variables = solution[keyname][knots_toshift * dims[keyname]:] + \
                            knots_toshift * solution[keyname][-knots_toshift * dims[keyname]:]  # same as last knot
                            # knots_toshift * dims[keyname] * new_values    # zero new values

        # shifted_sol.update({keyname: shifted_variables})
        shifted_sol_array = np.hstack((shifted_sol_array, shifted_variables))

    return shifted_sol_array


def get_swing_durations(previous_dur, swing_id, desired_gait_pattern, time_shifting, horizon_dur, swing_dur=2.0, stance_dur=1.0):
    '''
    Compute the swing durations for the next optimization based on a default duration of swing and stance periods and
    the optimization horizon. Compute also the swing_id for the next optimization based on desired gait pattern and
    swing_id of previous solution.
    :param previous_dur: swing durations of last optimization
    :param swing_id: swing id of last optimization
    :param desired_gait_pattern: desired gait pattern to be followed
    :param time_shifting: shifting of the optimization horizon
    :param horizon_dur: duration of the horizon
    :param swing_dur: duration of swing periods
    :param stance_dur: duration of stance periods
    :return: swing_durations and new_swing_id for the next optimization
    '''

    finished_step = False
    started_step = False        # flags

    max_step_num = len(desired_gait_pattern)    # maximum number of steps defined in the desired gait pattern
    new_swing_id = swing_id                     # first set the new swing id same as the previous

    durations_flat = [a for k in previous_dur for a in k]   # convert to flat list
    durations_flat = (np.array(durations_flat) - time_shifting).tolist()    # shift all timings
    durations_flat = [round(a, 2) for a in durations_flat]      # round on 2 decimal digits

    # if first swing phase has elapsed already
    if durations_flat[1] <= 0.0:
        durations_flat = durations_flat[2:]     # delete first swing phase
        new_swing_id = new_swing_id[1:]         # delete first swing id
        finished_step = True                    # flag to show that swing phase was finished

    # else if first swing phase has started but not elapsed
    elif durations_flat[0] < 0.0:
        durations_flat[0] = 0.0     # set starting of first swing phase to 0.0 time of the horizon

    # time of the next swing phase
    new_swing_time = durations_flat[-1] + stance_dur

    # if next swing phase is within the horizon to plan
    if horizon_dur > new_swing_time:
        last_swing_id = new_swing_id[-1]        # the last leg that is stepping in the horizon
        last_swing_index = desired_gait_pattern.index(last_swing_id)        # index of the last step
        new_swing_id.append(        # append new swing leg id based of desired gait pattern
            desired_gait_pattern[last_swing_index - max_step_num + 1]
        )

        # append new swing phase timings
        durations_flat.append(new_swing_time)
        durations_flat.append(min(new_swing_time + swing_dur, horizon_dur))
        started_step = True  # flag to show that swing phase was finished

    # if duration of the last swing phase to be planned is less than default duration, then swing phase should
    # last until the end of the horizon
    last_duration = round(durations_flat[-1] - durations_flat[-2], 2)
    if last_duration < swing_dur:
        durations_flat[-1] = horizon_dur

    # convert to list of lists
    half_list_size = int(len(durations_flat)/2)     # half size of the flat list
    swing_durations = [[durations_flat[2*a], durations_flat[2*a+1]] for a in range(half_list_size)]

    # print('IIIIIIIIIIIII', swing_durations)
    # print('IIIIIIIIIIIII', new_swing_id)
    return swing_durations, new_swing_id, [started_step, finished_step]


def get_swing_targets(gait_pattern, contacts, strides):
    '''
    Returns the target positions of the legs that are planned to swing.
    :param gait_pattern: order of legs to be swinged
    :param contacts: current contact positions = leg ee positions
    :param strides: the stride length in the 3 directions --> [dx, dy, dz] where dx is a list of length for each
    swing leg
    :return: Returns swing_tgt which is a list of lists with the target position for the legs to swing.
    '''

    tgt_dx = strides[0]
    tgt_dy = strides[1]
    tgt_dz = strides[2]

    step_num = len(gait_pattern)    # number of planned steps

    swing_tgt = []  # target positions as list

    for i in range(step_num):
        # targets
        swing_tgt.append([contacts[gait_pattern[i]][0] + tgt_dx[i],
                          contacts[gait_pattern[i]][1] + tgt_dy[i],
                          contacts[gait_pattern[i]][2] + tgt_dz[i]])

    return swing_tgt


def get_current_leg_pos(swing_trj, previous_gait_pattern, time_shifting, freq):
    '''
    Get the position of the legs at the desired time in the horizon of the last optimization based on this last plan.
    :param swing_trj: the planned trajectories of the swing legs from the last motion plan.
    It is a list of dictionaries with keys ['x', 'y', 'z', 's'] as returned by the walking.interpolate method.
    :param previous_gait_pattern: order of the swing legs of the last optimization
    :param time_shifting: desired time wrt to the start of the last optimization horizon
    :param freq: frequency at which the last motion plan was interpolated
    :return: swing_ee_pos which is a list of np.array(x,y,z). It is the position of the swing legs of the last
    motion plan at the desired time within the last opt. horizon.
    '''

    trj_index = int(time_shifting * freq)       # index of the trajectory to which the desired time corresponds
    step_num = len(previous_gait_pattern)       # number of last planned steps

    # todo swing_trj does not consider repeated swing legs

    # append positions of the legs that were last planned to swing
    swing_ee_pos = []
    for i in range(step_num):
        swing_ee_pos.append(np.array([swing_trj[i][coord_name][trj_index] for coord_name in ['x', 'y', 'z']]))
    return swing_ee_pos


def get_updated_nlp_params(previous_params, knots_to_shift, different_step_phase, swing_id, swing_t, swing_tgt,
                           contacts, swing_clearance, nlp_discretization=0.2, swing_phase_dur=2.0):

    knot_num = len(previous_params)
    opt_horizon = float(knot_num - 1) * nlp_discretization
    next_params = previous_params[knots_to_shift:]

    new_knot_times = [nlp_discretization * i for i in range(knot_num-3, knot_num)]
    new_knots_in_last_swing = [swing_t[-1][0] <= i <= swing_t[-1][1] for i in new_knot_times]

    for k in range(knot_num - 3, knot_num):
        if True:#new_knots_in_last_swing[i] is True:
            # time that maximum clearance occurs - approximate
            clearance_times = swing_t[-1][0] + 0.5 * swing_phase_dur

            # number of steps
            # step_num = len(swing_id)

            # swing feet positions at maximum clearance
            clearance_swing_position = []

            if contacts[swing_id[-1]][2] >= swing_tgt[-1][2]:
                clearance_swing_position.append(contacts[swing_id[-1]][0:2].tolist() +
                                                [contacts[swing_id[-1]][2] + swing_clearance])
            else:
                clearance_swing_position.append(contacts[swing_id[-1]][0:2].tolist() +
                                                [swing_tgt[-1][2] + swing_clearance])

            # for k in range(knot_num - 3, knot_num):
            current_knot_params = constraints.set_contact_parameters(
                contacts, [swing_id[-1]], swing_tgt[-1], [clearance_times], clearance_swing_position[-1],
                k, nlp_discretization, steps_number=1
            )

            next_params.append(current_knot_params)
        # else:
        #     next_params.append()

    # # if different_step_phase[0] is True:
    # if swing_t[-1][1] == opt_horizon:
    #     # time that maximum clearance occurs - approximate
    #     clearance_times = swing_t[-1][0] + 0.5 * swing_phase_dur
    #
    #     # number of steps
    #     # step_num = len(swing_id)
    #
    #     # swing feet positions at maximum clearance
    #     clearance_swing_position = []
    #
    #     if contacts[swing_id[-1]][2] >= swing_tgt[-1][2]:
    #         clearance_swing_position.append(contacts[swing_id[-1]][0:2].tolist() +
    #                                         [contacts[swing_id[-1]][2] + swing_clearance])
    #     else:
    #         clearance_swing_position.append(contacts[swing_id[-1]][0:2].tolist() +
    #                                         [swing_tgt[-1][2] + swing_clearance])
    #
    #     for k in range(knot_num-3, knot_num):
    #         current_knot_params = constraints.set_contact_parameters(
    #             contacts, [swing_id[-1]], swing_tgt[-1], [clearance_times], clearance_swing_position[-1],
    #             k, nlp_discretization, steps_number=1
    #         )
    #
    #         next_params.append(current_knot_params)
    # else:
    #     next_params = next_params + next_params[-knots_to_shift:]

    return next_params


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
    optim_horizon = 7.0
    payload_m = [10.0,10.0]
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
    swing_id = [2, 0]
    swing_tgt = [[-0.24942160289010637, 0.34977278149106616, -0.718849844313593],
                 [0.4494214287188567, 0.3497726202477771, -0.7188498443127623]]
    swing_clear = 0.050
    swing_t = [[1.0, 3.0], [4.0, 6.0]]

    swing_contacts = [np.array([-0.3494216, 0.34977278, -0.71884984]),
                      np.array([0.34942143, 0.34977262, -0.71884984])]

    minimum_force, int_freq = 100, 300
    # call the solver of the optimization problem
    sol = walk.solve(x0=x0, contacts=contacts, mov_contact_initial=moving_contact, swing_id=swing_id,
                     swing_tgt=swing_tgt, swing_clearance=swing_clear, swing_t=swing_t, min_f=minimum_force)
    interpl = walk.interpolate(sol, swing_contacts, swing_tgt, swing_clear, swing_t, int_freq)

    # receding time of the horizon
    knots_shift = 3
    horizon_shift = knots_shift * nlp_discr

    # for i in range(20):
    while True:
        # print('________', knots_shift*walk._dimx , (knots_shift + 1)*walk._dimx)
        # update arguments of solve function
        x0 = sol['x'][knots_shift*walk._dimx : (knots_shift + 1)*walk._dimx]
        moving_contact = [[np.array(sol['Pl_mov'][knots_shift*walk._dimp_mov : (knots_shift + 1)*walk._dimp_mov]),
                           np.array(sol['DPl_mov'][knots_shift*walk._dimp_mov : (knots_shift + 1)*walk._dimp_mov])],
                          [np.array(sol['Pr_mov'][knots_shift*walk._dimp_mov : (knots_shift + 1)*walk._dimp_mov]),
                           np.array(sol['DPr_mov'][knots_shift*walk._dimp_mov : (knots_shift + 1)*walk._dimp_mov])]]

        # swing contacts based on previous plan at the desired time (start of next planning horizon)
        prev_swing_leg_pos = get_current_leg_pos(interpl['sw'], swing_id, horizon_shift, 300)
        for i in swing_id:
            contacts[i] = prev_swing_leg_pos[swing_id.index(i)]

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
        swing_t, swing_id, another_step = get_swing_durations(prev_swing_t, prev_swing_id, [2, 0, 3, 1],
                                                              horizon_shift, optim_horizon)
        # debug some stuff
        print('======', prev_swing_id)
        print('====== New Swing timings:', swing_t)
        print('====== New Swing id:', swing_id)
        print('====== Another step:', another_step)

        # form position of swing legs for next optimization
        if another_step[0] is True:
            next_swing_leg_pos = prev_swing_leg_pos + [np.array(contacts[swing_id[-1]])]
        else:
            next_swing_leg_pos = prev_swing_leg_pos

        if another_step[1] is True:
            next_swing_leg_pos = next_swing_leg_pos[1:]

        # # debug some stuff
        # print('!!!!Prev swing leg pos:', prev_swing_leg_pos)
        # print('!!!!Next swing leg pos:', next_swing_leg_pos)

        # get initial guess
        shifted_guess = shift_solution(sol, 1, variables_dim)

        # update tgt_dx heuristically
        new_step_num = len(swing_id)
        tgt_dx = [0.1] * new_step_num
        tgt_dy = [0.0] * new_step_num
        tgt_dz = [0.0] * new_step_num
        print('!!!!', tgt_dx, tgt_dy, tgt_dz)

        # get target positions fot the swing legs
        swing_tgt = get_swing_targets(swing_id, contacts, [tgt_dx, tgt_dy, tgt_dz])

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
        print('PPPPPPPPP', walk._P)
        new_nlp_params = get_updated_nlp_params(walk._P, knots_shift, another_step, swing_id, swing_t,
                                                swing_tgt, contacts, swing_clear)

        sol = walk.solve(x0=x0, contacts=contacts, mov_contact_initial=moving_contact, swing_id=swing_id,
                         swing_tgt=swing_tgt, swing_clearance=swing_clear, swing_t=swing_t, min_f=minimum_force,
                         init_guess=shifted_guess, state_lamult=sol['lam_x'], constr_lamult=sol['lam_g'],
                         nlp_params=new_nlp_params)
        print('PPPPPPPPP', walk._P)
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

        print(next_swing_leg_pos)
        interpl = walk.interpolate(sol, next_swing_leg_pos, swing_tgt, swing_clear, swing_t, int_freq)

    # optim_horizon = 7.0
    # nlp_discr = 0.2
    # payload_m = [10.0, 10.0]
    # inclination_deg = 0.0
    # arm_box_conservative = False
    #
    # walk = Gait(mass=112, N=int(optim_horizon / nlp_discr), dt=nlp_discr, payload_masses=payload_m,
    #                     slope_deg=inclination_deg, conservative_box=arm_box_conservative)
    # # walk = SimpleGait(mass=112, N=int(optim_horizon / nlp_discr), dt=nlp_discr,
    # #                     slope_deg=inclination_deg)
    # x0 = [0.06730191639719647, 0.01552940421869955, -0.020572223202718995,
    #       -0.03245918116395145, 0.04225748795988705, 0.003309643059751207,
    #       -0.05942779069968623, 0.010279407423706612, -0.00017711019279872668]
    #
    # contacts = [np.array([ 0.34942143,  0.34977262, -0.71884984]),
    #             np.array([ 0.34942143, -0.34977262, -0.71884984]),
    #             np.array([-0.3494216 ,  0.34977278, -0.71884984]),
    #             np.array([-0.3494216 , -0.34977278, -0.71884984])]
    #
    # moving_contact = [[np.array([0.4299506 , 0.34797929, 0.32349138]),
    #                    np.array([-0.091154  ,  0.1098635 ,  0.02881774])],
    #                   [np.array([0.43488849, 0.10052735, 0.31455664]),
    #                    np.array([-0.10567345,  0.10306788,  0.04993922])]]
    #
    # swing_id = [1, 2, 0]
    # swing_tgt = [[0.4494214287161541, -0.3497726202507546, -0.7188498443129442],
    #              [-0.24942160289010637, 0.34977278149106616, -0.718849844313593],
    #              [0.4494214287188567, 0.3497726202477771, -0.7188498443127623]]
    #
    #
    # swing_clear = 0.050
    # swing_t = [[0.2, 2.2], [3.2, 5.2], [6.2, 7.0]]
    #
    # # call the solver of the optimization problem
    # sol = walk.solve(x0=x0, contacts=contacts, mov_contact_initial=moving_contact, swing_id=swing_id,
    #                  swing_tgt=swing_tgt, swing_clearance=swing_clear, swing_t=swing_t, min_f=100)
    # # sol = walk.solve(x0=x0, contacts=contacts, swing_id=swing_id,
    # #                  swing_tgt=swing_tgt, swing_clearance=swing_clear, swing_t=swing_t, min_f=100)
    # int_freq = 300
    # swing_contacts = [np.array([ 0.34942143, -0.34977262, -0.71884984]),
    #                   np.array([-0.3494216 ,  0.34977278, -0.71884984]),
    #                   np.array([0.34942143,  0.34977262, -0.71884984])]
    # interpl = walk.interpolate(sol, swing_contacts, swing_tgt, swing_clear, swing_t, int_freq)
    # walk.print_trj(sol, interpl, int_freq, contacts, swing_id)
