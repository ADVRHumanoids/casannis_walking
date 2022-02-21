import numpy as np
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


def get_swing_durations(previous_dur, swing_id, desired_gait_pattern, time_shifting, horizon_dur,
                        swing_dur=2.0, stance_dur=1.0):
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
    new_swing_time = durations_flat[-2] + swing_dur + stance_dur

    # if next swing phase is within the horizon to plan, add it in the list
    if horizon_dur > new_swing_time:
        # identify which swing leg is the next one
        last_swing_id = new_swing_id[-1]        # the last leg that is stepping in the horizon
        last_swing_index = desired_gait_pattern.index(last_swing_id)        # index of the last step
        new_swing_id.append(        # append new swing leg id based of desired gait pattern
            desired_gait_pattern[last_swing_index - max_step_num + 1]
        )

        # append new swing phase timings
        durations_flat.append(new_swing_time)
        durations_flat.append(min(new_swing_time + swing_dur, horizon_dur))
        started_step = True  # flag to show that swing phase was finished

    # if duration of the last swing phase to be planned is less than default duration,
    # then swing phase should last more
    # last_duration = round(durations_flat[-1] - durations_flat[-2], 2)
    final_swing_phase_end_time = durations_flat[-2] + swing_dur
    # if final_swing_phase_end_time < horizon_dur:
    durations_flat[-1] = min(final_swing_phase_end_time, horizon_dur)
    # if last_duration < swing_dur:
    #     durations_flat[-1] = durations_flat[-2] + swing_dur

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


def get_current_leg_pos(swing_trj, previous_gait_pattern, desired_time, freq):
    '''
    Get the position of the legs at the desired time in the horizon of the last optimization based on this last plan.
    :param swing_trj: the planned trajectories of the swing legs from the last motion plan.
    It is a list of dictionaries with keys ['x', 'y', 'z', 's'] as returned by the walking.interpolate method.
    :param previous_gait_pattern: order of the swing legs of the last optimization
    :param desired_time: desired time wrt to the start of the last optimization horizon
    :param freq: frequency at which the last motion plan was interpolated
    :return: swing_ee_pos which is a list of np.array(x,y,z). It is the position of the swing legs of the last
    motion plan at the desired time within the last opt. horizon.
    '''

    trj_index = int(desired_time * freq)       # index of the trajectory to which the desired time corresponds
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
        if True: #new_knots_in_last_swing[k - knot_num + 3] is True:
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

    return next_params