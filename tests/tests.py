import numpy as np
import copy


def get_swing_durations(previous_dur, swing_id, desired_gait_pattern, time_shifting, horizon_dur, swing_dur=2.0, stance_dur=1.0):
    #  todo: code a generator of:
    #  initial contacts  of legs

    step_num = len(desired_gait_pattern)
    new_gait_pattern = swing_id

    durations_flat = [a for k in previous_dur for a in k]
    durations_flat = (np.array(durations_flat) - time_shifting).tolist()
    durations_flat = [round(a, 2) for a in durations_flat]

    if durations_flat[1] <= 0.0:
        durations_flat = durations_flat[2:]
        new_gait_pattern = new_gait_pattern[1:]

    elif durations_flat[0] < 0.0:
        durations_flat[0] = 0.0

    new_swing_time = durations_flat[-1] + stance_dur

    if horizon_dur > new_swing_time:
        #gait = [3,1,4,2]
        last_swing_id = new_gait_pattern[-1]
        last_swing_index = desired_gait_pattern.index(last_swing_id)
        new_gait_pattern.append(desired_gait_pattern[last_swing_index - step_num + 1])

        durations_flat.append(new_swing_time)
        durations_flat.append(min(new_swing_time + swing_dur, horizon_dur))

    last_duration = round(durations_flat[-1] - durations_flat[-2], 2)
    if last_duration < swing_dur:
        durations_flat[-1] = horizon_dur

    half_list_size = int(len(durations_flat)/2)
    swing_durations = [[durations_flat[2*a], durations_flat[2*a+1]] for a in range(half_list_size)]

    return swing_durations, new_gait_pattern


def get_current_leg_pos(swing_trj, time_shifting, freq):
    #contacts = [np.array(x) for x in f_cont] = [nparray[x,y,z], ...]

    trj_index = time_shifting * freq
    leg_ee_pos = []
    for i in range(4):
        leg_ee_pos.append(np.array([swing_trj[i][coord_name][trj_index] for coord_name in ['x', 'y', 'z']]))

    return leg_ee_pos


def get_swing_targets(gait_pattern, contacts, strides):

    tgt_dx = [k[0] for k in strides]
    tgt_dy = [k[1] for k in strides]
    tgt_dz = [k[2] for k in strides]

    step_num = len(gait_pattern)

    swing_tgt = []  # target positions as list

    for i in range(step_num):
        # targets
        swing_tgt.append([contacts[gait_pattern[i] - 1][0] + tgt_dx[i],
                          contacts[gait_pattern[i] - 1][1] + tgt_dy[i],
                          contacts[gait_pattern[i] - 1][2] + tgt_dz[i]])

    return swing_tgt


if __name__ == '__main__':

    swing_dur = [[1.0, 3.0], [4.0, 6.0], [7.0, 9.0]]
    swing_id = [3, 1, 4]
    for i in range(50):
    # while True:
        swing_dur, swing_id = get_swing_durations(swing_dur, swing_id, [3, 1, 4, 2], 0.2, 10.0)

        # print('Loop: ', ': ', swing_dur)
        print('gait pattern: ', swing_id)
