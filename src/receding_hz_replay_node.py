#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from centauro_contact_detection.msg import Contacts as Contacts_msg
from casannis_walking.msg import PayloadAware_plans as MotionPlan_msg
from casannis_walking.msg import Pa_interpolated_trj as Trj_msg
from std_msgs.msg import Bool

import matplotlib.pyplot as plt

# radius of centauro wheels
R = 0.078
#task_name_contact = ["contact1", "contact2", "contact3", "contact4"]  # FL_wheel
task_name_contact = ['FL_wheel', 'FR_wheel', 'HL_wheel', 'HR_wheel']

task_name_moving_contact = ['left_hand', 'right_hand']
rod_plate_CoG_urdf = [4.8407693e-02,  2.0035723e-02,  7.7533287e-02]    # not used currently


def contacts_callback(msg):

    # pass to global scope
    global sw_contact_msg
    sw_contact_msg = msg


def motion_plan_callback(msg):

    global motion_plan_waiting
    motion_plan_waiting = True

    # pass message to global scope
    global received_motion_plan
    received_motion_plan = {
        'x': msg.state,
        'u': msg.control,
        'Pl_mov': msg.left_arm_pos,
        'Pr_mov': msg.right_arm_pos,
        'DPl_mov': msg.left_arm_vel,
        'DPr_mov': msg.right_arm_vel,
        'F': msg.leg_forces,
        'F_virt_l': msg.left_arm_force,
        'F_virt_r': msg.right_arm_force,
    }
    print('Received message')


def interpolated_trj_callback(msg):

    global trj_waiting
    trj_waiting = True

    # pass message to global scope
    global received_trj
    received_trj = {
        'horizon_shift': msg.horizon_shift,
        'horizon_dur': msg.horizon_dur,
        'id': msg.id,
        't': msg.time,
        'swing_t': msg.swing_t,
        'swing_id': msg.swing_id,
        'com': [msg.com_pos_x, msg.com_pos_y, msg.com_pos_z],
        'p_mov_l': [msg.left_arm_pos_x, msg.left_arm_pos_y, msg.left_arm_pos_z],
        'p_mov_r': [msg.right_arm_pos_x, msg.right_arm_pos_y, msg.right_arm_pos_z],
        'leg_ee': [[msg.fl_leg_pos_x, msg.fl_leg_pos_y, msg.fl_leg_pos_z],
                   [msg.fr_leg_pos_x, msg.fr_leg_pos_y, msg.fr_leg_pos_z],
                   [msg.hl_leg_pos_x, msg.hl_leg_pos_y, msg.hl_leg_pos_z],
                   [msg.hr_leg_pos_x, msg.hr_leg_pos_y, msg.hr_leg_pos_z]]
    }

    print('Received trj message')


def casannis(int_freq):

    """
    This function call the optimization problem constructor, the solver, the interpolator and interfaces with cartesio
    through ros topics
    Args:
        int_freq: desired interpolation frequency (affects only the interpolation and not the optimal solution)
    Returns:
        publish the desired data to cartesio through ros topics

    """

    # # map feet to a string for publishing to the corresponding topic
    id_name = ['FL', 'FR', 'HL', 'HR']
    # id_contact_name = ['f_left', 'f_right', 'h_left', 'h_right']

    f_init = []     # position of wheel frames

    # # loop for all feet
    for i in range(len(id_name)):
        f_init.append(rospy.wait_for_message("/cartesian/" + id_name[i] + "_wheel/current_reference",
                                             PoseStamped,
                                             timeout=None))

    # hands
    lhand_init = rospy.wait_for_message("/cartesian/left_hand/current_reference", PoseStamped, timeout=None)
    rhand_init = rospy.wait_for_message("/cartesian/right_hand/current_reference", PoseStamped, timeout=None)

    f_pub_ = []     # list of publishers for the swing foot
    com_msg = PoseStamped()     # message to be published for com
    f_msg = []                  # list of messages to be published for swing feet

    # Leg ee trj publishers
    for i in range(4):
        f_pub_.append(rospy.Publisher('/cartesian/' + task_name_contact[i] + '/reference',
                                      PoseStamped,
                                      queue_size=10))
        # feet trj messages
        f_msg.append(PoseStamped())

        # keep same orientation
        f_msg[i].pose.orientation = f_init[i].pose.orientation

    # CoM trj publisher
    com_pub_ = rospy.Publisher('/cartesian/com/reference', PoseStamped, queue_size=10)

    # hands' publishers and msgs
    left_h_pub_ = rospy.Publisher('/cartesian/' + task_name_moving_contact[0] + '/reference', PoseStamped, queue_size=10)
    right_h_pub_ = rospy.Publisher('/cartesian/' + task_name_moving_contact[1] + '/reference', PoseStamped, queue_size=10)
    lh_msg = PoseStamped()
    rh_msg = PoseStamped()
    lh_msg.pose.orientation = lhand_init.pose.orientation  # hands
    rh_msg.pose.orientation = rhand_init.pose.orientation

    # Subscriber for contact flags
    rospy.Subscriber('/contacts', Contacts_msg, contacts_callback)

    # Subscriber for motion plans
    rospy.Subscriber('/PayloadAware/motion_plan', MotionPlan_msg, motion_plan_callback)
    rospy.Subscriber('/PayloadAware/interpolated_trj', Trj_msg, interpolated_trj_callback)

    # wait until planners sends a message that is connected
    planner_connection = rospy.wait_for_message("/PayloadAware/connection", Bool, timeout=None)

    # Number of points to be published
    horizon_shift = received_trj['horizon_shift']
    horizon_dur = received_trj['horizon_dur']
    N_shift = int(horizon_shift * int_freq)  # points --> time * interpolation frequency
    N_horizon = int(horizon_dur * int_freq)

    # debug
    # print('horizon_shift', horizon_shift)
    # print('kkkkkkkk', N_horizon)
    # horizon_completance_counter = 0
    plans_counter = 0
    while True:
        print('.................................')      # debug
        print('.................................')
        print('.........plan n. ', received_trj['id'])

        swing_id = received_trj['swing_id']
        step_num = len(received_trj['swing_id'])

        # convert to list of lists
        half_list_size = int(len(received_trj['swing_t']) / 2)  # half size of the flat list
        swing_t = [[received_trj['swing_t'][2 * a], received_trj['swing_t'][2 * a + 1]] for a in range(half_list_size)]
        # print('Swing t: ', swing_t)
        # print('Swing id: ', swing_id)

        # plt.figure()  # debug
        # for la in range(step_num):
        #     for si in range(2, 3):
        #         plt.plot(received_trj['leg_ee'][la][si])
        # plt.show()

        rate = rospy.Rate(int_freq)  # Frequency trj publishing
        # loop interpolation points to publish on a specified frequency
        for counter in range(N_shift):

            if not rospy.is_shutdown():

                horizon_trj_point = plans_counter * N_shift + counter

                # check if current time is within swing phase and contact detection
                for i in range(step_num):

                    # swing phase check
                    if swing_t[i][0] <= received_trj['t'][counter] <= swing_t[i][1]:
                        swing_phase = i
                        break

                    else:
                        swing_phase = -1    # not in swing phase

                # com trajectory
                com_msg.pose.position.x = received_trj['com'][0][counter]   #interpl['x'][0][counter]
                com_msg.pose.position.y = received_trj['com'][1][counter]
                com_msg.pose.position.z = received_trj['com'][2][counter]

                # hands trajectory
                lh_msg.pose.position.x = received_trj['p_mov_l'][0][counter]    #interpl['p_mov_l'][0][counter]
                lh_msg.pose.position.y = received_trj['p_mov_l'][1][counter]
                lh_msg.pose.position.z = received_trj['p_mov_l'][2][counter]

                rh_msg.pose.position.x = received_trj['p_mov_r'][0][counter]
                rh_msg.pose.position.y = received_trj['p_mov_r'][1][counter]
                rh_msg.pose.position.z = received_trj['p_mov_r'][2][counter]

                # swing foot
                current_sw_leg_id = swing_id[swing_phase]
                print('.....', horizon_trj_point)
                f_msg[current_sw_leg_id].pose.position.x = \
                    received_trj['leg_ee'][current_sw_leg_id][0][horizon_trj_point]
                f_msg[current_sw_leg_id].pose.position.y = \
                    received_trj['leg_ee'][current_sw_leg_id][1][horizon_trj_point]
                f_msg[current_sw_leg_id].pose.position.z =\
                    received_trj['leg_ee'][current_sw_leg_id][2][horizon_trj_point] + R   # add radius

                # publish com trajectory regardless contact detection
                com_msg.header.stamp = rospy.Time.now()
                com_pub_.publish(com_msg)

                # publish hands trajectory regardless contact detection
                lh_msg.header.stamp = rospy.Time.now()
                left_h_pub_.publish(lh_msg)

                rh_msg.header.stamp = rospy.Time.now()
                right_h_pub_.publish(rh_msg)

                # if swing_phase == -1:
                #     pass
                #
                # # swing phase
                # else:
                f_msg[current_sw_leg_id].header.stamp = rospy.Time.now()
                f_pub_[current_sw_leg_id].publish(f_msg[current_sw_leg_id])

                if horizon_trj_point == N_horizon - 1:
                    plans_counter = 0

                # if horizon_completance_counter == N_horizon - 1:
                #     horizon_completance_counter = 0
                # else:
                #     horizon_completance_counter += 1
                rate.sleep()
        plans_counter += 1

    # print the trajectories
    # try:
    #     # there was early contact detected
    #     if early_contact.index(True) + 1:
    #         print("Early contact detected. Trj Counter is:", executed_trj, "out of total", N_total-1)
    #
    #         if rospy.get_param(planner_node_prefix + "/plots"):
    #             walk.print_trj(sol, interpl, int_freq, contacts, [x-1 for x in swing_id], executed_trj)
    # except:
    #     print("No early contact detected")
    #
    #     if rospy.get_param(planner_node_prefix + "/plots"):
    #         walk.print_trj(sol, interpl, int_freq, contacts, [x-1 for x in swing_id], [N_total-1, N_total-1, N_total-1, N_total-1])


if __name__ == '__main__':

    # desired interpolation frequency
    interpolation_freq = 300

    rospy.init_node('casannis_replay', anonymous=True)

    try:
        casannis(interpolation_freq)
    except rospy.ROSInterruptException:
        pass