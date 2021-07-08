#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from step_with_payload import Walking
import numpy as np
from centauro_contact_detection.msg import Contacts as contacts_msg

# radius of centauro wheels/ must set to zero if control the contact point position
R = 0.078
task_name_contact = ['FL_wheel', 'FR_wheel', 'HL_wheel', 'HR_wheel']
#task_name_contact = ["contact1", "contact2", "contact3", "contact4"]  # FL_wheel
#R = 0.0

task_name_moving_contact = ['left_hand', 'right_hand']


def contacts_callback(msg):

    # pass to global scope
    global sw_contact_msg
    sw_contact_msg = msg


def casannis(int_freq):

    """
    This function call the optimization problem constructor, the solver, the interpolator and interfaces with cartesio
    through ros topics
    Args:
        int_freq: desired interpolation frequency (affects only the interpolation and not the optimal solution)
    Returns:
        publish the desired data to cartesio through ros topics

    """

    # accept one message for com and feet initial position
    com_init = rospy.wait_for_message("/cartesian/com/current_reference", PoseStamped, timeout=None)
    fl_init = rospy.wait_for_message("/cartesian/" + task_name_contact[0] + "/current_reference", PoseStamped, timeout=None)
    fr_init = rospy.wait_for_message("/cartesian/" + task_name_contact[1] + "/current_reference", PoseStamped, timeout=None)
    hl_init = rospy.wait_for_message("/cartesian/" + task_name_contact[2] + "/current_reference", PoseStamped, timeout=None)
    hr_init = rospy.wait_for_message("/cartesian/" + task_name_contact[3] + "/current_reference", PoseStamped, timeout=None)

    # hands
    lhand_init = rospy.wait_for_message("/cartesian/left_hand/current_reference", PoseStamped, timeout=None)
    rhand_init = rospy.wait_for_message("/cartesian/right_hand/current_reference", PoseStamped, timeout=None)

    # all current feet info in a list to be used after selecting the swing leg
    f_init = [fl_init, fr_init, hl_init, hr_init]

    # hands
    h_init = [lhand_init, rhand_init]

    # define contacts, take into account the radius of the wheels
    fl_cont = [fl_init.pose.position.x, fl_init.pose.position.y, fl_init.pose.position.z - R]
    fr_cont = [fr_init.pose.position.x, fr_init.pose.position.y, fr_init.pose.position.z - R]
    hl_cont = [hl_init.pose.position.x, hl_init.pose.position.y, hl_init.pose.position.z - R]
    hr_cont = [hr_init.pose.position.x, hr_init.pose.position.y, hr_init.pose.position.z - R]

    # hands
    lh_mov = [lhand_init.pose.position.x, lhand_init.pose.position.y, lhand_init.pose.position.z]
    rh_mov = [rhand_init.pose.position.x, rhand_init.pose.position.y, rhand_init.pose.position.z]

    contacts = [np.array(fl_cont), np.array(fr_cont), np.array(hl_cont), np.array(hr_cont)]

    # hands
    moving_contact = [[np.array(lh_mov), np.zeros(3)], [np.array(rh_mov), np.zeros(3)]]

    # state vector
    c0 = np.array([com_init.pose.position.x, com_init.pose.position.y, com_init.pose.position.z])
    dc0 = np.zeros(3)
    ddc0 = np.zeros(3)
    x0 = np.hstack([c0, dc0, ddc0])

    # Get ROS Parameters

    # ID of the foot to be moved, get from parameters
    swing_id = rospy.get_param("~sw_id")    # from command line as swing_id:=1/2/3/4

    # map swing id to a string for publishing to the corresponding topic
    id_name = ['FL', 'FR', 'HL', 'HR']
    id_contact_name = ['f_left', 'f_right', 'h_left', 'h_right']

    # Target position of the foot wrt to the current position
    tgt_dx = rospy.get_param("~tgt_dx")
    tgt_dy = rospy.get_param("~tgt_dy")
    tgt_dz = rospy.get_param("~tgt_dz")

    # Clearance to be achieved, counted from the highest point
    swing_clear = rospy.get_param("~clear")

    # force threshold
    minimum_force = rospy.get_param("~min_for")

    # target position as array
    swing_tgt = np.array([contacts[swing_id - 1][0] + tgt_dx, contacts[swing_id - 1][1] + tgt_dy, contacts[swing_id - 1][2] + tgt_dz])

    # time period of the swing phase
    swing_t = rospy.get_param("~sw_t")  # from command line as swing_t:="[a,b]"
    swing_t = swing_t.rstrip(']').lstrip('[').split(',')    # convert swing_t from "[a, b]" to [a,b]
    swing_t = [float(i) for i in swing_t]

    # apply or no contact detection
    cont_detection = rospy.get_param("~cont_det")  # from command line as contact_det:=True/False

    # Publishers for the swing foot, com in the cartesian space
    # f_pub_ = rospy.Publisher('/cartesian/' + id_name[swing_id-1] + '_wheel/reference', PoseStamped, queue_size=10)
    f_pub_ = rospy.Publisher('/cartesian/' + task_name_contact[swing_id-1] + '/reference', PoseStamped, queue_size=10)
    com_pub_ = rospy.Publisher('/cartesian/com/reference', PoseStamped, queue_size=10)
    left_h_pub_ = rospy.Publisher('/cartesian/' + task_name_moving_contact[0] + '/reference', PoseStamped, queue_size=10)
    right_h_pub_ = rospy.Publisher('/cartesian/' + task_name_moving_contact[1] + '/reference', PoseStamped, queue_size=10)

    # Subscriber for contact flags
    rospy.Subscriber('/contacts', contacts_msg, contacts_callback)

    # Messages to be published for com and swing foot
    com_msg = PoseStamped()
    f_msg = PoseStamped()
    lh_msg = PoseStamped()      # hands
    rh_msg = PoseStamped()
    # keep the same orientation of the swinging foot
    f_msg.pose.orientation = f_init[swing_id-1].pose.orientation
    lh_msg.pose.orientation = lhand_init.pose.orientation       # hands
    rh_msg.pose.orientation = rhand_init.pose.orientation

    # Construct the class the optimization problem
    walk = Walking(mass=95, N=40, dt=0.2)

    # call the solver of the optimization problem
    # sol is the directory returned by solve class function contains state, forces, control values
    sol = walk.solve(x0=x0, contacts=contacts, mov_contact_initial=moving_contact, swing_id=swing_id-1, swing_tgt=swing_tgt,
                     swing_clearance=swing_clear, swing_t=swing_t, min_f=minimum_force)

    # interpolate the values, pass solution values and interpolation freq. (= publish freq.)
    interpl = walk.interpolate(sol, contacts[swing_id-1], swing_tgt, swing_clear, swing_t, int_freq)

    # All points to be published
    N_total = int(walk._N * walk._dt * int_freq)  # total points --> total time * interpl. frequency
    executed_trj = N_total - 1

    # contact detection
    early_contact = False

    # time starting contact detection
    t_early = swing_t[0] + 0.7 * (swing_t[1] - swing_t[0])

    # trj points during swing phase
    #N_swing_total = int((swing_t[1] - swing_t[0]) * int_freq)

    # approximate distance covered during swing
    tgt_ds = interpl['sw']['s']

    # mean velocity of the swing foot
    mean_foot_velocity = tgt_ds / (swing_t[1] - swing_t[0])
    print('Mean foot velocity is:', mean_foot_velocity, 'm/sec')

    rate = rospy.Rate(int_freq)  # Frequency trj publishing
    # loop interpolation points to publish on a specified frequency
    for counter in range(N_total):

        if not rospy.is_shutdown():

            # com trajectory
            com_msg.pose.position.x = interpl['x'][0][counter]
            com_msg.pose.position.y = interpl['x'][1][counter]
            com_msg.pose.position.z = interpl['x'][2][counter]

            # hands trajectory
            lh_msg.pose.position.x = interpl['p_mov_l'][0][counter]
            lh_msg.pose.position.y = interpl['p_mov_l'][1][counter]
            lh_msg.pose.position.z = interpl['p_mov_l'][2][counter]

            rh_msg.pose.position.x = interpl['p_mov_r'][0][counter]
            rh_msg.pose.position.y = interpl['p_mov_r'][1][counter]# - 0.3
            rh_msg.pose.position.z = interpl['p_mov_r'][2][counter]

            # swing foot trajectory
            f_msg.pose.position.x = interpl['sw']['x'][counter]
            f_msg.pose.position.y = interpl['sw']['y'][counter]
            # add radius as origin of the wheel frame is in the center
            f_msg.pose.position.z = interpl['sw']['z'][counter] + R

            # publish com trajectory regardless contact detection
            com_msg.header.stamp = rospy.Time.now()
            com_pub_.publish(com_msg)

            # publish hands trajectory regardless contact detection
            lh_msg.header.stamp = rospy.Time.now()
            left_h_pub_.publish(lh_msg)

            rh_msg.header.stamp = rospy.Time.now()
            right_h_pub_.publish(rh_msg)

            # whole or part of the trj is published regardless contact detection
            if not cont_detection or interpl['t'][counter] <= t_early:

                # publish swing trajectory
                f_msg.header.stamp = rospy.Time.now()
                f_pub_.publish(f_msg)

            # If no early contact detected
            elif not early_contact:

                # if there is contact
                if getattr(getattr(sw_contact_msg, id_contact_name[swing_id-1]), 'data'):

                    early_contact = True  # stop swing trajectory

                    executed_trj = counter

                # if no contact
                else:

                    # publish swing trajectory
                    f_msg.header.stamp = rospy.Time.now()
                    f_pub_.publish(f_msg)

        rate.sleep()

    # print the trajectories
    if early_contact:
        print("Early contact detected. Trj Counter is:", executed_trj, "out of total", N_total-1)

    if rospy.get_param("~plots"):
        walk.print_trj(sol, interpl, int_freq, contacts, swing_id-1, executed_trj)


if __name__ == '__main__':

    rospy.init_node('casannis_step', anonymous=True)

    # desired interpolation frequency
    interpolation_freq = 300

    try:
        casannis(interpolation_freq)
    except rospy.ROSInterruptException:
        pass
