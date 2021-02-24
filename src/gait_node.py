#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import WrenchStamped
import numpy as np
import math
from centauro_contact_detection.msg import contacts as contacts_msg
from gait import Gait

# radius of centauro wheels
R = 0.078


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

    rospy.init_node('casannis', anonymous=True)

    # map feet to a string for publishing to the corresponding topic
    id_name = ['FL', 'FR', 'HL', 'HR']
    id_contact_name = ['f_left', 'f_right', 'h_left', 'h_right']

    # accept one message for com and feet initial position
    com_init = rospy.wait_for_message("/cartesian/com/current_reference", PoseStamped, timeout=None)

    f_init = []     # position of wheel frames
    f_cont = []     # position of contact frames

    # loop for all feet
    for i in range(len(id_name)):
        f_init.append(rospy.wait_for_message("/cartesian/" + id_name[i] + "_wheel/current_reference", PoseStamped, timeout=None))
        f_cont.append([f_init[i].pose.position.x, f_init[i].pose.position.y, f_init[i].pose.position.z - R])

    # contact points as array
    contacts = [np.array(x) for x in f_cont]

    # state vector
    c0 = np.array([com_init.pose.position.x, com_init.pose.position.y, com_init.pose.position.z])
    dc0 = np.zeros(3)
    ddc0 = np.zeros(3)
    x0 = np.hstack([c0, dc0, ddc0])

    # Get ROS Parameters

    # ID of the foot to be moved, get from parameters
    swing_id = rospy.get_param("~sw_id")    # from command line as swing_id:=1/2/3/4
    swing_id = swing_id.rstrip(']').lstrip('[').split(',')  # convert swing_id from "[a, b]" to [a,b]
    swing_id = [int(i) for i in swing_id]

    # number of steps
    step_num = len(swing_id)

    # Target position of the foot wrt to the current position
    tgt_dx = rospy.get_param("~tgt_dx")  # get from command line as target_dx
    tgt_dy = rospy.get_param("~tgt_dy")
    tgt_dz = rospy.get_param("~tgt_dz")

    # Clearance to be achieved, counted from the highest point
    swing_clear = rospy.get_param("~clear")  # get from command line as target_dx

    # Swing velocity
    swing_vel = rospy.get_param("~sw_vel")

    # apply or no contact detection
    cont_detection = rospy.get_param("~cont_det")  # from command line as contact_det:=True/False

    # variables to loop for swing legs
    swing_tgt = []  # target positions as list
    swing_t = []    # time periods of the swing phases
    f_pub_ = []     # list of publishers for the swing foot
    com_msg = PoseStamped()     # message to be published for com
    f_msg = []                  # list of messages to be published for swing feet
    swing_contacts = []         # contact positions of the swing feet

    for i in range(step_num):
        # targets
        swing_tgt.append([contacts[swing_id[i] - 1][0] + tgt_dx, contacts[swing_id[i] - 1][1] + tgt_dy, contacts[swing_id[i] - 1][2] + tgt_dz])

        # swing phases
        swing_t.append(rospy.get_param("~sw_t" + str(i+1)))  # from command line as swing_t:="[a,b]"
        swing_t[i] = swing_t[i].rstrip(']').lstrip('[').split(',')  # convert swing_t from "[a, b]" to [a,b]
        swing_t[i] = [float(i) for i in swing_t[i]]

        # swing feet trj publishers
        f_pub_.append(rospy.Publisher('/cartesian/' + id_name[swing_id[i] - 1] + '_wheel/reference', PoseStamped, queue_size=10))

        # feet trj messages
        f_msg.append(PoseStamped())

        # keep same orientation
        f_msg[i].pose.orientation = f_init[swing_id[i] - 1].pose.orientation

        swing_contacts.append(contacts[swing_id[i] - 1])

    # CoM trj publisher
    com_pub_ = rospy.Publisher('/cartesian/com/reference', PoseStamped, queue_size=10)

    # Subscriber for contact flags
    rospy.Subscriber('/contacts', contacts_msg, contacts_callback)

    # object class of the optimization problem
    walk = Gait(mass=90, N=int((swing_t[-1][1] + 1.0) / 0.1), dt=0.1)

    # call the solver of the optimization problem
    # sol is the directory returned by solve class function contains state, forces, control values
    sol = walk.solve(x0=x0, contacts=contacts, swing_id=[x-1 for x in swing_id], swing_tgt=swing_tgt, swing_t=swing_t, min_f=100)

    # interpolate the trj, pass solution values and interpolation frequency
    interpl = walk.interpolate(sol, swing_contacts, swing_tgt, swing_clear, swing_t, int_freq)

    # All points to be published
    N_total = int(walk._N * walk._dt * int_freq)  # total points --> total time * interpl. frequency

    # default value for executed trj points
    executed_trj = N_total - 1

    # early contact detection default value
    early_contact = False

    # time activating contact detection
    t_early = 0.5 * (swing_t[0][0] + swing_t[0][1])

    # trj points during all swing phases
    N_swing_total = int(int_freq * sum([swing_t[i][1] - swing_t[i][0] for i in range(step_num)]))

    # approximate distance covered during swing
    tgt_ds = sum([interpl['sw'][i]['s'] for i in range(step_num)])

    # publish freq wrt the desired swing velocity
    freq = swing_vel * N_swing_total / tgt_ds

    rate = rospy.Rate(freq)  # Frequency trj publishing
    # loop interpolation points to publish on a specified frequency
    for counter in range(N_total):

        if not rospy.is_shutdown():

            # com trajectory
            com_msg.pose.position.x = interpl['x'][0][counter]
            com_msg.pose.position.y = interpl['x'][1][counter]
            com_msg.pose.position.z = interpl['x'][2][counter]

            # swing feet
            for i in range(step_num):
                f_msg[i].pose.position.x = interpl['sw'][i]['x'][counter]
                f_msg[i].pose.position.y = interpl['sw'][i]['y'][counter]
                # add radius as origin of the wheel frame is in the center
                f_msg[i].pose.position.z = interpl['sw'][i]['z'][counter] + R

                # publish swing trajectory
                f_msg[i].header.stamp = rospy.Time.now()
                f_pub_[i].publish(f_msg[i])

            # publish com trajectory regardless contact detection
            com_msg.header.stamp = rospy.Time.now()
            com_pub_.publish(com_msg)

        rate.sleep()

    # print the trajectories
    if early_contact:
        print("Early contact detected. Trj Counter is:", executed_trj, "out of total", N_total-1)

    if rospy.get_param("~plots"):
        walk.print_trj(interpl, int_freq, freq, executed_trj)


if __name__ == '__main__':

    # desired interpolation frequency
    interpolation_freq = 300

    try:
        casannis(interpolation_freq)
    except rospy.ROSInterruptException:
        pass