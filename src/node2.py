#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import WrenchStamped
from walking import Walking
import numpy as np
from centauro_contact_detection.msg import contacts as contacts_msg
from std_msgs.msg import Bool

# radius of centauro wheels
R = 0.078


def casannis(pub_freq):

    """
    This function call the optimization problem constructor, the solver, the interpolator and interfaces with cartesio
    through ros topics
    Args:
        pub_freq: desired publish frequency (affects only the interpolation and not the optimal solution)

    Returns:
        publish the desired data to cartesio through ros topics

    """

    rospy.init_node('casannis', anonymous=True)

    # accept one message for com and feet initial position
    com_init = rospy.wait_for_message("/cartesian/com/current_reference", PoseStamped, timeout=None)
    fl_init = rospy.wait_for_message("/cartesian/FL_wheel/current_reference", PoseStamped, timeout=None)
    fr_init = rospy.wait_for_message("/cartesian/FR_wheel/current_reference", PoseStamped, timeout=None)
    hl_init = rospy.wait_for_message("/cartesian/HL_wheel/current_reference", PoseStamped, timeout=None)
    hr_init = rospy.wait_for_message("/cartesian/HR_wheel/current_reference", PoseStamped, timeout=None)

    # all current feet info in a list to be used after selecting the swing leg
    f_init = [fl_init, fr_init, hl_init, hr_init]

    # define contacts, take into account the radius of the wheels
    fl_cont = [fl_init.pose.position.x, fl_init.pose.position.y, fl_init.pose.position.z - R]
    fr_cont = [fr_init.pose.position.x, fr_init.pose.position.y, fr_init.pose.position.z - R]
    hl_cont = [hl_init.pose.position.x, hl_init.pose.position.y, hl_init.pose.position.z - R]
    hr_cont = [hr_init.pose.position.x, hr_init.pose.position.y, hr_init.pose.position.z - R]

    contacts = [np.array(fl_cont), np.array(fr_cont), np.array(hl_cont), np.array(hr_cont)]

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
    tgt_dx = rospy.get_param("~tgt_dx")  # get from command line as target_dx
    tgt_dy = rospy.get_param("~tgt_dy")
    tgt_dz = rospy.get_param("~tgt_dz")

    # target position as array
    swing_tgt = np.array([contacts[swing_id - 1][0] + tgt_dx, contacts[swing_id - 1][1] + tgt_dy, contacts[swing_id - 1][2] + tgt_dz])

    # time period of the swing phase
    swing_t = rospy.get_param("~sw_t")  # from command line as swing_t:="[a,b]"
    swing_t = swing_t.rstrip(']').lstrip('[').split(',')    # convert swing_t from "[a, b]" to [a,b]
    swing_t = [float(i) for i in swing_t]

    # apply or no contact detection
    cont_detection = rospy.get_param("~cont_det")  # from command line as contact_det:=True/False

    # Publishers for the swing foot, com in the cartesian space and contact flag
    f_pub_ = rospy.Publisher('/cartesian/' + id_name[swing_id-1] + '_wheel/reference', PoseStamped, queue_size=10)
    com_pub_ = rospy.Publisher('/cartesian/com/reference', PoseStamped, queue_size=10)

    # Messages to be published for com and swing foot
    com_msg = PoseStamped()
    f_msg = PoseStamped()

    # keep the same orientation of the swinging foot
    f_msg.pose.orientation = f_init[swing_id-1].pose.orientation

    # Construct the class the optimization problem
    walk = Walking(mass=90, N=50, dt=0.1)

    # call the solver of the optimization problem
    # sol is the directory returned by solve class function contains state, forces, control values
    sol = walk.solve(x0=x0, contacts=contacts, swing_id=swing_id-1, swing_tgt=swing_tgt, swing_t=swing_t, min_f=100)

    # debug
    '''print("X0 is:", x0)
    print("contacts is:", contacts)
    print("swing id is:", swing_id-1)
    print("swing target is:", swing_tgt)
    print("swing time:", swing_t)'''

    # interpolate the values, pass solution values and interpolation freq. (= publish freq.)
    interpl = walk.interpolate(sol, contacts[swing_id-1], swing_tgt, swing_t, pub_freq)

    # All points to be published
    N_total = int(walk._N * walk._dt * pub_freq)  # total points --> total time * frequency

    # contact detection
    i = 0   # counter i for contact detection
    early_contact = False

    # time starting contact detection
    t_early = 0.5 * (swing_t[0] + swing_t[1])



    rate = rospy.Rate(pub_freq)  # Frequency trj publishing
    # loop interpolation points to publish on a specified frequency
    for counter in range(N_total):

        if not rospy.is_shutdown():

            # com trajectory
            com_msg.pose.position.x = interpl['x'][0][counter]
            com_msg.pose.position.y = interpl['x'][1][counter]
            com_msg.pose.position.z = interpl['x'][2][counter]

            # swing foot trajectory
            f_msg.pose.position.x = interpl['sw'][0][counter]
            f_msg.pose.position.y = interpl['sw'][1][counter]
            # add radius as origin of the wheel frame is in the center
            f_msg.pose.position.z = interpl['sw'][2][counter] + R

            # publish com trajectory regardless contact detection
            com_msg.header.stamp = rospy.Time.now()
            com_pub_.publish(com_msg)

            # whole or part of the trj is published regardless contact detection
            if not cont_detection or interpl['t'][counter] <= t_early:

                # publish swing trajectory
                f_msg.header.stamp = rospy.Time.now()
                f_pub_.publish(f_msg)

            # If no early contact detected
            elif not early_contact:

                # receive contact flag of the swing leg
                sw_contact_sub = rospy.wait_for_message("/contacts", contacts_msg)

                # if there is contact
                if getattr(getattr(sw_contact_sub, id_contact_name[swing_id-1]), 'data'):

                    early_contact = True  # stop swing trajectory

                    executed_trj = counter

                # if no contact
                else:

                    # publish swing trajectory
                    f_msg.header.stamp = rospy.Time.now()
                    f_pub_.publish(f_msg)

        rate.sleep()

    # Late contact detection if no early contact detected
    '''if cont_detection and i != window:

        # velocity of the foot
        vel = 0.3

        i = 0

        # check the force on the foot until 5 consecutive threshold violations
        while i < window:
            # receive force in z direction of the swing leg
            fl_force_sub_ = rospy.wait_for_message("/cartesian/force_estimation/contact_" + str(swing_id), WrenchStamped)

            # force threshold to consider as contact
            if fl_force_sub_.wrench.force.z > thres:

                # count the threshold violations
                i = i + 1

            # the 5 threshold violations must be consecutive, otherwise counter is set to zero again
            elif fl_force_sub_.wrench.force.z <= thres and i != 0:
                i = 0

            # publish swing foot trajectory
            f_msg.header.stamp = rospy.Time.now()
            f_msg.pose.position.z = f_msg.pose.position.z - vel/pub_freq
            f_pub_.publish(f_msg)

            rate.sleep()'''

    # print the trajectories
    if not cont_detection:
        executed_trj = counter
    print("Early contact detected. Trj Counter is:", executed_trj, "out of total", N_total)

    if rospy.get_param("~plots"):
        walk.print_trj(interpl, pub_freq, executed_trj)


if __name__ == '__main__':

    # desired publish frequency
    freq = 500

    try:
        casannis(freq)
    except rospy.ROSInterruptException:
        pass