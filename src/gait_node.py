#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import WrenchStamped
from walking import Walking
import numpy as np
import math
from centauro_contact_detection.msg import contacts as contacts_msg
from gait import Gait

# radius of centauro wheels
R = 0.078


def contacts_callback (msg):

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
    swing_id = swing_id.rstrip(']').lstrip('[').split(',')  # convert swing_id from "[a, b]" to [a,b]
    swing_id = [int(i) for i in swing_id]

    # map swing id to a string for publishing to the corresponding topic
    id_name = ['FL', 'FR', 'HL', 'HR']
    id_contact_name = ['f_left', 'f_right', 'h_left', 'h_right']

    # Target position of the foot wrt to the current position
    tgt_dx = rospy.get_param("~tgt_dx")  # get from command line as target_dx
    tgt_dy = rospy.get_param("~tgt_dy")
    tgt_dz = rospy.get_param("~tgt_dz")

    # Clearance to be achieved, counted from the highest point
    swing_clear = rospy.get_param("~clear")  # get from command line as target_dx

    # Swing velocity
    swing_vel = rospy.get_param("~sw_vel")

    # approximate distance covered during swing
    tgt_ds = len(swing_id) * math.sqrt(tgt_dx**2 + tgt_dy**2 + (tgt_dy + 2*0.1)**2)

    # target position as array
    swing_tgt = np.array([[contacts[swing_id[0] - 1][0] + tgt_dx, contacts[swing_id[0] - 1][1] + tgt_dy, contacts[swing_id[0] - 1][2] + tgt_dz],\
                          [contacts[swing_id[1] - 1][0] + tgt_dx, contacts[swing_id[1] - 1][1] + tgt_dy, contacts[swing_id[1] - 1][2] + tgt_dz],\
                          [contacts[swing_id[2] - 1][0] + tgt_dx, contacts[swing_id[2] - 1][1] + tgt_dy, contacts[swing_id[2] - 1][2] + tgt_dz],\
                          [contacts[swing_id[3] - 1][0] + tgt_dx, contacts[swing_id[3] - 1][1] + tgt_dy, contacts[swing_id[3] - 1][2] + tgt_dz]])

    # time period of the swing phase
    swing_t1 = rospy.get_param("~sw_t1")  # from command line as swing_t:="[a,b]"
    swing_t1 = swing_t1.rstrip(']').lstrip('[').split(',')    # convert swing_t from "[a, b]" to [a,b]
    swing_t1 = [float(i) for i in swing_t1]

    swing_t2 = rospy.get_param("~sw_t2")
    swing_t2 = swing_t2.rstrip(']').lstrip('[').split(',')
    swing_t2 = [float(i) for i in swing_t2]

    swing_t3 = rospy.get_param("~sw_t3")
    swing_t3 = swing_t3.rstrip(']').lstrip('[').split(',')
    swing_t3 = [float(i) for i in swing_t3]

    swing_t4 = rospy.get_param("~sw_t4")
    swing_t4 = swing_t4.rstrip(']').lstrip('[').split(',')
    swing_t4 = [float(i) for i in swing_t4]

    swing_t = [swing_t1, swing_t2, swing_t3, swing_t4]

    # apply or no contact detection
    cont_detection = rospy.get_param("~cont_det")  # from command line as contact_det:=True/False

    # Publishers for the swing foot, com in the cartesian space
    f_pub1_ = rospy.Publisher('/cartesian/' + id_name[swing_id[0]-1] + '_wheel/reference', PoseStamped, queue_size=10)
    f_pub2_ = rospy.Publisher('/cartesian/' + id_name[swing_id[1] - 1] + '_wheel/reference', PoseStamped, queue_size=10)
    f_pub3_ = rospy.Publisher('/cartesian/' + id_name[swing_id[2] - 1] + '_wheel/reference', PoseStamped, queue_size=10)
    f_pub4_ = rospy.Publisher('/cartesian/' + id_name[swing_id[3] - 1] + '_wheel/reference', PoseStamped, queue_size=10)
    com_pub_ = rospy.Publisher('/cartesian/com/reference', PoseStamped, queue_size=10)

    # Subscriber for contact flags
    rospy.Subscriber('/contacts', contacts_msg, contacts_callback)

    # Messages to be published for com and swing foot
    com_msg = PoseStamped()
    f_msg1 = PoseStamped()
    f_msg2 = PoseStamped()
    f_msg3 = PoseStamped()
    f_msg4 = PoseStamped()

    # keep the same orientation of the swinging foot
    f_msg1.pose.orientation = f_init[swing_id[0]-1].pose.orientation
    f_msg2.pose.orientation = f_init[swing_id[1] - 1].pose.orientation
    f_msg3.pose.orientation = f_init[swing_id[2] - 1].pose.orientation
    f_msg4.pose.orientation = f_init[swing_id[3] - 1].pose.orientation

    # Construct the class the optimization problem
    walk = Gait(mass=90, N=int(swing_t4[1]/0.1), dt=0.1)

    # call the solver of the optimization problem
    swing_ids = [swing_id[0]-1, swing_id[1]-1, swing_id[2]-1, swing_id[3]-1]
    # sol is the directory returned by solve class function contains state, forces, control values
    sol = walk.solve(x0=x0, contacts=contacts, swing_id=swing_ids, swing_tgt=swing_tgt, swing_t=swing_t, min_f=100)

    # interpolate the values, pass solution values and interpolation freq. (= publish freq.)
    swing_contacts = [contacts[swing_id[0]-1], contacts[swing_id[1]-1], contacts[swing_id[2]-1], contacts[swing_id[3]-1]]
    interpl = walk.interpolate(sol, swing_contacts, swing_tgt, swing_clear, swing_t, int_freq)

    # All points to be published
    N_total = int(walk._N * walk._dt * int_freq)  # total points --> total time * interpl. frequency
    executed_trj = N_total - 1

    # contact detection
    early_contact = False

    # time starting contact detection
    t_early = 0.5 * (swing_t[0][0] + swing_t[0][1])

    # trj points during swing phase
    N_swing_total = int((swing_t[0][1] - swing_t[0][0] + swing_t[1][1] - swing_t[1][0] + swing_t[2][1] - swing_t[2][0] + swing_t[3][1] - swing_t[3][0]) * int_freq)

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

            # swing foot trajectory 1
            f_msg1.pose.position.x = interpl['sw'][0][0][counter]
            f_msg1.pose.position.y = interpl['sw'][0][1][counter]
            # add radius as origin of the wheel frame is in the center
            f_msg1.pose.position.z = interpl['sw'][0][2][counter] + R

            # swing foot trajectory 2
            f_msg2.pose.position.x = interpl['sw'][1][0][counter]
            f_msg2.pose.position.y = interpl['sw'][1][1][counter]
            # add radius as origin of the wheel frame is in the center
            f_msg2.pose.position.z = interpl['sw'][1][2][counter] + R

            # swing foot trajectory 3
            f_msg3.pose.position.x = interpl['sw'][2][0][counter]
            f_msg3.pose.position.y = interpl['sw'][2][1][counter]
            # add radius as origin of the wheel frame is in the center
            f_msg3.pose.position.z = interpl['sw'][2][2][counter] + R

            # swing foot trajectory 4
            f_msg4.pose.position.x = interpl['sw'][3][0][counter]
            f_msg4.pose.position.y = interpl['sw'][3][1][counter]
            # add radius as origin of the wheel frame is in the center
            f_msg4.pose.position.z = interpl['sw'][3][2][counter] + R

            # publish com trajectory regardless contact detection
            com_msg.header.stamp = rospy.Time.now()
            com_pub_.publish(com_msg)

            # publish swing trajectory
            f_msg1.header.stamp = rospy.Time.now()
            f_pub1_.publish(f_msg1)

            f_msg2.header.stamp = rospy.Time.now()
            f_pub2_.publish(f_msg2)

            f_msg3.header.stamp = rospy.Time.now()
            f_pub3_.publish(f_msg3)

            f_msg4.header.stamp = rospy.Time.now()
            f_pub4_.publish(f_msg4)

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