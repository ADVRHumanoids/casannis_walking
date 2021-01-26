#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from walking import Walking
import numpy as np
from swing_leg_trj import swing_leg
from matplotlib import pyplot as plt


def casannis(pub_freq):

    """
    This function call the optimization problem constructor, the solver, the interpolators and interfaces with cartesio
    through ros topics
    Args:
        pub_freq: desired publish frequency

    Returns:
        publish the desired data to cartesio through ros topics

    """

    # Construct the class the optimization problem
    walk = Walking(90.0, 30, 0.1)

    # radius of centauro wheels
    r = 0.078

    # Publishers for com and foot for cartesio
    com_pub_ = rospy.Publisher('/cartesian/com/reference', PoseStamped, queue_size=10)
    fl_pub_ = rospy.Publisher('/cartesian/FL_wheel/reference', PoseStamped, queue_size=10)

    rospy.init_node('casannis', anonymous=True)

    # accept one message for com and feet initial position
    com_init = rospy.wait_for_message("/cartesian/com/current_reference", PoseStamped, timeout=None)
    fl_init = rospy.wait_for_message("/cartesian/FL_wheel/current_reference", PoseStamped, timeout=None)
    fr_init = rospy.wait_for_message("/cartesian/FR_wheel/current_reference", PoseStamped, timeout=None)
    hl_init = rospy.wait_for_message("/cartesian/HL_wheel/current_reference", PoseStamped, timeout=None)
    hr_init = rospy.wait_for_message("/cartesian/HR_wheel/current_reference", PoseStamped, timeout=None)

    # define contacts, take into account the radius of the wheels
    fl_cont = [fl_init.pose.position.x, fl_init.pose.position.y, fl_init.pose.position.z - r]
    fr_cont = [fr_init.pose.position.x, fr_init.pose.position.y, fr_init.pose.position.z - r]
    hl_cont = [hl_init.pose.position.x, hl_init.pose.position.y, hl_init.pose.position.z - r]
    hr_cont = [hr_init.pose.position.x, hr_init.pose.position.y, hr_init.pose.position.z - r]

    contacts = [np.array(fl_cont), np.array(fr_cont), np.array(hl_cont), np.array(hr_cont)]

    # state vector
    c0 = np.array([com_init.pose.position.x, com_init.pose.position.y, com_init.pose.position.z])
    dc0 = np.zeros(3)
    ddc0 = np.zeros(3)
    x0 = np.hstack([c0, dc0, ddc0])

    # ID of the foot to be moved, get from parameters
    swing_id = 0# rospy.get_param("~swing_id") # get from command line
    print("Swing id received:", swing_id)

    # Target position of the foot to be moved wrt to the current position
    tgt_dx = 0.1#rospy.get_param("~tgt_dx") # get from command line
    tgt_dy = 0#rospy.get_param("~tgt_dy")
    tgt_dz = 0.1#rospy.get_param("~tgt_dz")
    swing_tgt = np.array([contacts[0][0] + tgt_dx, contacts[0][1] + tgt_dy, contacts[0][2] + tgt_dz])

    # time period of the swing phase ?get from parameters
    swing_t = (0.5, 2.5)

    # call the solver of the optimization problem
    # sol is the directory returned by solve class function contains state, forces, control values
    sol = walk.solve(x0=x0, contacts=contacts, swing_id=swing_id, swing_tgt=swing_tgt, swing_t=swing_t)

    # interpolate the values, pass solution values and interpolation freq. (= publish freq.)
    interpl = walk.interpolate(sol, pub_freq)

    # All points to be published
    N_total = int(walk._N * walk._dt * pub_freq)  # total points --> total time * frequency

    # interpolation of the swing foot trajectory
    swing_trj = swing_leg(contacts[0], swing_tgt, swing_t, pub_freq)

    # Messages to be published for com and swing foot
    com_msg = PoseStamped()
    fl_msg = PoseStamped()

    # keep the same orientation of the swinging foot
    fl_msg.pose.orientation = fl_init.pose.orientation

    rate = rospy.Rate(pub_freq)  # Frequency of publish process
    # loop interpolation points to publish on a specified frequency
    for counter in range(N_total):

        if not rospy.is_shutdown():

            # com trajectory
            com_msg.pose.position.x = interpl['x'][0][counter]
            com_msg.pose.position.y = interpl['x'][1][counter]
            com_msg.pose.position.z = interpl['x'][2][counter]

            # swing foot trajectory
            fl_msg.pose.position.x = swing_trj['x'][counter]
            fl_msg.pose.position.y = swing_trj['y'][counter]
            # add radius as origin of the wheel frame is in the center
            fl_msg.pose.position.z = swing_trj['z'][counter] + r

            # publish messages and attach time
            fl_msg.header.stamp = rospy.Time.now()
            fl_pub_.publish(fl_msg)

            com_msg.header.stamp = rospy.Time.now()
            com_pub_.publish(com_msg)

        rate.sleep()

    # Interpolated state plot
    '''state_labels = ['CoM Position', 'CoM Velocity', 'CoM Acceleration']
    plt.figure()
    for i, name in enumerate(state_labels):
        plt.subplot(3, 1, i + 1)
        for j in range(walk._dimc):
            plt.plot(interpl['t'], interpl['x'][walk._dimc * i + j], '-')
        plt.grid()
        plt.legend(['x', 'y', 'z'])
        plt.title(name)
    plt.xlabel('Time [s]')
    feet_labels = ['front left', 'front right', 'hind right', 'hind left']
    # Interpolated force plot
    plt.figure()
    for i, name in enumerate(feet_labels):
        plt.subplot(2, 2, i + 1)
        for k in range(3):
            plt.plot(walk._time, interpl['f'][3 * i + k][0](walk._time), '-')
        plt.grid()
        plt.title(name)
        plt.legend([str(name) + '_x', str(name) + '_y', str(name) + '_z'])
    plt.xlabel('Time [s]')
    plt.show()'''

    # plot swing trajectory
    s = np.linspace(0, walk._dt * walk._N, N_total)
    coord_labels = ['x', 'y', 'z']
    plt.figure()
    for i, name in enumerate(coord_labels):
        plt.subplot(3, 1, i + 1)
        plt.plot(s, swing_trj[name])
        plt.grid()
        plt.title('Trajectory ' + name)
    plt.xlabel('Time [s]')
    plt.show()


if __name__ == '__main__':

    # desired publish frequency
    freq = 150

    try:
        casannis(freq)
    except rospy.ROSInterruptException:
        pass

