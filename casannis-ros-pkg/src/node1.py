#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from walking import Walking
import numpy as np
from swing_leg_trj import swing_leg
from matplotlib import pyplot as plt

def casannis(pub_freq):
    '''

    Args:
        int_swing: interpolated swing trajectory for the foot to be lifted
        int_data: interpolated data to be published,
        that is a dictionary with time, state vector and forces
        pub_freq: desired publish frequency

    Returns:
        published the desired data to cartesio ros topics
    '''

    walk = Walking(90.0, 30, 0.1)
    r = 0.078

    # Publishers
    com_pub_ = rospy.Publisher('/cartesian/com/reference', PoseStamped, queue_size=10)
    fl_pub_ = rospy.Publisher('/cartesian/FL_wheel/reference', PoseStamped, queue_size=10)

    rospy.init_node('casannis', anonymous=True)

    # accept one message for com and feet initial position
    com_init = rospy.wait_for_message("/cartesian/com/current_reference", PoseStamped, timeout=None)
    fl_init = rospy.wait_for_message("/cartesian/FL_wheel/current_reference", PoseStamped, timeout=None)
    fr_init = rospy.wait_for_message("/cartesian/FR_wheel/current_reference", PoseStamped, timeout=None)
    hl_init = rospy.wait_for_message("/cartesian/HL_wheel/current_reference", PoseStamped, timeout=None)
    hr_init = rospy.wait_for_message("/cartesian/HR_wheel/current_reference", PoseStamped, timeout=None)

    # define contacts
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

    # ?get from parameters
    swing_id = 0

    # ?get from parameters
    swing_tgt = np.array([contacts[0][0] + 0.1, contacts[0][1], contacts[0][2]+0.1])
    for i in range(3):
        print(contacts[0][i])
    # ?get from parameters
    swing_t = (0.5, 2.5)

    # sol is the directory returned by solve class function contains state, forces, control values
    sol = walk.solve(x0=x0, contacts=contacts, swing_id=swing_id, swing_tgt=swing_tgt, swing_t=swing_t)

    # interpolate the values, pass solution values and number of interpolation points between knots
    interpl = walk.interpolate(sol, 20)

    dz = 0.1
    swing_trj = swing_leg(contacts[0], swing_tgt, dz, swing_t)

    # Messages
    com_msg = PoseStamped()
    fl_msg = PoseStamped()
    fl_msg.pose.orientation = fl_init.pose.orientation

    counter = 0
    rate = rospy.Rate(pub_freq)  # 100hz
    while not rospy.is_shutdown():

        if counter < 600:
            # com trajectory
            com_msg.pose.position.x = interpl['x'][0][counter]
            com_msg.pose.position.y = interpl['x'][1][counter]
            com_msg.pose.position.z = interpl['x'][2][counter]

            # swing leg trajectory
            fl_msg.pose.position.x = swing_trj['x'](0.5 * counter / pub_freq)
            fl_msg.pose.position.y = swing_trj['y'](0.5 * counter / pub_freq)
            fl_msg.pose.position.z = swing_trj['z'](0.5 * counter / pub_freq) + r # change reference coordinate system

            com_pub_.publish(com_msg)
            fl_pub_.publish(fl_msg)

        else:
            break

        counter = counter + 2
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
    # plot splines
    s = np.linspace(0.5, 2.5, 100)
    coord_labels = ['x', 'y', 'z']
    plt.figure()
    for i, name in enumerate(coord_labels):
        plt.subplot(3, 1, i + 1)
        plt.plot(s, swing_trj[name](s), '-')
        plt.grid()
        plt.title('Trajectory ' + name)
    plt.xlabel('Time [s]')
    plt.show()


if __name__ == '__main__':

    '''# create an instance of the class
    walk = Walking(90.0, 30, 0.1)

    # initial state
    c0 = np.array([0.1, 0.0, 0.64])
    dc0 = np.zeros(3)
    ddc0 = np.zeros(3)
    x0 = np.hstack([c0, dc0, ddc0])

    contacts = [
        np.array([0.35, 0.35, 0.0]),  # fl
        np.array([0.35, -0.35, 0.0]),  # fr
        np.array([-0.35, -0.35, 0.0]),  # hr
        np.array([-0.35, 0.35, 0.0])  # hl
    ]

    swing_id = 0

    swing_tgt = np.array([0.45, 0.35, 0.1])

    swing_t = (1.0, 2.0)

    # sol is the directory returned by solve class function contains state, forces, control values
    sol = walk.solve(x0=x0, contacts=contacts, swing_id=swing_id, swing_tgt=swing_tgt, swing_t=swing_t)

    # interpolate the values, pass solution values and number of interpolation points between knots
    interpl = walk.interpolate(sol, 20)

    dz = 0.1
    dt = 3.0
    swing_trj = swing_leg(contacts[0], swing_tgt, dz, dt)'''

    freq = 100
    #casannis(interpl, swing_trj, freq)
    casannis(freq)
    '''try:
        talker(interpl)
    except rospy.ROSInterruptException:
        pass
    '''
