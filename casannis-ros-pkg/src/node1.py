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
    swing_id = rospy.get_param("~swing_id") # get from command line
    print("Swing id received:", swing_id)

    # Target position of the foot to be moved wrt to the current position
    tgt_dx = rospy.get_param("~tgt_dx") # get from command line
    tgt_dy = rospy.get_param("~tgt_dy")
    tgt_dz = rospy.get_param("~tgt_dz")
    swing_tgt = np.array([contacts[0][0] + tgt_dx, contacts[0][1] + tgt_dy, contacts[0][2] + tgt_dz])

    # time period of the swing phase ?get from parameters
    swing_t = (0.5, 2.5)

    # call the solver of the optimization problem
    # sol is the directory returned by solve class function contains state, forces, control values
    sol = walk.solve(x0=x0, contacts=contacts, swing_id=swing_id, swing_tgt=swing_tgt, swing_t=swing_t)

    # interpolate the values, pass solution values and number of interpolation points between knots
    N_int = 20  # interpolation points between knots
    interpl = walk.interpolate(sol, N_int)

    N_total = walk._N * N_int  # total points --> Opt. knots * Int. points

    # height difference that the foot will reach during swing phase wrt the target
    dz = 0.1    # it will go dz m higher than target during swing phase

    # interpolation of the swing foot trajectory
    swing_trj = swing_leg(contacts[0], swing_tgt, dz, swing_t)

    # Messages to be published for com and swing foot
    com_msg = PoseStamped()
    fl_msg = PoseStamped()


    # keep the same orientation of the swinging foot
    fl_msg.pose.orientation = fl_init.pose.orientation

    counter = 0 # counter for describing interpolation points

    rate = rospy.Rate(pub_freq)  # Frequency of publish process

    # step of publishing interpolating points according to publ. frequency
    step = int(N_total / (walk._dt * walk._N * pub_freq))

    # loop interpolation points to publish on a specified frequency
    for counter in range(0, N_total, step):

        if not rospy.is_shutdown():
            # com trajectory
            com_msg.pose.position.x = interpl['x'][0][counter]
            com_msg.pose.position.y = interpl['x'][1][counter]
            com_msg.pose.position.z = interpl['x'][2][counter]

            # tt represent the absolute time of the optimization problem
            tt = counter * 0.005

            # If we are in the swing phase
            if swing_trj['t'][0] <= tt <= swing_trj['t'][5]:

                # Check which splane should be activated according to time
                if swing_trj['t'][0] <= tt <= swing_trj['t'][1]:

                    # swing leg trajectory
                    fl_msg.pose.position.x = swing_trj['x'][0](tt)
                    fl_msg.pose.position.y = swing_trj['y'][0](tt)
                    # add radius as origin of the wheel frame is in the center
                    fl_msg.pose.position.z = swing_trj['z'][0](tt) + r

                elif swing_trj['t'][1] <= tt <= swing_trj['t'][2]:

                    # swing leg trajectory
                    fl_msg.pose.position.x = swing_trj['x'][1](tt)
                    fl_msg.pose.position.y = swing_trj['y'][1](tt)
                    fl_msg.pose.position.z = swing_trj['z'][1](tt) + r  # add radius

                elif swing_trj['t'][2] <= tt <= swing_trj['t'][3]:

                    # swing leg trajectory
                    fl_msg.pose.position.x = swing_trj['x'][2](tt)
                    fl_msg.pose.position.y = swing_trj['y'][2](tt)
                    fl_msg.pose.position.z = swing_trj['z'][2](tt) + r  # add radius

                elif swing_trj['t'][3] <= tt <= swing_trj['t'][4]:

                    # swing leg trajectory
                    fl_msg.pose.position.x = swing_trj['x'][3](tt)
                    fl_msg.pose.position.y = swing_trj['y'][3](tt)
                    fl_msg.pose.position.z = swing_trj['z'][3](tt) + r  # add radius

                elif swing_trj['t'][4] <= tt <= swing_trj['t'][5]:

                    # swing leg trajectory
                    fl_msg.pose.position.x = swing_trj['x'][4](tt)
                    fl_msg.pose.position.y = swing_trj['y'][4](tt)
                    fl_msg.pose.position.z = swing_trj['z'][4](tt) + r  # add radius

                # attach time to message
                fl_msg.header.stamp = rospy.Time.now()

                # publish messages
                fl_pub_.publish(fl_msg)

            # attach time to message
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
    s = np.linspace(0.5, 2.5, 100)
    coord_labels = ['x', 'y', 'z']
    plt.figure()
    for i, name in enumerate(coord_labels):
        plt.subplot(3, 1, i + 1)
        for k in range(len(swing_trj['t'])-1):
            plt.plot(s[20*k:20*(k+1)], swing_trj[name][k](s[20*k:20*(k+1)]), '-')
        plt.grid()
        plt.title('Trajectory ' + name)
    plt.xlabel('Time [s]')
    plt.show()


if __name__ == '__main__':

    # desired publish frequency
    freq = 100
    #casannis(freq)
    try:
        casannis(freq)
    except rospy.ROSInterruptException:
        pass

