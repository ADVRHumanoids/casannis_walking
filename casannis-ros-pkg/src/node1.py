#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
from walking import Walking
import numpy as np


def talker(int_data, pub_freq):
    '''

    Args:
        int_data: interpolated data to be published,
        that is a dictionary with time, state vector and forces
        pub_freq: desired publish frequency

    Returns:
        published the desired data to cartesio ros topics
    '''

    com_pub_ = rospy.Publisher('/cartesian/com/reference', PoseStamped, queue_size=10)
    rospy.init_node('talker', anonymous=True)

    msg = PoseStamped()

    counter = 0
    rate = rospy.Rate(pub_freq) # 100hz
    while not rospy.is_shutdown():

        if counter < 600:

            msg.pose.position.x = int_data['x'][0][counter]
            msg.pose.position.y = int_data['x'][1][counter]
            msg.pose.position.z = int_data['x'][2][counter] - 0.64  # change reference coordinate system

            com_pub_.publish(msg)

        else:
            break

        counter = counter + 2
        rate.sleep()


if __name__ == '__main__':

    # create an instance of the class
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

    freq = 100
    talker(interpl, freq)
    '''try:
        talker(interpl)
    except rospy.ROSInterruptException:
        pass
    '''
