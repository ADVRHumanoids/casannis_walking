#!/usr/bin/env python3

import rospy
from casannis_walking.msg import Four_contacts
from geometry_msgs.msg import WrenchStamped
import time

# contacts flag
window = 5
#contacts_window = [[True] * window for i in range(4)]
contacts_window = [[True, True, True, True, True],[True, True, True, True, True],[True,True,True,True,True],[True,True,True,True,True],]
#contacts_flag = [True] * 4
contacts_flag = [True,True,True,True]


def callback(msg, i):
    '''

    Args:
        msg: message received from the topic
        i: index of foot, from 1 to 4

    Returns:
        refreshes the contact flags
    '''

    global contacts_window
    global contacts_flag
    print("----------------------")
    print(contacts_window[i-1].reverse())
    print("Force is :", msg.wrench.force.z)
    # if force in z direction is below a threshold
    if msg.wrench.force.z < contact_threshold:

        # find the index of the window to set to False
        print(contacts_window[i - 1])
        try:
            violation_level = 4 - contacts_window[i - 1][::-1].index(False, 0, window) + 1

        except:
            violation_level = 0

        print("Foot number", i, "window violation number", violation_level)
        if violation_level == window:
            contacts_flag[i - 1] = False

        else:
            # assign false value
            contacts_window[i - 1][violation_level] = False
            contacts_flag[i - 1] = True
    # force above a threshold
    else:
        # set all window to True
        contacts_window[i - 1] = [True] * window
        contacts_flag[i - 1] = True
        rospy.sleep(0.002)


def contacts(pub_freq):

    # Feet contact state publisher
    contacts_pub_ = rospy.Publisher('/feet_contact_state', Four_contacts, queue_size=10)

    rospy.init_node('feet_contact_state', anonymous=True)

    # contact force threshold
    global contact_threshold
    contact_threshold = rospy.get_param("~contact_threshold")
    print(contact_threshold)

    callback_lambda1 = lambda x: callback(x, 1)
    callback_lambda2 = lambda x: callback(x, 2)
    callback_lambda3 = lambda x: callback(x, 3)
    callback_lambda4 = lambda x: callback(x, 4)

    rospy.Subscriber('/cartesian/force_estimation/contact_1', WrenchStamped, callback_lambda1)
    rospy.Subscriber('/cartesian/force_estimation/contact_2', WrenchStamped, callback_lambda2)
    rospy.Subscriber('/cartesian/force_estimation/contact_3', WrenchStamped, callback_lambda3)
    rospy.Subscriber('/cartesian/force_estimation/contact_4', WrenchStamped, callback_lambda4)

    contacts_msg = Four_contacts()

    rate = rospy.Rate(pub_freq)
    while True:

        # update values
        contacts_msg.f_left.data = contacts_flag[0]
        contacts_msg.f_right.data = contacts_flag[1]
        contacts_msg.h_left.data = contacts_flag[2]
        contacts_msg.h_right.data = contacts_flag[3]

        # publish
        contacts_pub_.publish(contacts_msg)

        rate.sleep()


if __name__ == '__main__':

    # desired publish frequency
    freq = 100

    try:
        contacts(freq)
    except rospy.ROSInterruptException:
        pass