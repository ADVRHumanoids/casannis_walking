#!/usr/bin/env python3

import rospy
from casannis_walking.msg import Four_contacts
from geometry_msgs.msg import WrenchStamped

# contacts flag
contacts_flag = [False] * 4


def callback(msg, i):

    global contacts_flag

    if msg.wrench.force.z > contact_threshold:
        contacts_flag[i-1] = True
    else:
        contacts_flag[i-1] = False


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
    freq = 300

    try:
        contacts(freq)
    except rospy.ROSInterruptException:
        pass