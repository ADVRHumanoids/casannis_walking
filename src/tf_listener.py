#!/usr/bin/env python2
import rospy

import tf2_ros
from geometry_msgs.msg import TransformStamped

# ---------------> Need to transfer this node in yiannis_centauro_pytools

def get_transform(frame, base_frame):

    # topic to publish the requested transform
    publ_topic = 'tf_' + base_frame + '_to_' + frame
    transform_pub = rospy.Publisher(publ_topic, TransformStamped, queue_size=10)

    # construct listener
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    #listener = tf.TransformListener()

    # define a msg
    #trans_msg = TransformStamped()

    # receive in a loop
    rate = rospy.Rate(200.0)
    trans = False
    while not trans:
        try:
            trans = tfBuffer.lookup_transform(frame, base_frame, rospy.Time())
            #(trans, rot) = listener.lookupTransform(frame, base_frame, rospy.Time(0))

            '''trans_msg.transform.translation.x = trans[0]
            trans_msg.transform.translation.y = trans[1]
            trans_msg.transform.translation.z = trans[2]
            trans_msg.transform.rotation.x = rot[0]
            trans_msg.transform.rotation.y = rot[1]
            trans_msg.transform.rotation.z = rot[2]
            trans_msg.transform.rotation.w = rot[3]'''

            #transform_pub.publish(trans_msg)

            transform_pub.publish(trans)
            #print('Transform is:')
            #rospy.loginfo(trans)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

        rate.sleep()

    return trans


if __name__ == '__main__':

    b_frame = 'fixed_frame'
    c_frame = 'wheel_2'

    rospy.init_node('tf_listener_' + b_frame + '_to_' + c_frame)

    get_transform(b_frame, c_frame)