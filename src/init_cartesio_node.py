#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Imu
from cartesian_interface.srv import ResetWorld
from geometry_msgs.msg import Pose


def cartesian_reset_world_client(world_pose):

    rospy.wait_for_service('/cartesian/reset_world')

    try:
        srv_handle = rospy.ServiceProxy('/cartesian/reset_world', ResetWorld)
        resp1 = srv_handle(world_pose, '')
        print(resp1.message)

    except rospy.ServiceException as e:
        print("Service call failed: %s" %e)


def cartesio_initializer():

    rospy.init_node('initialize_cartesio', anonymous=True)

    # accept imu orientation
    imu_msg_ = rospy.wait_for_message("/xbotcore/imu/imu_link", Imu, timeout=None)

    new_world = Pose()
    new_world.position.x = 0.0
    new_world.position.y = 0.0
    new_world.position.z = 0.0
    new_world.orientation.x = - imu_msg_.orientation.x
    new_world.orientation.y = - imu_msg_.orientation.y
    new_world.orientation.z = - imu_msg_.orientation.z
    new_world.orientation.w = imu_msg_.orientation.w

    cartesian_reset_world_client(new_world)


if __name__ == '__main__':

    cartesio_initializer()