#! /usr/bin/env python2

import rospy

# Brings in the SimpleActionClient
import actionlib

# Brings in the messages used by the action, including the
# goal message and the result message.
import casannis_walking.msg
# from hhcm_perception.msg import Obstacles as Obst_array
from casannis_walking.msg import Obstancles as Obst_array

def demo_client():
    # Creates the SimpleActionClient, passing the type of the action
    # (DemoAction) to the constructor.
    client = actionlib.SimpleActionClient('demo_action_ioannis', casannis_walking.msg.DemoAction)

    # Waits until the action server has started up and started
    # listening for goals.
    client.wait_for_server()

    # Creates a goal to send to the action server.
    goal_msg = casannis_walking.msg.DemoGoal
    goal_msg.platform = perceive_platform()

    # print platform information
    print('Perceived platform:')
    rospy.loginfo(goal_msg.platform)

    # Sends the goal to the action server.
    client.send_goal(goal_msg)

    # Waits for the server to finish performing the action.
    client.wait_for_result()

    # Prints out the result of executing the action
    return client.get_result()  # A DemoResult

def perceive_platform():

    rospy.sleep(2)  # wait for the perception to stabilize

    boxes = rospy.wait_for_message("/obstacles", Obst_array, timeout=None)
    rospy.loginfo(len(boxes.obstacles))

    # keep obstacle that is less than 0.4 m height
    for i in range(len(boxes.obstacles)):
        if boxes.obstacles[i].height < 0.4:
            platform = boxes.obstacles[i]
            break

    return platform


if __name__ == '__main__':
    try:
        # Initializes a rospy node so that the SimpleActionClient can
        # publish and subscribe over ROS.
        rospy.init_node('demo_ioannis_client')
        result = demo_client()
        print("Result:", result)
    except rospy.ROSInterruptException:
        print("program interrupted before completion")