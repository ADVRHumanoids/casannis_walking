#!/usr/bin/env python2

import rospy
import actionlib
import casannis_walking.msg

import gait_payload_node as gait_payload
from yiannis_centauro_pytools import initialize_cartesio


class DemoAction(object):

    # create messages that are used to publish feedback/result
    _feedback = casannis_walking.msg.DemoFeedback()
    _result = casannis_walking.msg.DemoResult()

    def __init__(self, name):
        self._action_name = name

        self._as = actionlib.SimpleActionServer(self._action_name, casannis_walking.msg.DemoAction,
                                            execute_cb=self.callback, auto_start=False)
        self._as.start()

    def callback(self, goal):

        print('start callback')

        # helper variables
        success = True

        # demo feedback
        self._feedback.demo_feedback.data = 'This is the feedback'

        # publish info to the console for the user
        rospy.loginfo('Executing the demo action with goal {}'.format(goal.platform))

        # start executing the action
        for i in range(1, 2):

            # check that preempt has not been requested by the client
            if self._as.is_preempt_requested():
                rospy.loginfo('%s: Preempted' % self._action_name)
                self._as.set_preempted()
                success = False
                break

            self._feedback.demo_feedback.data = 'Feedback in the execution'
            # publish the feedback
            self._as.publish_feedback(self._feedback)

            # publish frequency
            freq = 300

            # step
            rospy.set_param("~inclination_deg", 0.0)
            rospy.set_param("~box_conservative", False)

            rospy.set_param("~sw_t1", "[1.0, 3.0]")
            rospy.set_param("~sw_t2", "[4.0, 6.0]")
            rospy.set_param("~sw_t3", "[7.0, 9.0]")
            rospy.set_param("~sw_t4", "[10.0, 12.0]")
            rospy.set_param("~cont_det", False)
            rospy.set_param("~plots", False)
            rospy.set_param("~clear", 0.05)
            rospy.set_param("~min_for", 120)
            rospy.set_param("~mass_payl", "[10.0, 10.0]")
            rospy.set_param("~forward_arms", True)
            rospy.set_param("~linear_fvirt", False)

            rospy.set_param("~sw_id", "[3, 1, 4, 2]")
            rospy.set_param("~tgt_dx", "[0.2 ,0.2, 0.2, 0.2]")
            rospy.set_param("~tgt_dy", "[0.0 ,0.0, 0.0, 0.0]")
            rospy.set_param("~tgt_dz", "[0.0 ,0.1, 0.0, 0.1]")

            rospy.set_param("~imu_sensor", True)

            for i in range(2):
                gait_payload.casannis(freq)
                #rospy.sleep(3)
                #initialize_cartesio.cartesio_initializer()

            rospy.set_param("~tgt_dx", "[0.2 ,0.25, 0.2, 0.25]")
            gait_payload.casannis(freq)

            rospy.set_param("~tgt_dx", "[0.2 ,0.25, 0.2, 0.25]")
            gait_payload.casannis(freq)
            #initialize_cartesio.cartesio_initializer()

            rospy.set_param("~sw_id", "[3, 4, 1, 2]")
            rospy.set_param("~tgt_dx", "[0.24 ,0.24, 0.24, 0.24]")
            rospy.set_param("~tgt_dy", "[0.0 ,0.0, 0.0, 0.0]")
            rospy.set_param("~tgt_dz", "[0.1 ,0.1, 0.1, 0.1]")

            gait_payload.casannis(freq)
            #initialize_cartesio.cartesio_initializer()

            rospy.set_param("~tgt_dx", "[0.25 ,0.25, 0.25, 0.25]")
            gait_payload.casannis(freq)
            #rospy.sleep(3)
            #initialize_cartesio.cartesio_initializer()

            rospy.set_param("~tgt_dx", "[0.2 ,0.2, 0.2, 0.2]")
            rospy.set_param("~tgt_dz", "[0.1 ,0.1, 0.0, 0.0]")
            gait_payload.casannis(freq)
            #initialize_cartesio.cartesio_initializer()

            rospy.set_param("~tgt_dx", "[0.3 ,0.3, 0.3, 0.3]")
            gait_payload.casannis(freq)
            #initialize_cartesio.cartesio_initializer()

            rospy.set_param("~tgt_dx", "[0.2 ,0.2, 0.2, 0.2]")
            gait_payload.casannis(freq)
            #initialize_cartesio.cartesio_initializer()

            rospy.set_param("~tgt_dx", "[0.3 ,0.3, 0.3, 0.3]")
            gait_payload.casannis(freq)
            #initialize_cartesio.cartesio_initializer()

            rospy.loginfo('%s: Roll completed' % (self._action_name))

        if success:
            self._result.demo_result = True
            rospy.loginfo('%s: Succeeded' % self._action_name)
            self._as.set_succeeded(self._result)


if __name__ == '__main__':

    rospy.init_node('demo_action_ioannis')

    server = DemoAction('demo_action_ioannis')

    rospy.spin()