#!/usr/bin/env python2

import rospy
import actionlib
import casannis_walking.msg

import step_node as step
import roll_node as roll

from tf_listener import get_transform

wheel_radius = 0.078
safety_from_edges = 0.03
bad_roll = 0.0#0.04


class DemoAction(object):

    # create messages that are used to publish feedback/result
    _feedback = casannis_walking.msg.DemoFeedback()
    _result = casannis_walking.msg.DemoResult()

    def __init__(self, name):

        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, casannis_walking.msg.DemoAction,
                                                execute_cb=self.execute_callback_fast, auto_start=False)
        self._as.start()

    def execute_callback_fast(self, goal):

        # robot & platform localization in global frame
        pelvis_pose_init = get_transform('fixed_frame', 'pelvis')
        pelvis_x = pelvis_pose_init.transform.translation.x

        # info from the received goal
        simulation = goal.simulation
        h_platform = goal.platform.height
        len_platform = goal.platform.depth
        edge1_x = 1.0
        edge2_x = edge1_x + len_platform

        if simulation:
            h_error = 0.03
            contact_detection = True
        else:
            h_error = 0.0
            contact_detection = False

        # helper variables
        rate = rospy.Rate(100)
        success = True

        # demo feedback
        self._feedback.demo_feedback = 'This is the feedback'

        # publish info to the console for the user
        rospy.loginfo('%s: Executing the demo action with goal %i with seeds %s' % (
            self._action_name, goal.simulation, self._feedback.demo_feedback))

        # start executing the action
        for i in range(1, 2):

            # check that preempt has not been requested by the client
            if self._as.is_preempt_requested():
                rospy.loginfo('%s: Preempted' % self._action_name)
                self._as.set_preempted()
                success = False
                break

            self._feedback.demo_feedback = 'Feedback in the execution'
            # publish the feedback
            self._as.publish_feedback(self._feedback)

            # publish frequency
            freq = 300

            # distance to be covered
            fr_wheel_x = get_transform('fixed_frame', 'contact_2').transform.translation.x
            print('FR wheel pose is:',fr_wheel_x )
            dist1 = edge1_x - fr_wheel_x - wheel_radius - safety_from_edges + bad_roll
            print('distance to roll is:', dist1)

            # roll 1, 3, 4
            rospy.set_param('~sw_id', "[1, 2, 3, 4]")

            rospy.set_param('~tgt_dx1', (0.2 + dist1))
            rospy.set_param('~tgt_dx2', dist1)
            rospy.set_param('~tgt_dx3', (0.2 + dist1))
            rospy.set_param('~tgt_dx4', (0.2 + dist1))

            rospy.set_param('~tgt_dy1', 0.0)
            rospy.set_param('~tgt_dy2', 0.0)
            rospy.set_param('~tgt_dy3', 0.0)
            rospy.set_param('~tgt_dy4', 0.0)

            rospy.set_param('~sw_t1', "[0.0, 4.0]")
            rospy.set_param('~sw_t2', "[0.0, 4.0]")
            rospy.set_param('~sw_t3', "[0.0, 4.0]")
            rospy.set_param('~sw_t4', "[0.0, 4.0]")

            roll.roll_feet(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

            # step on 2
            rospy.set_param('~sw_id', 2)
            rospy.set_param('~tgt_dx', 0.25)
            rospy.set_param('~tgt_dy', 0.0)
            rospy.set_param('~tgt_dz', (h_platform - h_error))
            rospy.set_param('~clear', (0.05 + h_error))
            rospy.set_param('~sw_t', "[2.0, 6.0]")
            rospy.set_param('~min_for', 100)
            rospy.set_param('~cont_det', contact_detection)
            rospy.set_param('~plots', False)

            step.casannis(freq)
            rospy.loginfo('%s: Step completed' % (self._action_name))

            # distance to be covered - get transformation between feet 2 and 4
            hr_wheel_x = get_transform('fixed_frame', 'contact_4').transform.translation.x
            print('HR wheel pose is:', hr_wheel_x)
            dist_2 = edge1_x - hr_wheel_x - wheel_radius - safety_from_edges + bad_roll
            print('distance to roll is:', dist_2)
            #dist_2 = 0.7 # temporary

            # roll 1, 2, 3
            rospy.set_param('~sw_id', "[1, 2, 3, 4]")
            rospy.set_param('~tgt_dx1', (0.1 + dist_2))
            rospy.set_param('~tgt_dx2', (0.1 + dist_2))
            rospy.set_param('~tgt_dx3', (0.1 + dist_2))
            rospy.set_param('~tgt_dx4', dist_2)

            rospy.set_param('~sw_t1', "[0.0, 6.0]")
            rospy.set_param('~sw_t2', "[0.0, 6.0]")
            rospy.set_param('~sw_t3', "[0.0, 6.0]")
            rospy.set_param('~sw_t4', "[0.0, 6.0]")

            roll.roll_feet(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

            # step on 4
            rospy.set_param('~sw_id', 4)
            rospy.set_param('~tgt_dx', 0.25)
            rospy.set_param('~tgt_dy', 0.0)
            rospy.set_param('~tgt_dz', 0.17)
            rospy.set_param('~clear', 0.08)

            step.casannis(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

            # distance to be covered
            fr_wheel_x = get_transform('fixed_frame', 'contact_2').transform.translation.x
            print('FR wheel pose is:', fr_wheel_x)
            dist_3 = edge2_x - fr_wheel_x - safety_from_edges
            print('distance to roll is:', dist_3)
            #dist_3 = 1.33

            # roll 1, 3, 4 and then roll 3
            rospy.set_param('~sw_id', "[1, 2, 3, 4]")
            rospy.set_param('~tgt_dx1', (0.2 + dist_3))
            rospy.set_param('~tgt_dx2', dist_3)
            rospy.set_param('~tgt_dx3', (0.2 + 0.1 + dist_3))
            rospy.set_param('~tgt_dx4', (0.2 + dist_3))

            rospy.set_param('~sw_t1', "[0.0, 6.0]")
            rospy.set_param('~sw_t2', "[0.0, 6.0]")
            rospy.set_param('~sw_t3', "[0.0, 6.0]")
            rospy.set_param('~sw_t4', "[0.0, 6.0]")

            roll.roll_feet(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

            # step off 2
            rospy.set_param('~sw_id', 2)
            rospy.set_param('~tgt_dx', 0.2)
            rospy.set_param('~tgt_dy', 0.0)
            rospy.set_param('~tgt_dz', - 0.23)
            rospy.set_param('~clear', 0.04)

            step.casannis(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

            # distance to be covered
            hr_wheel_x = get_transform('fixed_frame', 'contact_4').transform.translation.x
            dist_4 = edge2_x - hr_wheel_x - safety_from_edges + bad_roll
            print('distance to roll is:', dist_4)
            #dist_4 = 0.33

            # roll 1, 2, 3
            rospy.set_param('~sw_id', "[1, 2, 3, 4]")
            rospy.set_param('~tgt_dx1', (0.2 + dist_4))
            rospy.set_param('~tgt_dx2', (0.2 + dist_4))
            rospy.set_param('~tgt_dx3', (0.2 + dist_4))
            rospy.set_param('~tgt_dx4', dist_4)

            roll.roll_feet(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

            # step off 4
            rospy.set_param('~sw_id', 4)
            rospy.set_param('~tgt_dx', 0.2)
            rospy.set_param('~tgt_dy', 0.0)
            rospy.set_param('~tgt_dz', - 0.23)
            rospy.set_param('~clear', 0.04)

            step.casannis(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

            # roll 1, 2
            rospy.set_param('~sw_id', "[1, 2]")
            rospy.set_param('~tgt_dx1', 0.2)
            rospy.set_param('~tgt_dx2', 0.2)
            rospy.set_param('~tgt_dx3', 0.2)
            rospy.set_param('~tgt_dx4', 0.2)

            roll.roll_feet(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

        if success:
            self._result.demo_result = True
            rospy.loginfo('%s: Succeeded' % self._action_name)
            self._as.set_succeeded(self._result)

    def execute_callback(self, goal):

        # robot & platform localization in global frame
        pelvis_x = 0.045
        fr_wheel_x = 0.394  # to be found from tfs
        edge1_x = 1.0

        # distances to be covered
        dist1 = edge1_x - fr_wheel_x - wheel_radius - safety_from_edges

        simulation = goal.simulation
        h_platform = goal.platform.height
        len_platform = goal.platform.depth

        if simulation:
            h_error = 0.03
            contact_detection = True
        else:
            h_error = 0.0
            contact_detection = False

        # helper variables
        rate = rospy.Rate(100)
        success = True

        # demo feedback
        self._feedback.demo_feedback = 'This is the feedback'

        # publish info to the console for the user
        rospy.loginfo('%s: Executing the demo action with goal %i with seeds %s' % (
            self._action_name, goal.simulation, self._feedback.demo_feedback))

        # start executing the action
        for i in range(1, 2):

            # check that preempt has not been requested by the client
            if self._as.is_preempt_requested():
                rospy.loginfo('%s: Preempted' % self._action_name)
                self._as.set_preempted()
                success = False
                break

            self._feedback.demo_feedback = 'Feedback in the execution'
            # publish the feedback
            self._as.publish_feedback(self._feedback)

            # publish frequency
            freq = 300

            # roll 1, 3, 4
            rospy.set_param('~sw_id', "[1, 3, 4]")

            rospy.set_param('~tgt_dx1', 0.2)
            rospy.set_param('~tgt_dx2', 0.2)
            rospy.set_param('~tgt_dx3', 0.2)
            rospy.set_param('~tgt_dx4', 0.2)

            rospy.set_param('~tgt_dy1', 0.0)
            rospy.set_param('~tgt_dy2', 0.0)
            rospy.set_param('~tgt_dy3', 0.0)
            rospy.set_param('~tgt_dy4', 0.0)

            rospy.set_param('~sw_t1', "[0.0, 4.0]")
            rospy.set_param('~sw_t2', "[0.0, 4.0]")
            rospy.set_param('~sw_t3', "[0.0, 4.0]")
            rospy.set_param('~sw_t4', "[0.0, 4.0]")

            roll.roll_feet(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

            # move forward
            self.move_fwd(dist1, freq)

            # step on 2
            rospy.set_param('~sw_id', 2)
            rospy.set_param('~tgt_dx', 0.25)
            rospy.set_param('~tgt_dy', 0.0)
            rospy.set_param('~tgt_dz', (h_platform - h_error))
            rospy.set_param('~clear', (0.05 + h_error))
            rospy.set_param('~sw_t', "[2.0, 6.0]")
            rospy.set_param('~min_for', 100)
            rospy.set_param('~cont_det', contact_detection)
            rospy.set_param('~plots', False)

            step.casannis(freq)
            rospy.loginfo('%s: Step completed' % (self._action_name))

            # roll 1, 2, 3
            rospy.set_param('~sw_id', "[1, 2, 3]")
            rospy.set_param('~tgt_dx1', 0.1)
            rospy.set_param('~tgt_dx2', 0.1)
            rospy.set_param('~tgt_dx3', 0.1)
            rospy.set_param('~tgt_dx4', 0.1)

            roll.roll_feet(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

            # get transformation between feet 2 and 4
            dist2 = get_transform('contact_2', 'contact_4')
            rospy.loginfo(dist2)

            # move forward
            self.move_fwd(0.74, freq)

            # step on 4
            rospy.set_param('~sw_id', 4)
            rospy.set_param('~tgt_dx', 0.25)
            rospy.set_param('~tgt_dy', 0.0)
            rospy.set_param('~tgt_dz', (h_platform - h_error))
            rospy.set_param('~clear', (0.05 + h_error))

            step.casannis(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

            # roll 1, 3, 4
            rospy.set_param('~sw_id', "[1, 3, 4]")
            rospy.set_param('~tgt_dx1', 0.25)
            rospy.set_param('~tgt_dx2', 0.25)
            rospy.set_param('~tgt_dx3', 0.25)
            rospy.set_param('~tgt_dx4', 0.25)

            roll.roll_feet(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

            # roll 3
            rospy.set_param('~sw_id', "[3]")
            rospy.set_param('~tgt_dx1', 0.1)
            rospy.set_param('~tgt_dx2', 0.1)
            rospy.set_param('~tgt_dx3', 0.1)
            rospy.set_param('~tgt_dx4', 0.1)

            roll.roll_feet(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

            # move forward
            self.move_fwd(1.33, freq)

            # step off 2
            rospy.set_param('~sw_id', 2)
            rospy.set_param('~tgt_dx', 0.25)
            rospy.set_param('~tgt_dy', 0.0)
            rospy.set_param('~tgt_dz', -(h_platform + h_error))
            rospy.set_param('~clear', 0.04)

            step.casannis(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

            # roll 1, 2, 3
            rospy.set_param('~sw_id', "[1, 2, 3]")
            rospy.set_param('~tgt_dx1', 0.2)
            rospy.set_param('~tgt_dx2', 0.2)
            rospy.set_param('~tgt_dx3', 0.2)
            rospy.set_param('~tgt_dx4', 0.2)

            roll.roll_feet(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

            # move forward
            self.move_fwd(0.33, freq)

            # step off 4
            rospy.set_param('~sw_id', 4)
            rospy.set_param('~tgt_dx', 0.25)
            rospy.set_param('~tgt_dy', 0.0)
            rospy.set_param('~tgt_dz', -(h_platform + h_error))
            rospy.set_param('~clear', 0.04)

            step.casannis(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

            # roll 1, 2
            rospy.set_param('~sw_id', "[1, 2]")
            rospy.set_param('~tgt_dx1', 0.2)
            rospy.set_param('~tgt_dx2', 0.2)
            rospy.set_param('~tgt_dx3', 0.2)
            rospy.set_param('~tgt_dx4', 0.2)

            roll.roll_feet(freq)
            rospy.loginfo('%s: Roll completed' % (self._action_name))

            # this step is not necessary, the sequence is computed at 1 Hz for demonstration purposes
            #rate.sleep()

        if success:
            self._result.demo_result = True
            rospy.loginfo('%s: Succeeded' % self._action_name)
            self._as.set_succeeded(self._result)

    def move_fwd(self, distance, pub_freq):

        rospy.set_param('~sw_id', "[1, 2, 3, 4]")
        rospy.set_param('~tgt_dx1', distance)
        rospy.set_param('~tgt_dx2', distance)
        rospy.set_param('~tgt_dx3', distance)
        rospy.set_param('~tgt_dx4', distance)

        rospy.set_param('~tgt_dy1', 0.0)
        rospy.set_param('~tgt_dy2', 0.0)
        rospy.set_param('~tgt_dy3', 0.0)
        rospy.set_param('~tgt_dy4', 0.0)

        rospy.set_param('~sw_t1', "[0.0, 4.0]")
        rospy.set_param('~sw_t2', "[0.0, 4.0]")
        rospy.set_param('~sw_t3', "[0.0, 4.0]")
        rospy.set_param('~sw_t4', "[0.0, 4.0]")

        roll.roll_feet(pub_freq)
        rospy.loginfo('%s: Roll completed' % (self._action_name))


if __name__ == '__main__':

    rospy.init_node('demo_action_ioannis')

    server = DemoAction('demo_action_ioannis')

    rospy.spin()