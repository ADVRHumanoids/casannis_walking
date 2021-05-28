#!/usr/bin/env python2

import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
import trj_interpolation as interpol
from shapely.geometry import Polygon
from matplotlib import pyplot as plt


# radius of centauro wheels
R = 0.078

# map feet to a string for publishing to the corresponding topic
id_name = ['FL', 'FR', 'HL', 'HR']
task_name_contact = ["contact1_rp", "contact2_rp", "contact3_rp", "contact4_rp"]
leg_num = len(id_name)


def roll_feet(freq):
    '''

    Args:
        freq: interpolation and publish frequency

    Returns:
        node for rolling the feet of centauro. Compatible with a driving stack.
    '''

    f_init = []     # position of wheel frames
    f_cont = []     # position of contact frames

    # loop for all feet
    for i in range(leg_num):

        # initial wheel frame
        '''f_init.append(rospy.wait_for_message("/cartesian/" + id_name[i] + "_wheel/current_reference",
                                             PoseStamped,
                                             timeout=None))'''
        f_init.append(rospy.wait_for_message("/cartesian/" + task_name_contact[i] + "/current_reference",
                                             PoseStamped,
                                             timeout=None))
        # initial contact position
        f_cont.append([f_init[i].pose.position.x, f_init[i].pose.position.y, f_init[i].pose.position.z - R])

    # contact points as array
    contacts = [np.array(x) for x in f_cont]

    # com initial position
    com_sub = rospy.wait_for_message("/cartesian/com/current_reference", PoseStamped, timeout=None)
    com_init = [com_sub.pose.position.x, com_sub.pose.position.y, com_sub.pose.position.z]

    # Get ROS Parameters

    # ID of the foot to be moved, get from parameters
    swing_id = rospy.get_param("~sw_id")    # from command line as swing_id:=1/2/3/4
    swing_id = swing_id.rstrip(']').lstrip('[').split(',')  # convert swing_id from "[a, b]" to [a,b]
    swing_id = [int(i) for i in swing_id]

    # number of rolling steps
    step_num = len(swing_id)

    # Target position of the foot wrt to the current position
    tgt_dx = []
    tgt_dy = []
    tgt_dz = 0.0

    for i in range(step_num):

        # dx, dy of every swing foot from parameter server (as tgt_dx1)
        tgt_dx.append(rospy.get_param("~tgt_dx" + str(swing_id[i])))
        tgt_dy.append(rospy.get_param("~tgt_dy" + str(swing_id[i])))

    print(tgt_dx)   # debug
    print(tgt_dy)

    # variables to loop for swing legs
    swing_tgt = []  # target positions as list
    swing_t = []    # time periods of the swing phases
    f_pub_ = []     # list of publishers for the swing foot
    f_msg = []                  # list of messages to be published for swing feet
    swing_contacts = []         # contact positions of the swing feet

    for i in range(step_num):

        # targets
        swing_tgt.append([contacts[swing_id[i] - 1][0] + tgt_dx[i],
                          contacts[swing_id[i] - 1][1] + tgt_dy[i],
                          contacts[swing_id[i] - 1][2] + tgt_dz])

        # swing phases
        swing_t.append(rospy.get_param("~sw_t" + str(swing_id[i])))  # from command line as swing_t:="[a,b]"
        swing_t[i] = swing_t[i].rstrip(']').lstrip('[').split(',')  # convert swing_t from "[a, b]" to ['a','b']
        swing_t[i] = [float(swing_t[i][0]), float(swing_t[i][1])]

        # swing feet trj publishers
        #foot_topic = '/cartesian/' + id_name[swing_id[i] - 1] + '_wheel/reference'
        foot_topic = '/cartesian/' + task_name_contact[swing_id[i] - 1] + '/reference'

        f_pub_.append(rospy.Publisher(foot_topic, PoseStamped, queue_size=10))

        # feet trj messages
        f_msg.append(PoseStamped())

        # keep same orientation
        f_msg[i].pose.orientation = f_init[swing_id[i] - 1].pose.orientation

        swing_contacts.append(contacts[swing_id[i] - 1])

    total_time = [min([k[0] for k in swing_t]), max([k[1] for k in swing_t])]

    print("total time is:", total_time)     # debug

    interpl_trj = []  # interpolate the trj at a specified interpolation frequency

    for i in range(step_num):
        interpl_trj.append(interpol.swing_trj_triangle(sw_curr=swing_contacts[i], sw_tgt=swing_tgt[i],
                                                       clear=0, sw_t=swing_t[i], total_t=total_time,
                                                       resol=freq, spline_order=3))

    # final support polygon
    polygon_points = []

    for i in range(leg_num):

        polygon_points.append([round(contacts[i][0], 4), round(contacts[i][1], 4)])

        if i+1 in swing_id:

            indic = swing_id.index(i + 1)
            polygon_points[i][0] = round(polygon_points[i][0] + tgt_dx[indic], 4)
            polygon_points[i][1] = round(polygon_points[i][1] + tgt_dy[indic], 4)

    polygon = Polygon(polygon_points)

    # send com at the centroid of the polygon and a bit forward
    com_tgt = [polygon.centroid.coords[0][0] + 0.05, polygon.centroid.coords[0][1]] + [com_init[2]]

    interpl_trj.append(interpol.swing_trj_triangle(sw_curr=com_init, sw_tgt=com_tgt,
                                                   clear=0, sw_t=swing_t[0], total_t=total_time,
                                                   resol=freq, spline_order=3))

    com_pub_ = rospy.Publisher('/cartesian/com/reference', PoseStamped, queue_size=10)
    com_msg = PoseStamped()

    # All points to be published
    N_total = int((total_time[1] - total_time[0]) * freq)  # total points --> total time * interpolation frequency

    # approximate distance covered during swing from all feet
    tgt_ds = sum([interpl_trj[i]['s'] for i in range(step_num)])

    # mean velocity of every swing foot
    mean_foot_velocity = tgt_ds / (step_num * (total_time[1] - total_time[0]))
    print('Mean foot velocity is:', mean_foot_velocity, 'm/sec')

    rate = rospy.Rate(freq)  # Frequency trj publishing
    # loop interpolation points to publish on a specified frequency
    for counter in range(N_total):

        if not rospy.is_shutdown():

            for i in range(step_num):
                # swing foot
                f_msg[i].pose.position.x = interpl_trj[i]['x'][counter]
                f_msg[i].pose.position.y = interpl_trj[i]['y'][counter]
                # add radius as origin of the wheel frame is in the center
                f_msg[i].pose.position.z = interpl_trj[i]['z'][counter] + R

                # publish swing trajectory
                f_msg[i].header.stamp = rospy.Time.now()
                f_pub_[i].publish(f_msg[i])

            com_msg.pose.position.x = interpl_trj[step_num]['x'][counter]
            com_msg.pose.position.y = interpl_trj[step_num]['y'][counter]
            com_msg.pose.position.z = interpl_trj[step_num]['z'][counter]

            # publish swing trajectory
            com_msg.header.stamp = rospy.Time.now()
            com_pub_.publish(com_msg)

        rate.sleep()

    # plot swing trajectory
    # All points to be published
    '''s = np.linspace(total_time[0], total_time[1], N_total)
    coord_labels = ['x', 'y', 'z']
    for j in range(step_num+1):
        plt.figure()
        for i, name in enumerate(coord_labels):
            plt.subplot(3, 1, i + 1)
            plt.plot(s, interpl_trj[j][name])  # nominal trj
            plt.grid()
            plt.legend(['nominal', 'real'])
            plt.title('Trajectory ' + name)
        plt.xlabel('Time [s]')
    plt.show()'''


if __name__ == '__main__':

    rospy.init_node('casannis_roller', anonymous=True)

    # desired interpolation & publish frequency
    int_freq = 300

    try:
        roll_feet(int_freq)
    except rospy.ROSInterruptException:
        pass