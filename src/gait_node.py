#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from centauro_contact_detection.msg import Contacts as Contacts_msg
from gait import Gait

# radius of centauro wheels
R = 0.078
#task_name_contact = ["contact1", "contact2", "contact3", "contact4"]  # FL_wheel
task_name_contact = ['FL_wheel', 'FR_wheel', 'HL_wheel', 'HR_wheel']


def contacts_callback(msg):

    # pass to global scope
    global sw_contact_msg
    sw_contact_msg = msg


def casannis(int_freq):

    """
    This function call the optimization problem constructor, the solver, the interpolator and interfaces with cartesio
    through ros topics
    Args:
        int_freq: desired interpolation frequency (affects only the interpolation and not the optimal solution)
    Returns:
        publish the desired data to cartesio through ros topics

    """

    rospy.init_node('casannis', anonymous=True)

    # map feet to a string for publishing to the corresponding topic
    id_name = ['FL', 'FR', 'HL', 'HR']
    id_contact_name = ['f_left', 'f_right', 'h_left', 'h_right']

    # accept one message for com and feet initial position
    com_init = rospy.wait_for_message("/cartesian/com/current_reference", PoseStamped, timeout=None)

    f_init = []     # position of wheel frames
    f_cont = []     # position of contact frames

    # loop for all feet
    for i in range(len(id_name)):
        f_init.append(rospy.wait_for_message("/cartesian/" + id_name[i] + "_wheel/current_reference",
                                             PoseStamped,
                                             timeout=None))
        f_cont.append([f_init[i].pose.position.x, f_init[i].pose.position.y, f_init[i].pose.position.z - R])

        '''f_init.append(rospy.wait_for_message("/cartesian/" + task_name_contact[i] + "/current_reference",
                                             PoseStamped,
                                             timeout=None))
        f_cont.append([f_init[i].pose.position.x, f_init[i].pose.position.y, f_init[i].pose.position.z])'''

    # contact points as array
    contacts = [np.array(x) for x in f_cont]

    # state vector
    c0 = np.array([com_init.pose.position.x, com_init.pose.position.y, com_init.pose.position.z])
    dc0 = np.zeros(3)
    ddc0 = np.zeros(3)
    x0 = np.hstack([c0, dc0, ddc0])

    # Get ROS Parameters

    # ID of the foot to be moved, get from parameters
    swing_id = rospy.get_param("~sw_id")    # from command line as swing_id:=1/2/3/4
    swing_id = swing_id.rstrip(']').lstrip('[').split(',')  # convert swing_id from "[a, b]" to [a,b]
    swing_id = [int(i) for i in swing_id]

    # number of steps
    step_num = len(swing_id)

    # Target position of the foot wrt to the current position
    tgt_dx = rospy.get_param("~tgt_dx")  # get from command line as target_dx
    tgt_dy = rospy.get_param("~tgt_dy")
    tgt_dz = rospy.get_param("~tgt_dz")

    # Clearance to be achieved, counted from the highest point
    swing_clear = rospy.get_param("~clear")  # get from command line as target_dx

    # force threshold
    minimum_force = rospy.get_param("~min_for")

    # apply or no contact detection
    cont_detection = rospy.get_param("~cont_det")  # from command line as contact_det:=True/False

    # variables to loop for swing legs
    swing_tgt = []  # target positions as list
    swing_t = []    # time periods of the swing phases
    f_pub_ = []     # list of publishers for the swing foot
    com_msg = PoseStamped()     # message to be published for com
    f_msg = []                  # list of messages to be published for swing feet
    swing_contacts = []         # contact positions of the swing feet

    for i in range(step_num):
        # targets
        swing_tgt.append([contacts[swing_id[i] - 1][0] + tgt_dx, contacts[swing_id[i] - 1][1] + tgt_dy, contacts[swing_id[i] - 1][2] + tgt_dz])

        # swing phases
        swing_t.append(rospy.get_param("~sw_t" + str(i+1)))  # from command line as swing_t:="[a,b]"
        swing_t[i] = swing_t[i].rstrip(']').lstrip('[').split(',')  # convert swing_t from "[a, b]" to [a,b]
        swing_t[i] = [float(i) for i in swing_t[i]]

        # swing feet trj publishers
        '''f_pub_.append(rospy.Publisher('/cartesian/' + id_name[swing_id[i] - 1] + '_wheel/reference',
                                      PoseStamped,
                                      queue_size=10))'''

        f_pub_.append(rospy.Publisher('/cartesian/' + task_name_contact[swing_id[i] - 1] + '/reference',
                                      PoseStamped,
                                      queue_size=10))

        # feet trj messages
        f_msg.append(PoseStamped())

        # keep same orientation
        f_msg[i].pose.orientation = f_init[swing_id[i] - 1].pose.orientation

        swing_contacts.append(contacts[swing_id[i] - 1])

    # CoM trj publisher
    com_pub_ = rospy.Publisher('/cartesian/com/reference', PoseStamped, queue_size=10)

    # Subscriber for contact flags
    rospy.Subscriber('/contacts', Contacts_msg, contacts_callback)

    # object class of the optimization problem
    walk = Gait(mass=95, N=int((swing_t[-1][1] + 1.0) / 0.2), dt=0.2)

    # call the solver of the optimization problem
    # sol is the directory returned by solve class function contains state, forces, control values
    sol = walk.solve(x0=x0, contacts=contacts, swing_id=[x-1 for x in swing_id], swing_tgt=swing_tgt,
                     swing_clearance=swing_clear, swing_t=swing_t, min_f=minimum_force)

    # interpolate the trj, pass solution values and interpolation frequency
    interpl = walk.interpolate(sol, swing_contacts, swing_tgt, swing_clear, swing_t, int_freq)

    # All points to be published
    N_total = int(walk._N * walk._dt * int_freq)  # total points --> total time * interpolation frequency

    # executed trj points
    executed_trj = []

    # early contact flags default values
    early_contact = [False, False, False, False]

    # times activating contact detection
    t_early = [swing_t[i][0] + 0.7 * (swing_t[i][1] - swing_t[i][0]) for i in range(step_num)]

    # time intervals [swing_start, early_cont_detection_start, swing_stop]
    delta_t_early = [[swing_t[i][0], t_early[i], swing_t[i][1]] for i in range(step_num)]

    # trj points during all swing phases
    N_swing_total = int(int_freq * sum([swing_t[i][1] - swing_t[i][0] for i in range(step_num)]))

    # approximate distance covered during swing
    tgt_ds = sum([interpl['sw'][i]['s'] for i in range(step_num)])

    # publish freq wrt the desired swing velocity
    #freq = swing_vel * N_swing_total / tgt_ds

    # mean velocity of the swing foot
    mean_foot_velocity = tgt_ds / (step_num * (swing_t[0][1] - swing_t[0][0]))
    print('Mean foot velocity is:', mean_foot_velocity, 'm/sec')

    rate = rospy.Rate(int_freq)  # Frequency trj publishing
    # loop interpolation points to publish on a specified frequency
    for counter in range(N_total):

        if not rospy.is_shutdown():

            # check if current time is within swing phase and contact detection
            for i in range(step_num):

                # swing phase check
                if delta_t_early[i][0] <= interpl['t'][counter] <= delta_t_early[i][2]:
                    swing_phase = i

                    # time for contact detection
                    if interpl['t'][counter] >= delta_t_early[i][1]:
                        early_check = True

                    else:
                        early_check = False
                    break

                else:
                    swing_phase = -1    # not in swing phase
                    early_check = False

            # com trajectory
            com_msg.pose.position.x = interpl['x'][0][counter]
            com_msg.pose.position.y = interpl['x'][1][counter]
            com_msg.pose.position.z = interpl['x'][2][counter]

            # swing foot
            f_msg[swing_phase].pose.position.x = interpl['sw'][swing_phase]['x'][counter]
            f_msg[swing_phase].pose.position.y = interpl['sw'][swing_phase]['y'][counter]
            # add radius as origin of the wheel frame is in the center
            f_msg[swing_phase].pose.position.z = interpl['sw'][swing_phase]['z'][counter] + R

            # publish com trajectory regardless contact detection
            com_msg.header.stamp = rospy.Time.now()
            com_pub_.publish(com_msg)

            if swing_phase == -1:
                pass

            # do not check for early contact
            elif not cont_detection or early_check is False:

                # publish swing trajectory
                f_msg[swing_phase].header.stamp = rospy.Time.now()
                f_pub_[swing_phase].publish(f_msg[swing_phase])

            # If no early contact detected yet
            elif not early_contact[swing_phase]:

                # if there is contact
                if getattr(getattr(sw_contact_msg, id_contact_name[swing_id[swing_phase] - 1]), 'data'):

                    early_contact[swing_phase] = True  # stop swing trajectory of this foot

                    executed_trj.append(counter)    # save counter
                    print("early contact detected ", counter)

                # if no contact
                else:
                    # publish swing trajectory
                    f_msg[swing_phase].header.stamp = rospy.Time.now()
                    f_pub_[swing_phase].publish(f_msg[swing_phase])

        rate.sleep()

    # print the trajectories
    try:
        # there was early contact detected
        if early_contact.index(True) + 1:
            print("Early contact detected. Trj Counter is:", executed_trj, "out of total", N_total-1)

            if rospy.get_param("~plots"):
                walk.print_trj(sol, interpl, int_freq, executed_trj)
    except:
        print("No early contact detected")

        if rospy.get_param("~plots"):
            walk.print_trj(sol, interpl, int_freq, [N_total-1, N_total-1, N_total-1, N_total-1])


if __name__ == '__main__':

    # desired interpolation frequency
    interpolation_freq = 300

    try:
        casannis(interpolation_freq)
    except rospy.ROSInterruptException:
        pass