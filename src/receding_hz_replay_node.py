#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from centauro_contact_detection.msg import Contacts as Contacts_msg
from casannis_walking.msg import PayloadAware_plans as MotionPlan_msg
# radius of centauro wheels
R = 0.078
#task_name_contact = ["contact1", "contact2", "contact3", "contact4"]  # FL_wheel
task_name_contact = ['FL_wheel', 'FR_wheel', 'HL_wheel', 'HR_wheel']

task_name_moving_contact = ['left_hand', 'right_hand']
rod_plate_CoG_urdf = [4.8407693e-02,  2.0035723e-02,  7.7533287e-02]    # not used currently


def contacts_callback(msg):

    # pass to global scope
    global sw_contact_msg
    sw_contact_msg = msg


def motion_plan_callback(msg):
    # pass message to global scope
    global received_motion_plan
    received_motion_plan = {
        'x': msg.state,
        'u': msg.control,
        'Pl_mov': msg.left_arm_pos,
        'Pr_mov': msg.right_arm_pos,
        'DPl_mov': msg.left_arm_vel,
        'DPr_mov': msg.right_arm_vel,
        'F': msg.leg_forces,
        'F_virt_l': msg.left_arm_force,
        'F_virt_r': msg.right_arm_force,
    }


def casannis(int_freq):

    """
    This function call the optimization problem constructor, the solver, the interpolator and interfaces with cartesio
    through ros topics
    Args:
        int_freq: desired interpolation frequency (affects only the interpolation and not the optimal solution)
    Returns:
        publish the desired data to cartesio through ros topics

    """

    #  name prefix for the node that published the motion plans
    planner_node_prefix = '/casannis_planner'
    # get inclination of terrain
    inclination_deg = rospy.get_param(planner_node_prefix + "/inclination_deg")

    # get param conservative_box which defines if the box constraints are near the chest or not
    arm_box_conservative = rospy.get_param(planner_node_prefix + "/box_conservative")

    # select gait among different developments
    forward_arm_config = rospy.get_param(planner_node_prefix + "/forward_arms")
    linear_fvirt = rospy.get_param(planner_node_prefix + "/linear_fvirt")

    if forward_arm_config:
        if linear_fvirt:
            from gait_with_payload import Gait as SelectedGait
        else:
            from gait_with_payload import GaitNonlinear as SelectedGait
    else:
        if linear_fvirt:
            from gait_with_payload_backward_arms import Gait as SelectedGait
        else:
            from gait_with_payload_backward_arms import GaitNonlinearBackward as SelectedGait

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

    # hands
    lhand_init = rospy.wait_for_message("/cartesian/left_hand/current_reference", PoseStamped, timeout=None)
    rhand_init = rospy.wait_for_message("/cartesian/right_hand/current_reference", PoseStamped, timeout=None)

    # contact points as array
    contacts = [np.array(x) for x in f_cont]

    # hands
    h_init = [lhand_init, rhand_init]
    lh_mov = [lhand_init.pose.position.x, lhand_init.pose.position.y, lhand_init.pose.position.z]
    rh_mov = [rhand_init.pose.position.x, rhand_init.pose.position.y, rhand_init.pose.position.z]

    # hands
    moving_contact = [[np.array(lh_mov), np.zeros(3)], [np.array(rh_mov), np.zeros(3)]]

    # state vector
    c0 = np.array([com_init.pose.position.x, com_init.pose.position.y, com_init.pose.position.z])
    dc0 = np.zeros(3)
    ddc0 = np.zeros(3)
    x0 = np.hstack([c0, dc0, ddc0])

    # Get ROS Parameters

    # ID of the foot to be moved, get from parameters
    swing_id = rospy.get_param(planner_node_prefix + "/sw_id")    # from command line as swing_id:=1/2/3/4
    swing_id = swing_id.rstrip(']').lstrip('[').split(',')  # convert swing_id from "[a, b]" to [a,b]
    swing_id = [int(i) for i in swing_id]

    # number of steps
    step_num = len(swing_id)

    # Target position of feet wrt to the current position
    tgt_dx = rospy.get_param(planner_node_prefix + "/tgt_dx")    # from command line as swing_id:=1/2/3/4
    tgt_dx = tgt_dx.rstrip(']').lstrip('[').split(',')  # convert swing_id from "[a, b]" to [a,b]
    tgt_dx = [float(i) for i in tgt_dx]

    tgt_dy = rospy.get_param(planner_node_prefix + "/tgt_dy")    # from command line as swing_id:=1/2/3/4
    tgt_dy = tgt_dy.rstrip(']').lstrip('[').split(',')  # convert swing_id from "[a, b]" to [a,b]
    tgt_dy = [float(i) for i in tgt_dy]

    tgt_dz = rospy.get_param(planner_node_prefix + "/tgt_dz")    # from command line as swing_id:=1/2/3/4
    tgt_dz = tgt_dz.rstrip(']').lstrip('[').split(',')  # convert swing_id from "[a, b]" to [a,b]
    tgt_dz = [float(i) for i in tgt_dz]

    # Clearance to be achieved, counted from the highest point
    swing_clear = rospy.get_param(planner_node_prefix + "/clear")  # get from command line as target_dx

    # force threshold
    minimum_force = rospy.get_param(planner_node_prefix + "/min_for")

    # apply or no contact detection
    cont_detection = rospy.get_param(planner_node_prefix + "/cont_det")  # from command line as contact_det:=True/False

    # variables to loop for swing legs
    swing_tgt = []  # target positions as list
    swing_t = []    # time periods of the swing phases
    f_pub_ = []     # list of publishers for the swing foot
    com_msg = PoseStamped()     # message to be published for com
    f_msg = []                  # list of messages to be published for swing feet
    swing_contacts = []         # contact positions of the swing feet

    for i in range(step_num):
        print(i)
        # targets
        swing_tgt.append([contacts[swing_id[i] - 1][0] + tgt_dx[i],
                          contacts[swing_id[i] - 1][1] + tgt_dy[i],
                          contacts[swing_id[i] - 1][2] + tgt_dz[i]])

        # swing phases
        swing_t.append(rospy.get_param(planner_node_prefix + "/sw_t" + str(i+1)))  # from command line as swing_t:="[a,b]"
        swing_t[i] = swing_t[i].rstrip(']').lstrip('[').split(',')  # convert swing_t from "[a, b]" to [a,b]
        swing_t[i] = [float(ii) for ii in swing_t[i]]

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

    # receive weight of payloads
    payload_m = rospy.get_param(planner_node_prefix + "/mass_payl")  # from command line as swing_t:="[a,b]"
    payload_m = payload_m.rstrip(']').lstrip('[').split(',')  # convert swing_t from "[a, b]" to [a,b]
    payload_m = [float(i) for i in payload_m]

    print('Payload masses', payload_m)

    # CoM trj publisher
    com_pub_ = rospy.Publisher('/cartesian/com/reference', PoseStamped, queue_size=10)

    # hands' publishers and msgs
    left_h_pub_ = rospy.Publisher('/cartesian/' + task_name_moving_contact[0] + '/reference', PoseStamped, queue_size=10)
    right_h_pub_ = rospy.Publisher('/cartesian/' + task_name_moving_contact[1] + '/reference', PoseStamped, queue_size=10)
    lh_msg = PoseStamped()
    rh_msg = PoseStamped()
    lh_msg.pose.orientation = lhand_init.pose.orientation  # hands
    rh_msg.pose.orientation = rhand_init.pose.orientation

    # Subscriber for contact flags
    rospy.Subscriber('/contacts', Contacts_msg, contacts_callback)

    # Subscriber for motion plans
    rospy.Subscriber('/PayloadAware/motion_plan', MotionPlan_msg, motion_plan_callback)

    # # object class of the optimization problem
    walk = SelectedGait(mass=112, N=int((swing_t[-1][1] + 1.0) / 0.2), dt=0.2, payload_masses=payload_m,
                        slope_deg=inclination_deg, conservative_box=arm_box_conservative)
    #
    # # call the solver of the optimization problem
    # # sol is the directory returned by solve class function contains state, forces, control values
    # sol = walk.solve(x0=x0, contacts=contacts, mov_contact_initial=moving_contact, swing_id=[x-1 for x in swing_id],
    #                  swing_tgt=swing_tgt, swing_clearance=swing_clear, swing_t=swing_t, min_f=minimum_force)

    # interpolate the trj, pass solution values and interpolation frequency
    interpl = walk.interpolate(received_motion_plan, swing_contacts, swing_tgt, swing_clear, swing_t, int_freq)

    # All points to be published
    N_total = int(walk._problem_duration * int_freq)  # total points --> total time * interpolation frequency

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

            # hands trajectory
            lh_msg.pose.position.x = interpl['p_mov_l'][0][counter]
            lh_msg.pose.position.y = interpl['p_mov_l'][1][counter]
            lh_msg.pose.position.z = interpl['p_mov_l'][2][counter]

            rh_msg.pose.position.x = interpl['p_mov_r'][0][counter]
            rh_msg.pose.position.y = interpl['p_mov_r'][1][counter]
            rh_msg.pose.position.z = interpl['p_mov_r'][2][counter]

            # swing foot
            f_msg[swing_phase].pose.position.x = interpl['sw'][swing_phase]['x'][counter]
            f_msg[swing_phase].pose.position.y = interpl['sw'][swing_phase]['y'][counter]
            # add radius as origin of the wheel frame is in the center
            f_msg[swing_phase].pose.position.z = interpl['sw'][swing_phase]['z'][counter] + R

            # publish com trajectory regardless contact detection
            com_msg.header.stamp = rospy.Time.now()
            com_pub_.publish(com_msg)

            # publish hands trajectory regardless contact detection
            lh_msg.header.stamp = rospy.Time.now()
            left_h_pub_.publish(lh_msg)

            rh_msg.header.stamp = rospy.Time.now()
            right_h_pub_.publish(rh_msg)

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

            if rospy.get_param(planner_node_prefix + "/plots"):
                walk.print_trj(sol, interpl, int_freq, contacts, [x-1 for x in swing_id], executed_trj)
    except:
        print("No early contact detected")

        if rospy.get_param(planner_node_prefix + "/plots"):
            walk.print_trj(sol, interpl, int_freq, contacts, [x-1 for x in swing_id], [N_total-1, N_total-1, N_total-1, N_total-1])


if __name__ == '__main__':

    # desired interpolation frequency
    interpolation_freq = 300

    rospy.init_node('casannis_replay', anonymous=True)

    try:
        casannis(interpolation_freq)
    except rospy.ROSInterruptException:
        pass