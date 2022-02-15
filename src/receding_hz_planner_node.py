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


def shift_solution(solution, knots_toshift, dims):
    '''
    Generate the new initial guess by shifting the previous solution.
    :param solution: previous solution
    :param knots_toshift: knots to shift the new solution
    :param dims: list with dimensions of the different variables
    :return: the shifted solution which can be used as initial guess
    '''

    new_values = [0]
    # solution_keys = solution.keys()
    solution_based_ordered_keys = ['x', 'u', 'F', 'Pl_mov', 'Pr_mov', 'DPl_mov', 'DPr_mov', 'F_virt_l', 'F_virt_r']

    # shifted_sol = {}
    shifted_sol_array = []
    for keyname in solution_based_ordered_keys:
        shifted_variables = solution[keyname][knots_toshift * dims[keyname]:] + \
                            knots_toshift * solution[keyname][-knots_toshift * dims[keyname]:]  # same as last knot
                            # knots_toshift * dims[keyname] * new_values    # zero new values

        # shifted_sol.update({keyname: shifted_variables})
        shifted_sol_array = np.hstack((shifted_sol_array, shifted_variables))

    # print(shifted_sol)
    # shifted_sol_array = np.array(shifted_sol['x'] +
    #                              shifted_sol['u'] +
    #                              shifted_sol['F'] +
    #                              shifted_sol['Pl_mov'] +
    #                              shifted_sol['Pr_mov'] +
    #                              shifted_sol['DPl_mov'] +
    #                              shifted_sol['DPr_mov'] +
    #                              shifted_sol['F_virt_l'] +
    #                              shifted_sol['F_virt_r'])

    return shifted_sol_array


def casannis(int_freq):

    """
    This function call the optimization problem constructor, the solver, the interpolator and interfaces with cartesio
    through ros topics
    Args:
        int_freq: desired interpolation frequency (affects only the interpolation and not the optimal solution)
    Returns:
        publish the desired data to cartesio through ros topics

    """

    # get inclination of terrain
    inclination_deg = rospy.get_param("~inclination_deg")

    # get param conservative_box which defines if the box constraints are near the chest or not
    arm_box_conservative = rospy.get_param("~box_conservative")

    # select gait among different developments
    forward_arm_config = rospy.get_param("~forward_arms")
    linear_fvirt = rospy.get_param("~linear_fvirt")

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
    swing_id = rospy.get_param("~sw_id")    # from command line as swing_id:=1/2/3/4
    swing_id = swing_id.rstrip(']').lstrip('[').split(',')  # convert swing_id from "[a, b]" to [a,b]
    swing_id = [int(i) for i in swing_id]

    # number of steps
    step_num = len(swing_id)

    # Target position of feet wrt to the current position
    tgt_dx = rospy.get_param("~tgt_dx")    # from command line as swing_id:=1/2/3/4
    tgt_dx = tgt_dx.rstrip(']').lstrip('[').split(',')  # convert swing_id from "[a, b]" to [a,b]
    tgt_dx = [float(i) for i in tgt_dx]

    tgt_dy = rospy.get_param("~tgt_dy")    # from command line as swing_id:=1/2/3/4
    tgt_dy = tgt_dy.rstrip(']').lstrip('[').split(',')  # convert swing_id from "[a, b]" to [a,b]
    tgt_dy = [float(i) for i in tgt_dy]

    tgt_dz = rospy.get_param("~tgt_dz")    # from command line as swing_id:=1/2/3/4
    tgt_dz = tgt_dz.rstrip(']').lstrip('[').split(',')  # convert swing_id from "[a, b]" to [a,b]
    tgt_dz = [float(i) for i in tgt_dz]

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
        print(i)
        # targets
        swing_tgt.append([contacts[swing_id[i] - 1][0] + tgt_dx[i],
                          contacts[swing_id[i] - 1][1] + tgt_dy[i],
                          contacts[swing_id[i] - 1][2] + tgt_dz[i]])

        # swing phases
        swing_t.append(rospy.get_param("~sw_t" + str(i+1)))  # from command line as swing_t:="[a,b]"
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
    payload_m = rospy.get_param("~mass_payl")  # from command line as swing_t:="[a,b]"
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

    # motion plans publisher
    motionplan_pub_ = rospy.Publisher('/PayloadAware/motion_plan', MotionPlan_msg, queue_size=10)

    # Subscriber for contact flags
    rospy.Subscriber('/contacts', Contacts_msg, contacts_callback)

    # object class of the optimization problem
    walk = SelectedGait(mass=112, N=int((swing_t[-1][1] + 1.0) / 0.2), dt=0.2, payload_masses=payload_m,
                        slope_deg=inclination_deg, conservative_box=arm_box_conservative)

    # call the solver of the optimization problem
    sol = walk.solve(x0=x0, contacts=contacts, mov_contact_initial=moving_contact, swing_id=[x-1 for x in swing_id],
                     swing_tgt=swing_tgt, swing_clearance=swing_clear, swing_t=swing_t, min_f=minimum_force)

    plan_msg = MotionPlan_msg()
    plan_msg.state = sol['x']
    plan_msg.control = sol['u']
    plan_msg.left_arm_pos = sol['Pl_mov']
    plan_msg.right_arm_pos = sol['Pr_mov']
    plan_msg.left_arm_vel = sol['DPl_mov']
    plan_msg.right_arm_vel = sol['DPr_mov']
    plan_msg.leg_forces = sol['F']
    plan_msg.left_arm_force = sol['F_virt_l']
    plan_msg.right_arm_force = sol['F_virt_r']

    variables_dim = {
        'x': walk._dimx,
        'u': walk._dimu,
        'Pl_mov': walk._dimp_mov,
        'Pr_mov': walk._dimp_mov,
        'DPl_mov': walk._dimp_mov,
        'DPr_mov': walk._dimp_mov,
        'F': walk._dimf_tot,
        'F_virt_l': walk._dimf,
        'F_virt_r': walk._dimf
    }

    for i in range(1):
        # update arguments of solve function
        x0 = sol['x'][9:18]
        moving_contact = [[np.array(sol['Pl_mov'][3:6]), np.array(sol['DPl_mov'][3:6])],
                          [np.array(sol['Pr_mov'][3:6]), np.array(sol['DPr_mov'][3:6])]]

        swing_t = (np.array(swing_t) - walk._dt).tolist()

        shifted_guess = shift_solution(sol, 1, variables_dim)

        sol = walk.solve(x0=x0, contacts=contacts, mov_contact_initial=moving_contact, swing_id=[x-1 for x in swing_id],
                             swing_tgt=swing_tgt, swing_clearance=swing_clear, swing_t=swing_t, min_f=minimum_force,
                             init_guess=shifted_guess, state_lamult=sol['lam_x'], constr_lamult=sol['lam_g'])

    test_rate = rospy.Rate(100)
    # loop interpolation points to publish on a specified frequency
    while True:

        if not rospy.is_shutdown():
            # plan_msg.header.stamp = rospy.Time.now()
            motionplan_pub_.publish(plan_msg)

        test_rate.sleep()

    # interpolate the trj, pass solution values and interpolation frequency
    interpl = walk.interpolate(sol, swing_contacts, swing_tgt, swing_clear, swing_t, int_freq)

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

    # print the trajectories
    try:
        # there was early contact detected
        if early_contact.index(True) + 1:
            print("Early contact detected. Trj Counter is:", executed_trj, "out of total", N_total-1)

            if rospy.get_param("~plots"):
                walk.print_trj(sol, interpl, int_freq, contacts, [x-1 for x in swing_id], executed_trj)
    except:
        print("No early contact detected")

        if rospy.get_param("~plots"):
            walk.print_trj(sol, interpl, int_freq, contacts, [x-1 for x in swing_id], [N_total-1, N_total-1, N_total-1, N_total-1])


if __name__ == '__main__':

    # desired interpolation frequency
    interpolation_freq = 300

    rospy.init_node('casannis_planner', anonymous=True)

    try:
        casannis(interpolation_freq)
    except rospy.ROSInterruptException:
        pass