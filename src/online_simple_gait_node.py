#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from std_msgs.msg import Bool
from casannis_walking.msg import PayloadAware_plans as MotionPlan_msg
from casannis_walking.msg import Pa_interpolated_trj as Trj_msg
import matplotlib.pyplot as plt
from Receding_horizon import Receding_hz_handler as Receding
import Receding_horizon as rh
import time

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


def start_replan_callback(msg):
    global start_replanning
    start_replanning = msg.data


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

    # select gait among different development
    from gait import Gait as SelectedGait

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
    desired_gait = rospy.get_param("~des_gait")
    swing_id = swing_id.rstrip(']').lstrip('[').split(',')  # convert swing_id from "[a, b]" to [a,b]
    desired_gait = desired_gait.rstrip(']').lstrip('[').split(',')  # convert swing_id from "[a, b]" to [a,b]
    swing_id = [int(i)-1 for i in swing_id]
    desired_gait = [int(i)-1 for i in desired_gait]

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

    # construct stride list
    stride = [tgt_dx[0], tgt_dy[0], tgt_dz[0]]

    # Clearance to be achieved, counted from the highest point
    swing_clear = rospy.get_param("~clear")  # get from command line as target_dx

    # force threshold
    minimum_force = rospy.get_param("~min_for")

    # apply or no contact detection
    cont_detection = rospy.get_param("~cont_det")  # from command line as contact_det:=True/False

    # variables to loop for swing legs
    swing_t = []    # time periods of the swing phases
    swing_contacts = []         # contact positions of the swing feet

    for i in range(step_num):
        # swing phases
        swing_t.append(rospy.get_param("~sw_t" + str(i+1)))  # from command line as swing_t:="[a,b]"
        swing_t[i] = swing_t[i].rstrip(']').lstrip('[').split(',')  # convert swing_t from "[a, b]" to [a,b]
        swing_t[i] = [float(ii) for ii in swing_t[i]]

        swing_contacts.append(contacts[swing_id[i]])

    default_swing_dur = swing_t[0][1] - swing_t[0][0]
    default_stance_dur = swing_t[0][0]
    swing_tgt = rh.get_swing_targets(swing_id, contacts, [tgt_dx, tgt_dy, tgt_dz])

    # publisher to start replaying
    starting_pub_ = rospy.Publisher('PayloadAware/connection', Bool, queue_size=10)
    start_msg = Bool()
    start_msg.data = True

    # subscriber to wait for message before replanning
    start_replan_sub_ = rospy.Subscriber('/PayloadAware/connection', Bool, start_replan_callback)

    # motion plans publisher
    # motionplan_pub_ = rospy.Publisher('/PayloadAware/motion_plan', MotionPlan_msg, queue_size=10)
    intertrj_pub_ = rospy.Publisher('/PayloadAware/interpolated_trj', Trj_msg, queue_size=10)

    optim_horizon = swing_t[-1][1] + 1.0
    print('Optimize with horizon: ', optim_horizon)

    # receding time of the horizon
    nlp_discr = 0.2
    knots_shift = int(rospy.get_param("~shifted_knots"))

    horizon_shift = knots_shift * nlp_discr

    # handler
    mpc = Receding(horizon=optim_horizon, knots_toshift=knots_shift, nlp_dt=nlp_discr, desired_gait=desired_gait,
                   swing_dur=default_swing_dur, stance_dur=default_stance_dur, interpolation_freq=int_freq)

    # object class of the optimization problem
    walk = SelectedGait(mass=112, N=int(optim_horizon / nlp_discr), dt=nlp_discr, slope_deg=inclination_deg)

    variables_dim = {
        'x': walk._dimx,
        'u': walk._dimu,
        'F': walk._dimf_tot,
    }

    # call the solver of the optimization problem
    sol_previous = walk.solve(x0=x0, contacts=contacts, swing_id=swing_id, swing_tgt=swing_tgt,
                              swing_clearance=swing_clear, swing_t=swing_t, min_f=minimum_force)

    interpl_previous = walk.interpolate(sol_previous, swing_contacts, swing_tgt, swing_clear, swing_t, int_freq,
                                        swing_default_dur=default_swing_dur)

    # set fields of the motion plan message
    sw_leg_tostring = [['fl_leg_pos_x', 'fl_leg_pos_y', 'fl_leg_pos_z'],
                       ['fr_leg_pos_x', 'fr_leg_pos_y', 'fr_leg_pos_z'],
                       ['hl_leg_pos_x', 'hl_leg_pos_y', 'hl_leg_pos_z'],
                       ['hr_leg_pos_x', 'hr_leg_pos_y', 'hr_leg_pos_z']]

    com_tostring = ['com_pos_x', 'com_pos_y', 'com_pos_z']

    # plan_msg = MotionPlan_msg()
    # plan_msg.state = sol_previous['x']
    # plan_msg.control = sol_previous['u']
    # plan_msg.left_arm_pos = sol_previous['Pl_mov']
    # plan_msg.right_arm_pos = sol_previous['Pr_mov']
    # plan_msg.left_arm_vel = sol_previous['DPl_mov']
    # plan_msg.right_arm_vel = sol_previous['DPr_mov']
    # plan_msg.leg_forces = sol_previous['F']
    # plan_msg.left_arm_force = sol_previous['F_virt_l']
    # plan_msg.right_arm_force = sol_previous['F_virt_r']

    # interpolated trj message
    intertrj_msg = Trj_msg()
    intertrj_msg.id = 0
    intertrj_msg.horizon_shift = horizon_shift
    intertrj_msg.horizon_dur = optim_horizon
    intertrj_msg.swing_t = [a for k in swing_t for a in k]
    intertrj_msg.time = interpl_previous['t']
    intertrj_msg.swing_id = swing_id
    for j, coord_name in enumerate(['x', 'y', 'z']):
        setattr(intertrj_msg, com_tostring[j], interpl_previous['x'][j])     # com

        for i in range(len(swing_id)):                              # legs
            setattr(intertrj_msg, sw_leg_tostring[swing_id[i]][j], interpl_previous['sw'][i][coord_name])

    mpc.set_current_contacts(contacts)
    mpc.set_current_swing_tgt(swing_tgt)
    mpc.set_previous_solution(sol_previous)
    mpc.set_previous_interpolated_solution(interpl_previous)
    mpc.set_swing_durations(swing_t, swing_id)
    mpc.count_optimizations(1)

    # motionplan_pub_.publish(plan_msg)  # publish plan
    intertrj_pub_.publish(intertrj_msg)  # publish trj
    starting_pub_.publish(start_msg)    # publish to start replay
    global start_replanning
    start_replanning = False

    # for i in range(20):
    while True:
        # begin1 = time.time()
        # get shifted com and arm ee positions
        shifted_com_state = mpc.get_variable_after_knots_toshift(key_var='x', dimension_var=9)

        # new swing_t and swing_id for next optimization
        swing_t, swing_id, another_step = mpc.get_next_swing_durations(stride)

        # get initial guess
        shifted_guess = mpc.get_shifted_solution(variables_dim, arms=False)

        # get target positions fot the swing legs
        # swing_tgt = mpc.get_swing_targets(contacts, [tgt_dx, tgt_dy, tgt_dz])

        # old_nlp_params = walk._P
        new_nlp_params = mpc.get_updated_nlp_params(walk._P, swing_clear)

        # # update contacts
        # # contacts = [np.array(new_nlp_params[knots_shift][3*i:3*(i+1)]) for i in range(4)]
        # nlp_params_extension = new_nlp_params[-3:]

        # access previous solution
        sol_previous = mpc.get_previous_solution()
        interpl_previous = mpc.get_previous_interpolated_solution()

        # debug some stuff
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        print('@@@@@@@@@ prev_swing_t', mpc._prev_swing_t)
        print('@@@@@@@@@ new swing_t', mpc._swing_t)
        print('@@@@@@@@@ prev_swing_id', mpc._prev_swing_id)
        print('@@@@@@@@@ new swing_id', mpc._swing_id)
        print('@@@@@@@@@ Another step:', another_step)
        print('@@@@@@@@@ Contacts:', mpc._contacts)
        print('@@@@@@@@@ Swing tgt:', mpc._swing_tgt)
        # for i in range(knots_shift):
        #     print('@@@@@@@@@ New nlp params:', nlp_params_extension[i])
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

        # shift state langrange multipliers
        shift_lamultx = mpc.get_shifted_variable(sol_previous['lam_x'], int(walk._nvars / mpc._knot_number))

        # end1 = time.time()
        # print('*****Updating time (msec):', 10e3 * (end1 - begin1))

        # begin1 = time.time()
        # wait until you receive a message from replayer
        # while start_replanning is False:
        #     continue
        # start_replanning = False

        sol = walk.solve(x0=shifted_com_state, contacts=mpc._contacts, swing_id=mpc._swing_id,
                         swing_tgt=mpc._swing_tgt, swing_clearance=swing_clear, swing_t=mpc._swing_t, min_f=minimum_force,
                         init_guess=shifted_guess, state_lamult=shift_lamultx, constr_lamult=sol_previous['lam_g'],
                         nlp_params=new_nlp_params)
        # end1 = time.time()
        # print('*****Solution time (msec):', 10e3*(end1-begin1))

        # print(next_swing_leg_pos)
        interpl = walk.interpolate(sol, [mpc._contacts[ii] for ii in mpc._swing_id], mpc._swing_tgt, swing_clear,
                                   mpc._swing_t, int_freq, feet_ee_swing_trj=interpl_previous['sw'],
                                   shift_time=mpc._time_shifting, swing_default_dur=default_swing_dur,
                                   skip_useless=True)

        # # set fields of the message
        # plan_msg.state = sol['x']
        # plan_msg.control = sol['u']
        # plan_msg.left_arm_pos = sol['Pl_mov']
        # plan_msg.right_arm_pos = sol['Pr_mov']
        # plan_msg.left_arm_vel = sol['DPl_mov']
        # plan_msg.right_arm_vel = sol['DPr_mov']
        # plan_msg.leg_forces = sol['F']
        # plan_msg.left_arm_force = sol['F_virt_l']
        # plan_msg.right_arm_force = sol['F_virt_r']

        # interpolated trj message
        intertrj_msg.time = interpl['t']
        intertrj_msg.id = mpc.count_optimizations(1)
        intertrj_msg.horizon_shift = horizon_shift
        intertrj_msg.horizon_dur = optim_horizon
        intertrj_msg.swing_t = [a for k in swing_t for a in k]
        intertrj_msg.swing_id = swing_id
        for j, coord_name in enumerate(['x', 'y', 'z']):
            setattr(intertrj_msg, com_tostring[j], interpl['x'][j])     # com

            for i in range(len(swing_id)):                              # legs
                setattr(intertrj_msg, sw_leg_tostring[swing_id[i]][j], interpl['sw'][i][coord_name])

        # motionplan_pub_.publish(plan_msg)       # publish
        intertrj_pub_.publish(intertrj_msg)  # publish trj

        # set to general variables
        mpc.set_previous_solution(sol)
        mpc.set_previous_interpolated_solution(interpl)


if __name__ == '__main__':

    # desired interpolation frequency
    interpolation_freq = 300

    rospy.init_node('casannis_planner', anonymous=True)

    try:
        casannis(interpolation_freq)
    except rospy.ROSInterruptException:
        pass