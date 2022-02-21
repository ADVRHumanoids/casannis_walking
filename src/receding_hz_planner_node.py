#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from std_msgs.msg import Bool
from casannis_walking.msg import PayloadAware_plans as MotionPlan_msg
from casannis_walking.msg import Pa_interpolated_trj as Trj_msg
import matplotlib.pyplot as plt
import Receding_horizon as rh

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
    swing_id = [int(i)-1 for i in swing_id]

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
    swing_contacts = []         # contact positions of the swing feet

    for i in range(step_num):
        # swing phases
        swing_t.append(rospy.get_param("~sw_t" + str(i+1)))  # from command line as swing_t:="[a,b]"
        swing_t[i] = swing_t[i].rstrip(']').lstrip('[').split(',')  # convert swing_t from "[a, b]" to [a,b]
        swing_t[i] = [float(ii) for ii in swing_t[i]]

        swing_contacts.append(contacts[swing_id[i]])

    swing_tgt = rh.get_swing_targets(swing_id, contacts, [tgt_dx, tgt_dy, tgt_dz])

    # receive weight of payloads
    payload_m = rospy.get_param("~mass_payl")  # from command line as swing_t:="[a,b]"
    payload_m = payload_m.rstrip(']').lstrip('[').split(',')  # convert swing_t from "[a, b]" to [a,b]
    payload_m = [float(i) for i in payload_m]

    print('Payload masses', payload_m)

    # publisher to start replaying
    starting_pub_ = rospy.Publisher('PayloadAware/connection', Bool, queue_size=10)
    start_msg = Bool()
    start_msg.data = True

    # motion plans publisher
    motionplan_pub_ = rospy.Publisher('/PayloadAware/motion_plan', MotionPlan_msg, queue_size=10)
    intertrj_pub_ = rospy.Publisher('/PayloadAware/interpolated_trj', Trj_msg, queue_size=10)

    optim_horizon = swing_t[-1][1] + 1.0
    print('Optimize with horizon: ', optim_horizon)

    # object class of the optimization problem
    nlp_discr = 0.2
    walk = SelectedGait(mass=112, N=int(optim_horizon / nlp_discr), dt=nlp_discr, payload_masses=payload_m,
                        slope_deg=inclination_deg, conservative_box=arm_box_conservative)

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

    # debug some stuff
    print('================================================')
    print('================ Solver inputs =====================')
    print('================================================')
    print('**Initial state:', x0)
    print('**All contacts:', contacts)
    print('**Moving contact:', moving_contact)
    print('==Swing id:', swing_id)
    print('==Swing tgt:', swing_tgt)
    print('==Swing clear:', swing_clear)
    print('==Swing_t:', swing_t)
    print('================================================')
    print('================================================')
    print(swing_contacts)

    # call the solver of the optimization problem
    sol = walk.solve(x0=x0, contacts=contacts, mov_contact_initial=moving_contact, swing_id=swing_id,
                     swing_tgt=swing_tgt, swing_clearance=swing_clear, swing_t=swing_t, min_f=minimum_force)
    interpl = walk.interpolate(sol, swing_contacts, swing_tgt, swing_clear, swing_t, int_freq)

    # receding time of the horizon
    knots_shift = 3
    horizon_shift = knots_shift * nlp_discr

    # set fields of the motion plan message
    sw_leg_tostring = [['fl_leg_pos_x', 'fl_leg_pos_y', 'fl_leg_pos_z'],
                       ['fr_leg_pos_x', 'fr_leg_pos_y', 'fr_leg_pos_z'],
                       ['hl_leg_pos_x', 'hl_leg_pos_y', 'hl_leg_pos_z'],
                       ['hr_leg_pos_x', 'hr_leg_pos_y', 'hr_leg_pos_z']]

    sw_arm_tostring = [['left_arm_pos_x', 'left_arm_pos_y', 'left_arm_pos_z'],
                       ['right_arm_pos_x', 'right_arm_pos_y', 'right_arm_pos_z']]

    com_tostring = ['com_pos_x', 'com_pos_y', 'com_pos_z']

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

    # interpolated trj message
    intertrj_msg = Trj_msg()
    intertrj_msg.horizon_shift = horizon_shift
    intertrj_msg.horizon_dur = optim_horizon
    intertrj_msg.swing_t = [a for k in swing_t for a in k]
    intertrj_msg.time = interpl['t']
    intertrj_msg.swing_id = swing_id
    for j, coord_name in enumerate(['x', 'y', 'z']):
        setattr(intertrj_msg, com_tostring[j], interpl['x'][j])     # com

        for i, arm_name in enumerate(['p_mov_l', 'p_mov_r']):       # arms
            setattr(intertrj_msg, sw_arm_tostring[i][j], interpl[arm_name][j])

        for i in range(len(swing_id)):                              # legs
            setattr(intertrj_msg, sw_leg_tostring[swing_id[i]][j], interpl['sw'][i][coord_name])

    motionplan_pub_.publish(plan_msg)  # publish plan
    intertrj_pub_.publish(intertrj_msg)  # publish trj
    starting_pub_.publish(start_msg)    # publish to start replay

    solutions_counter = 1   # counter of solutions acquired
    # for i in range(20):
    while True:
        # start of the next horizon wrt to initial time
        start_of_next_horizon = solutions_counter * horizon_shift

        difference = start_of_next_horizon - optim_horizon
        if difference > 0.0:
            solutions_counter = 1
            start_of_next_horizon = solutions_counter * horizon_shift

        # print('________', knots_shift*walk._dimx , (knots_shift + 1)*walk._dimx)
        # update arguments of solve function
        x0 = sol['x'][knots_shift*walk._dimx : (knots_shift + 1)*walk._dimx]
        moving_contact = [[np.array(sol['Pl_mov'][knots_shift*walk._dimp_mov : (knots_shift + 1)*walk._dimp_mov]),
                           np.array(sol['DPl_mov'][knots_shift*walk._dimp_mov : (knots_shift + 1)*walk._dimp_mov])],
                          [np.array(sol['Pr_mov'][knots_shift*walk._dimp_mov : (knots_shift + 1)*walk._dimp_mov]),
                           np.array(sol['DPr_mov'][knots_shift*walk._dimp_mov : (knots_shift + 1)*walk._dimp_mov])]]

        # swing contacts based on previous plan at the desired time (start of next planning horizon)
        prev_swing_leg_pos = rh.get_current_leg_pos(interpl['sw'], swing_id, start_of_next_horizon, 300)
        for i in swing_id:
            contacts[i] = prev_swing_leg_pos[swing_id.index(i)]

        prev_swing_t = swing_t      # save old swing_t and swing_id
        prev_swing_id = swing_id

        # debug some stuff
        # print('**Initial state:', x0)
        # print('**Moving contact:', moving_contact)
        # print('**Initial contacts:', contacts)
        # print('**All contacts:', all_contacts)

        print('@@@@@@@@@ prev_swing_t', prev_swing_t)
        print('@@@@@@@@@ prev_swing_id', prev_swing_id)

        # new swing_t and swing_id for next optimization
        swing_t, swing_id, another_step = rh.get_swing_durations(prev_swing_t, prev_swing_id, [2, 0, 3, 1],
                                                                 horizon_shift, optim_horizon)
        # debug some stuff
        print('======', prev_swing_id)
        print('====== New Swing timings:', swing_t)
        print('====== New Swing id:', swing_id)
        print('====== Another step:', another_step)

        # form position of swing legs for next optimization
        if another_step[0] is True:
            next_swing_leg_pos = prev_swing_leg_pos + [np.array(contacts[swing_id[-1]])]
        else:
            next_swing_leg_pos = prev_swing_leg_pos

        if another_step[1] is True:
            next_swing_leg_pos = next_swing_leg_pos[1:]

        # # debug some stuff
        # print('!!!!Prev swing leg pos:', prev_swing_leg_pos)
        # print('!!!!Next swing leg pos:', next_swing_leg_pos)

        # get initial guess
        shifted_guess = rh.shift_solution(sol, 1, variables_dim)

        # update tgt_dx heuristically
        new_step_num = len(swing_id)
        tgt_dx = [tgt_dx[0]] * new_step_num
        tgt_dy = [tgt_dy[0]] * new_step_num
        tgt_dz = [tgt_dz[0]] * new_step_num
        print('!!!!', tgt_dx, tgt_dy, tgt_dz)

        # get target positions fot the swing legs
        swing_tgt = rh.get_swing_targets(swing_id, contacts, [tgt_dx, tgt_dy, tgt_dz])

        # debug some stuff
        print('================================================')
        print('================ Solver inputs =====================')
        print('================================================')
        print('**Initial state:', x0)
        print('**All contacts:', contacts)
        print('**Moving contact:', moving_contact)
        print('==Swing id:', swing_id)
        print('==Swing tgt:', swing_tgt)
        print('==Swing clear:', swing_clear)
        print('==Swing_t:', swing_t)
        print('================================================')
        print('================================================')

        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print('^^^^^^^^^^^^^^^^ Targets ^^^^^^^^^^^^^^^^^^^')
        print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        print('Swing tgt:', swing_tgt)
        print('Contacts:', contacts)
        # print('Next swing leg position', next_swing_leg_pos)

        new_nlp_params = rh.get_updated_nlp_params(walk._P, knots_shift, another_step, swing_id, swing_t,
                                                   swing_tgt, contacts, swing_clear, optim_horizon)

        print('PPPPPPPPPPPPPPPPPPPP: ', new_nlp_params[0][0:6])

        sol = walk.solve(x0=x0, contacts=contacts, mov_contact_initial=moving_contact, swing_id=swing_id,
                             swing_tgt=swing_tgt, swing_clearance=swing_clear, swing_t=swing_t, min_f=minimum_force,
                             init_guess=shifted_guess, state_lamult=sol['lam_x'], constr_lamult=sol['lam_g'],
                             nlp_params=new_nlp_params)

        # # debug force plot
        # tt = np.linspace(0.0, (swing_t[-1][1] + 1.0), walk._knot_number)
        # plt.figure()
        # for i, name in enumerate(['fl', 'fr', 'hl', 'hr']):
        #     plt.subplot(2, 2, i + 1)
        #     for k in range(3):
        #         plt.plot(tt, sol['F'][3 * i + k::12], '.-')
        #     plt.grid()
        #     plt.title(name)
        #     plt.legend([str(name) + '_x', str(name) + '_y', str(name) + '_z'])
        # plt.xlabel('Time [s]')
        # plt.show()

        print(next_swing_leg_pos)
        interpl = walk.interpolate(sol, next_swing_leg_pos, swing_tgt, swing_clear, swing_t, int_freq,
                                   feet_ee_swing_trj=interpl['sw'])
        # walk.print_trj(sol, interpl, int_freq, contacts, swing_id)
        # print('&&&&&', len(interpl['sw']))
        solutions_counter += 1

        # set fields of the message
        plan_msg.state = sol['x']
        plan_msg.control = sol['u']
        plan_msg.left_arm_pos = sol['Pl_mov']
        plan_msg.right_arm_pos = sol['Pr_mov']
        plan_msg.left_arm_vel = sol['DPl_mov']
        plan_msg.right_arm_vel = sol['DPr_mov']
        plan_msg.leg_forces = sol['F']
        plan_msg.left_arm_force = sol['F_virt_l']
        plan_msg.right_arm_force = sol['F_virt_r']

        # interpolated trj message
        intertrj_msg.time = interpl['t']
        intertrj_msg.horizon_shift = horizon_shift
        intertrj_msg.horizon_dur = optim_horizon
        intertrj_msg.swing_t = [a for k in swing_t for a in k]
        intertrj_msg.swing_id = swing_id
        for j, coord_name in enumerate(['x', 'y', 'z']):
            setattr(intertrj_msg, com_tostring[j], interpl['x'][j])     # com

            for i, arm_name in enumerate(['p_mov_l', 'p_mov_r']):       # arms
                setattr(intertrj_msg, sw_arm_tostring[i][j], interpl[arm_name][j])

            for i in range(len(swing_id)):                              # legs
                setattr(intertrj_msg, sw_leg_tostring[swing_id[i]][j], interpl['sw'][i][coord_name])

        motionplan_pub_.publish(plan_msg)       # publish
        intertrj_pub_.publish(intertrj_msg)  # publish trj

        solutions_counter += 1

if __name__ == '__main__':

    # desired interpolation frequency
    interpolation_freq = 300

    rospy.init_node('casannis_planner', anonymous=True)

    try:
        casannis(interpolation_freq)
    except rospy.ROSInterruptException:
        pass