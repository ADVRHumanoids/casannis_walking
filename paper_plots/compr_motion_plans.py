import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


if __name__ == "__main__":

    # get data from stepup experiment saved in yiannis_centauro_pytools module
    # # sc 1
    # with open('/home/ioannis/Desktop/analytical_cost_effect/interpl_analytical_sc1.txt', 'rb') as f:
    #     analytical_plans = pickle.load(f)
    #
    # with open('/home/ioannis/Desktop/analytical_cost_effect/interpl_compatible_sc1.txt', 'rb') as f:
    #     compatible_plans = pickle.load(f)
    #
    # with open('/home/ioannis/Desktop/analytical_cost_effect/interpl_nocost_sc1.txt', 'rb') as f:
    #     nocost_plans = pickle.load(f)

    # sc 2
    with open('/home/ioannis/Desktop/analytical_cost_effect/interpl_analytical_sc2.txt', 'rb') as f:
        analytical_plans2 = pickle.load(f)

    with open('/home/ioannis/Desktop/analytical_cost_effect/interpl_compatible_sc2.txt', 'rb') as f:
        compatible_plans2 = pickle.load(f)

    with open('/home/ioannis/Desktop/analytical_cost_effect/interpl_nocost_sc2.txt', 'rb') as f:
        nocost_plans2 = pickle.load(f)

    with open('/home/ioannis/Desktop/analytical_cost_effect/interpl_noconstraint_sc2.txt', 'rb') as f:
        noconstraint_plans2 = pickle.load(f)

    # # sc 2 dynamic
    # with open('/home/ioannis/Desktop/analytical_cost_effect/interpl_analytical_sc2_dyn.txt', 'rb') as f:
    #     analytical_plans2dyn = pickle.load(f)
    #
    # with open('/home/ioannis/Desktop/analytical_cost_effect/interpl_compatible_sc2_dyn.txt', 'rb') as f:
    #     compatible_plans2dyn = pickle.load(f)
    #
    # with open('/home/ioannis/Desktop/analytical_cost_effect/interpl_nocost_sc2_dyn.txt', 'rb') as f:
    #     nocost_plans2dyn = pickle.load(f)

    # time = analytical_plans[u't']
    time2 = analytical_plans2[u't']
    # time2dyn = analytical_plans2dyn[u't']

    # plans = [nocost_plans, compatible_plans, analytical_plans]
    # plans_num = len(plans)

    plans2 = [noconstraint_plans2, nocost_plans2, analytical_plans2]
    plans_num2 = len(plans2)

    # plans2dyn = [nocost_plans2dyn, compatible_plans2dyn, analytical_plans2dyn]
    # plans_num2dyn = len(plans2dyn)

    leg_number = 4
    linestyles = ['-', '--', ':']
    colorstyles = ['b', 'g', 'r']
    linewidths = [5.0, 5.0, 5.0]
    arm_names = ['Left', 'Right']
    coords = ['X', 'Y', 'Z']

    plt_quantity_strings = [u'ddp_mov_l', u'ddp_mov_r']
    num_plt_quantity = len(plt_quantity_strings)

    # sc1
    # plt.figure()
    # matplotlib.rc('xtick', labelsize=16)
    # matplotlib.rc('ytick', labelsize=16)
    # for k, quantity_name in enumerate(plt_quantity_strings):
    #     for i in range(3):
    #         plt.subplot(2, 3, 3*k+i+1)
    #         for j in range(plans_num):
    #             plt.plot(time, plans[j][quantity_name][i], linestyles[j], linewidth=linewidths[j])
    #         plt.grid()
    #         if k == 2:
    #             plt.xlabel('Time [s]', fontsize=20)
    #         plt.ylabel(coords[i]+' $[m/s^2]$', fontsize=20)
    # plt.legend(['analytical', 'compatible', 'no cost'], fontsize=15)
    # plt.suptitle('Arm Accelerations Sc. 1', fontsize=20)

    #sc2
    plt.figure()
    for k, quantity_name in enumerate(plt_quantity_strings):
        for i in range(3):
            plt.subplot(2, 3, 3*k+i+1)
            for j in range(plans_num2):
                plt.plot(time2, plans2[j][quantity_name][i], linewidth=linewidths[j], color=colorstyles[j])
            plt.grid()
            if k == 1:
                plt.xlabel('Time [s]', fontsize=20)
            plt.ylabel(coords[i]+' $[m/s^2]$', fontsize=20)
        #plt.legend(['no constraint', 'constraint', 'constraint + an. cost'], fontsize=20)
    plt.suptitle('Arm Accelerations Sc. 2', fontsize=20)

    # #sc2 dynamic
    # plt.figure()
    # for k, quantity_name in enumerate(plt_quantity_strings):
    #     for i in range(3):
    #         plt.subplot(2, 3, 3*k+i+1)
    #         for j in range(plans_num2dyn):
    #             plt.plot(time2dyn, plans2dyn[j][quantity_name][i], linewidth=linewidths[j], color=colorstyles[j])
    #         plt.grid()
    #         if k == 1:
    #             plt.xlabel('Time [s]', fontsize=20)
    #         plt.ylabel(coords[i]+' $[m/s^2]$', fontsize=20)
    # plt.legend(['no cost', 'compatible', 'analytical'], fontsize=20)
    # plt.suptitle('Arm Accelerations Sc. 2 dynamic', fontsize=20)

    plt.show()

    print('end of script')

