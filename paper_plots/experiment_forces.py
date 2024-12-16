#!/usr/bin/env python2

import pickle
import matplotlib.pyplot as plt 
import matplotlib 
import numpy as np
import matplotlib.gridspec as gridspec


if __name__ == "__main__":

    # get data from stepup experiment saved in yiannis_centauro_pytools module
    with open('/home/idadiotis/ocs2_ws/src/yiannis_centauro_pytools/txt/stepup_forces.txt', 'rb') as f:
        estimated_forces = pickle.load(f)
    # momentum_based estimated forces
    with open('/home/idadiotis/ocs2_ws/src/yiannis_centauro_pytools/txt/stepup_momentumbased_forces.txt', 'rb') as f:
        momentumbased_estimated_forces = pickle.load(f)

    swing_timings = {'1': [], '2': [], '3': [], '4': []}
    trj_start = [0.0, 17.8, 35.8, 55.0, 63.5, 80.0]
    force_sol = []
    planned_time_trj = []
    for i, name in enumerate(['1', '2', '3', '4', '5']):
        with open('/home/idadiotis/centauro_ws/src/casannis_walking/paper_plots/stepup' + name + '.txt', 'rb') as f:
            solution = pickle.load(f)
            force_sol += solution['F']
        if i == 0 or i == 1 or i == 2:
            dt = 13.0
            planned_time_trj += np.linspace(trj_start[i], trj_start[i] + dt, int(dt/0.2) + 1).tolist()

            # create swing  timings to plot them
            for l in range(1, 5):
                leg_swing_start = trj_start[i] + l * 1.0 + (l - 1) * 2
                swing_timings[str(l)].append([leg_swing_start, leg_swing_start + 2])
        elif i == 3:
            dt = 4.0
            planned_time_trj += np.linspace(trj_start[i], trj_start[i] + dt, int(dt/0.2) + 1).tolist()

            # create swing  timings to plot them
            for l in range(3, 4):
                leg_swing_start = trj_start[i] + 1.0
                swing_timings[str(l)].append([leg_swing_start, leg_swing_start + 2])
        else:
            dt = 9.0
            planned_time_trj += np.linspace(trj_start[i], trj_start[i] + dt, int(dt/0.2) + 1).tolist()

            # create swing  timings to plot them
            for l in range(4, 5):
                leg_swing_start = trj_start[i] + 3.0
                swing_timings[str(l)].append([leg_swing_start, leg_swing_start + 5])

    leg_number = 4
    linestyles = ['-', '--']
    colorstyles = ['g', 'r', 'b']
    linewidths = [3.5, 4, 3.5]
    leg_names = ['FL', 'FR', 'HL', 'HR']
    arm_names = ['Left', 'Right']
    coords = ['X', 'Y', 'Z']
    track_labels = ['measured', 'reference']
    end_estimation = 73.0
    time_trj = np.linspace(0.0, end_estimation, 1000 * end_estimation)

    # planned force plot
    horizont = np.linspace(0.0, len(force_sol)/3)
    plt.figure()
    for i, name in enumerate(leg_names):
        plt.subplot(4, 1, i + 1)
        for k in range(3):
            plt.plot(planned_time_trj, force_sol[3 * i + k::12], '-', color=colorstyles[k], linewidth=3)
        plt.grid()
        plt.title(name)
        plt.legend([str(name) + '_x', str(name) + '_y', str(name) + '_z'])
    plt.xlabel('Time [s]')

    fontsize_axes =  40
    fontsize_legends = 0.6 * 32
    linewidths = 4
    ticksize =28
    # planned vs estimated force
    for j, compon in enumerate(coords):
        # smaller figure
        fig = plt.figure(figsize=(15, 10))
        # Create a GridSpec object with 3 rows and 2 columns
        gs = gridspec.GridSpec(4, 1)  # Adjust height_ratios as needed
        # Create the subplots
        ax1 = plt.subplot(gs[0, 0])  
        ax2 = plt.subplot(gs[1, 0])  
        ax3 = plt.subplot(gs[2, 0])  
        ax4 = plt.subplot(gs[3, 0])
        ax = [ax1, ax2, ax3, ax4]
        for i in range(leg_number):     # loop over legs
            ax[i].plot(time_trj, [filt_value[j] for filt_value in estimated_forces[i]][:int(end_estimation*1000)],      # filtered
                     color='r', label='estimated', linewidth=linewidths)
            # plt.plot(time_trj, [filt_value[j] for filt_value in momentumbased_estimated_forces[i]][:int(end_estimation*1000)],      # momentum based
            #          ':', color='g', label='est', linewidth=1)
            ax[i].plot(planned_time_trj, force_sol[3 * i + j::12], '-', color='b', label='planned', linewidth=linewidths)
            ax[i].set_xlim([0.0, end_estimation+2.0])
            ax[i].set_ylabel(leg_names[i] + ' $[N]$', fontsize=fontsize_axes)
            ax[i].grid(True, axis='both', which='major', linestyle='--', alpha=0.7)
            ax[i].tick_params(axis='both', labelsize=ticksize)
            for ii in range(len(swing_timings[str(i+1)])):
                ax[i].axvspan(swing_timings[str(i+1)][ii][0], swing_timings[str(i+1)][ii][1], alpha=0.5, color='grey')
            if i < leg_number - 1:
                # ax[i].set_xticks([])  # Remove x-ticks
                # ax[i].xaxis.grid(True)
                ax[i].grid(True)
                ax[i].set_xticklabels([])
            else:
                ax[i].set_xlabel('Time $[s]$', fontsize=fontsize_axes)
        plt.legend(fontsize=fontsize_legends, loc="upper right")
        plt.suptitle('F' + compon + ' estimation', fontsize=fontsize_axes)
    plt.show()