import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


if __name__ == "__main__":

    # get data from stepup experiment saved in yiannis_centauro_pytools module
    with open('/home/ioannis/centauro_ws/src/yiannis_centauro_pytools/txt/stepup_forces.txt', 'rb') as f:
        estimated_forces = pickle.load(f)
    swing_timings = {'1': [], '2': [], '3': [], '4': []}
    trj_start = [0.0, 17.8, 35.8, 55.0, 63.5, 80.0]
    force_sol = []
    planned_time_trj = []
    for i, name in enumerate(['1', '2', '3', '4', '5']):
        with open('../paper_plots/stepup' + name + '.txt', 'rb') as f:
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

    # planned vs estimated force
    for j, compon in enumerate(coords):
        plt.figure()
        matplotlib.rc('xtick', labelsize=16)
        matplotlib.rc('ytick', labelsize=16)
        for i in range(leg_number):     # loop over legs
            plt.subplot(4, 1, i+1)
            plt.plot(time_trj, [filt_value[j] for filt_value in estimated_forces[i]][:int(end_estimation*1000)],      # filtered
                     color='r', label='est', linewidth=2)
            plt.plot(planned_time_trj, force_sol[3 * i + j::12], '-', color='b', label='plan', linewidth=3)
            plt.xlabel('Time $[s]$', fontsize=20)
            plt.xlim([0.0, end_estimation+2.0])
            plt.ylabel(leg_names[i] + ' force $[N]$', fontsize=20)
            plt.grid()
            for ii in range(len(swing_timings[str(i+1)])):
                plt.axvspan(swing_timings[str(i+1)][ii][0], swing_timings[str(i+1)][ii][1], alpha=0.2)
        plt.legend(fontsize=15)
        plt.suptitle('F' + compon + ' estimation', fontsize=20)
    plt.show()