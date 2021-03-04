#!/usr/bin/env python3

import casadi as cs
import numpy as np
from matplotlib import pyplot as plt
import time
from src.spline_optimization import Spline_optimization_z


class Spline_tests(Spline_optimization_z):

    def solver(self, waypoints_pos, midpoints_pos, ramps, obst):

        # waypoints
        Xl = []  # position lower bounds
        Xu = []  # position upper bounds
        DXl = []  # velocity lower bounds
        DXu = []  # velocity upper bounds
        DDXl = []  # acceleration lower bounds
        DDXu = []  # acceleration upper bounds

        # midpoints
        X_mid_u = []
        X_mid_l = []
        DX_mid_u = []
        DX_mid_l = []

        gl = []  # constraint lower bounds
        gu = []  # constraint upper bounds

        for k in range(self._N):

            # variable bounds
            # trj phases
            is_ramp = (0 < k < ramps)   # excluding initial
            is_obstacle = (ramps <= k < ramps + obst)
            is_obstacle_max = (k == int(ramps + obst / 2))
            is_landing = (ramps + obst <= k < self._N - 1)  # excluding final
            is_start_slow_down = (k == ramps + obst)

            # position bounds

            if k == 0 or k == self._N - 1:  # initial/target position
                x_max = waypoints_pos[k]
                x_min = waypoints_pos[k]

            elif is_ramp:   # ramp part
                print('ramp', k)
                x_max = cs.inf
                x_min = waypoints_pos[0]

            elif is_obstacle:   # main obstacle - clearance part

                if is_obstacle_max:     # maximum clearance
                    print('is obstacle max', k)
                    x_max = waypoints_pos[k]
                    x_min = waypoints_pos[k]

                else:   # lower than maximum clearance
                    print('obstacle', k)
                    x_max = waypoints_pos[k]
                    x_min = -cs.inf

            elif is_landing:    # landing part

                if is_start_slow_down:  # first point
                    x_max = waypoints_pos[k]
                    x_min = waypoints_pos[k]
                else:
                    x_max = cs.inf
                    x_min = waypoints_pos[-1]

            else:
                x_max = cs.inf
                x_min = -cs.inf

            Xu.append(x_max)
            Xl.append(x_min)

            # velocity bounds

            # start, end velocity
            if k == 0 or k == self._N - 1:
                dx_max = 0.0
                dx_min = 0.0

            # obstacle max clearance
            elif is_obstacle_max:
                dx_max = 0.0
                dx_min = 0.0

            # landing
            elif is_landing:
                dx_max = cs.inf
                dx_min = - 0.03

            else:
                dx_max = cs.inf
                dx_min = - cs.inf

            DXu.append(dx_max)
            DXl.append(dx_min)

            # acceleration bounds

            # ramp
            if is_ramp:
                ddx_max = cs.inf
                ddx_min = 0.001

            elif is_obstacle:
                ddx_max = 0.0
                ddx_min = - cs.inf

            #elif is_landing:

                #if is_start_slow_down:
                    #print('start slow down', k)
                    #ddx_max = -0.0001
                    #ddx_min = -cs.inf

                #else:
                #ddx_max = cs.inf
                #ddx_min = 0.0

            else:
                ddx_max = cs.inf
                ddx_min = - cs.inf

            DDXu.append(ddx_max)
            DDXl.append(ddx_min)

            # midpoints - variable constraints
            if not k == self._N - 1:

                # position
                if is_ramp:
                    x_mid_max = cs.inf
                    x_mid_min = waypoints_pos[k]
                #elif is_obstacle_max:
                    #x_mid_max = waypoints_pos[k]
                    #x_mid_min = - cs.inf
                #elif is_landing:
                    #x_mid_max = cs.inf
                    #x_mid_min = -cs.inf
                else:
                    x_mid_max = cs.inf
                    x_mid_min = -cs.inf

                X_mid_u.append(x_mid_max)
                X_mid_l.append(x_mid_min)

                # velocity
                if is_obstacle_max:
                    dx_mid_max = -0.000
                    dx_mid_min = - cs.inf
                elif is_landing:
                    if is_start_slow_down:
                        dx_mid_max = 0.0
                        dx_mid_min = -cs.inf
                    else:
                        dx_mid_max = 0.0
                        dx_mid_min = -cs.inf

                else:
                    dx_mid_max = cs.inf
                    dx_mid_min = - cs.inf

                DX_mid_u.append(dx_mid_max)
                DX_mid_l.append(dx_mid_min)

                gl.append(np.zeros(2))
                gu.append(np.zeros(2))

            # constraints pos, vel, acc at waypoints
            gl.append(np.zeros(2))
            gu.append(np.zeros(2))

        # format bounds and params according to solver
        lbv = cs.vertcat(*Xl, *DXl, *DDXl, *X_mid_l, *DX_mid_l)
        ubv = cs.vertcat(*Xu, *DXu, *DDXu, *X_mid_u, *DX_mid_u)
        lbg = cs.vertcat(*gl)
        ubg = cs.vertcat(*gu)

        # call solver
        self._sol = self._solver(lbx=lbv, ubx=ubv, lbg=lbg, ubg=ubg)

        return self._sol

    # ---------------------------------------------------------------------
    def solver2(self, waypoints_pos, midpoints_pos):

        # waypoints
        Xl = []  # position lower bounds
        Xu = []  # position upper bounds
        DXl = []  # velocity lower bounds
        DXu = []  # velocity upper bounds
        DDXl = []  # acceleration lower bounds
        DDXu = []  # acceleration upper bounds

        # midpoints
        X_mid_u = []
        X_mid_l = []
        DX_mid_u = []
        DX_mid_l = []

        gl = []  # constraint lower bounds
        gu = []  # constraint upper bounds

        for k in range(self._N):

            # variable bounds
            # trj phases
            #is_ramp = (0 < k < ramps)   # excluding initial
            #is_obstacle = (ramps <= k < ramps + obst)
            #is_obstacle_max = (k == int(ramps + obst / 2))
            #is_landing = (ramps + obst <= k < self._N - 1)  # excluding final
            #is_start_slow_down = (k == ramps + obst)

            # position bounds
            # start, end position
            if k == 0 or k == self._N - 1:
                print('initial,final', k)
                x_max = waypoints_pos[k]
                x_min = waypoints_pos[k]

            # ramp
            #elif is_ramp:
                #print('ramp', k)
                #x_max = cs.inf  # position[k+1]
                #x_min = position[0]

            # main obstacle - clearance
            #elif is_obstacle:

                #if is_obstacle_max:
                    #print('is obstacle max', k)
                    #x_max = position[k] + 0.02
                    #x_min = position[k]

                #else:
                    #print('obstacle', k)
                    #x_max = position[k]
                    #x_min = position[k] - 0.05

            #elif is_landing:

                #print('landing', k)
                #x_max = cs.inf  # position[k] + 0.02
                #x_min = position[-1]  # position[k+1]

            else:
                x_max = cs.inf
                x_min = waypoints_pos[k]

            Xu.append(x_max)
            Xl.append(x_min)

            # velocity bounds

            # start, end velocity
            if k == 0 or k == self._N - 1:
                dx_max = 0.0
                dx_min = 0.0

            # obstacle max clearance
            #elif is_obstacle_max:
                #dx_max = 0.0
                #dx_min = 0.0

            # landing
            # elif is_start_slow_down:
            # dx_max = 0.0
            # dx_min = - 0.03

            else:
                dx_max = cs.inf
                dx_min = - cs.inf

            DXu.append(dx_max)
            DXl.append(dx_min)

            # acceleration bounds

            # landing
            #if is_ramp:
                #ddx_max = cs.inf
                #ddx_min = 0.2

            #if is_obstacle:
                #ddx_max = 0.0
                #ddx_min = - cs.inf

            #elif is_landing:

                #if is_start_slow_down:
                    #print('start slow down', k)
                    #ddx_max = -0.0001
                    #ddx_min = -cs.inf

                #else:
                    #ddx_max = cs.inf
                    #ddx_min = 0.0

            #else:
            ddx_max = cs.inf
            ddx_min = - cs.inf

            DDXu.append(ddx_max)
            DDXl.append(ddx_min)

            # midpoints - variable constraints
            if not k == self._N - 1:
                x_mid_max = cs.inf
                x_mid_min = -cs.inf #midpoints_pos[k] + 0.05

                X_mid_u.append(x_mid_max)
                X_mid_l.append(x_mid_min)

                dx_mid_max = cs.inf
                dx_mid_min = - cs.inf

                DX_mid_u.append(dx_mid_max)
                DX_mid_l.append(dx_mid_min)

                gl.append(np.zeros(2))
                gu.append(np.zeros(2))

            # constraints pos, vel, acc at waypoints
            gl.append(np.zeros(2))
            gu.append(np.zeros(2))

        # format bounds and params according to solver
        lbv = cs.vertcat(*Xl, *DXl, *DDXl, *X_mid_l, *DX_mid_l)
        ubv = cs.vertcat(*Xu, *DXu, *DDXu, *X_mid_u, *DX_mid_u)
        lbg = cs.vertcat(*gl)
        ubg = cs.vertcat(*gu)

        # call solver
        self._sol = self._solver(lbx=lbv, ubx=ubv, lbg=lbg, ubg=ubg)

        return self._sol


if __name__ == "__main__":

    initial_pos = 0.0
    target_pos = 0.05
    terrain_conf = 0.05
    swing_time = [0.0, 6.0]
    N = 14  # number of waypoints

    ramp_points = 4     # including initial
    ramp_step = 0.005

    obstacle_points = 3
    clearance = 0.1

    if initial_pos >= target_pos:
        max_height = initial_pos + clearance
    else:
        max_height = target_pos + clearance

    slow_down_points = N - ramp_points - obstacle_points

    # times
    duration = swing_time[1] - swing_time[0]

    ramp_time = np.linspace(swing_time[0], 0.05 * duration, ramp_points).tolist()
    obst_time = np.linspace(0.05 * duration, 0.4 * duration, obstacle_points + 1).tolist()
    slow_time = np.linspace(0.4 * duration, swing_time[1], N + 1 - (ramp_points + obstacle_points)).tolist()

    times = ramp_time[: -1] + obst_time[: -1] + slow_time
    dt = [times[i + 1] - times[i] for i in range(N - 1)]

    time_midpoints = [(times[i] + 0.5 * dt[i]) for i in range(N - 1)]

    # Construct waypoints
    waypoints = []

    # ramp
    waypoints.append(initial_pos)
    for i in range(1, ramp_points):
        waypoints.append(initial_pos + i * ramp_step)

    # max clearance - obstacle
    for i in range(obstacle_points):
        waypoints.append(max_height)

    # slow down
    slow_down = np.linspace(target_pos + terrain_conf, target_pos, N - len(waypoints)).tolist()

    for i in range(len(slow_down)):
        waypoints.append(slow_down[i])

    # midpoints poisitons
    midpoints = [waypoints[i] + 0.5 * (waypoints[i + 1] - waypoints[i]) for i in range(N - 1)]

    start = time.time()

    my_object = Spline_tests(N, dt)
    solution = my_object.solver(waypoints, midpoints, ramp_points, obstacle_points)
    splines = my_object.get_splines(solution['x'][0:N], dt)

    end = time.time()

    print('Positions:', solution['x'][0:N])
    print('Velocities:', solution['x'][N:2 * N])
    print('Accelerations:', solution['x'][2 * N:3 * N])
    print('Computation time:', 1000 * (end - start), 'ms')

    # print results
    s = [np.linspace(0, dt[i], 100) for i in range(N - 1)]

    plt.figure()
    plt.plot(times, solution['x'][0:N], "o")
    plt.plot(time_midpoints, solution['x'][3 * N:(4 * N - 1)], ".")
    plt.plot(times, waypoints, "x")
    # plt.plot(time_midpoints, midpoints, ".")
    for i in range(N - 1):
        plt.plot([x + times[i] for x in s[i]], splines['pos'][i](s[i]))
    plt.grid()
    plt.legend(['assigned knots', 'assigned midpoints', 'initial knots'])
    plt.xlabel('time [s]')
    plt.ylabel('z position [m]')

    plt.figure()
    plt.plot(times, solution['x'][N:2 * N], "o")
    plt.plot(time_midpoints, solution['x'][(4 * N - 1):(5 * N - 2)], ".")
    for i in range(N - 1):
        plt.plot([x + times[i] for x in s[i]], splines['vel'][i](s[i]))
    plt.legend(['assigned knots', 'assigned midpoints', 'initial knots'])
    plt.grid()
    plt.xlabel('time [s]')
    plt.ylabel('z velocity [m/s]')

    plt.figure()
    plt.plot(times, solution['x'][2 * N:3 * N], "o")
    for i in range(N - 1):
        plt.plot([x + times[i] for x in s[i]], splines['acc'][i](s[i]))
    plt.plot(times, solution['x'][2 * N:3 * N], "o")
    plt.grid()
    plt.xlabel('time [s]')
    plt.ylabel('z acceleration [m/s^2]')

    plt.show()

    '''initial_pos = 0.0
    target_pos = 0.05
    terrain_conf = 0.05
    clearance = 0.1

    swing_time = [0.0, 6.0]
    N = 16  # number of waypoints

    # time
    duration = swing_time[1] - swing_time[0]
    times = np.linspace(swing_time[0], swing_time[1], N)
    dt = [times[i + 1] - times[i] for i in range(N - 1)]

    time_midpoints = [(times[i] + 0.5 * dt[i]) for i in range(N - 1)]

    # Construct waypoints
    clearance_time = 0.2 * duration
    slow_down_time = 0.4 * duration

    clearance_index = int(clearance_time / dt[0])
    slow_down_index = int(slow_down_time / dt[0])

    waypoints1 = np.linspace(initial_pos, target_pos + clearance, clearance_index)
    waypoints2 = np.linspace(target_pos + clearance, target_pos + terrain_conf, 1 + slow_down_index - clearance_index)
    waypoints3 = np.linspace(target_pos + terrain_conf, target_pos, N - slow_down_index + 1)

    waypoints = waypoints1.tolist() + waypoints2.tolist()[1:] + waypoints3.tolist()[1:]

    midpoints = [waypoints[i] + 0.5 * (waypoints[i + 1] - waypoints[i]) for i in range(N - 1)]

    start = time.time()

    my_object = Spline_tests(N, dt)
    solution = my_object.solver2(waypoints, midpoints)
    splines = my_object.get_splines(solution['x'][0:N], dt)

    end = time.time()

    print('Positions:', solution['x'][0:N])
    print('Velocities:', solution['x'][N:2 * N])
    print('Accelerations:', solution['x'][2 * N:3 * N])
    print('Computation time:', 1000 * (end - start), 'ms')

    # print results
    s = [np.linspace(0, dt[i], 100) for i in range(N - 1)]

    plt.figure()
    plt.plot(times, solution['x'][0:N], "o")
    plt.plot(time_midpoints, solution['x'][3 * N:(4 * N - 1)], ".")
    plt.plot(times, waypoints, "x")
    # plt.plot(time_midpoints, midpoints, ".")
    for i in range(N - 1):
        plt.plot([x + times[i] for x in s[i]], splines['pos'][i](s[i]))
    plt.grid()
    plt.legend(['assigned knots', 'assigned midpoints', 'initial knots'])
    plt.xlabel('time [s]')
    plt.ylabel('z position [m]')

    plt.figure()
    plt.plot(times, solution['x'][N:2 * N], "o")
    plt.plot(time_midpoints, solution['x'][(4 * N - 1):(5 * N - 2)], ".")
    for i in range(N - 1):
        plt.plot([x + times[i] for x in s[i]], splines['vel'][i](s[i]))
    plt.legend(['assigned knots', 'assigned midpoints', 'initial knots'])
    plt.grid()
    plt.xlabel('time [s]')
    plt.ylabel('z velocity [m/s]')

    plt.figure()
    plt.plot(times, solution['x'][2 * N:3 * N], "o")
    for i in range(N - 1):
        plt.plot([x + times[i] for x in s[i]], splines['acc'][i](s[i]))
    plt.plot(times, solution['x'][2 * N:3 * N], "o")
    plt.grid()
    plt.xlabel('time [s]')
    plt.ylabel('z acceleration [m/s^2]')

    plt.show()'''
