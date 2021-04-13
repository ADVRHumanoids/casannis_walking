import casadi as cs
import numpy as np
from matplotlib import pyplot as plt
import trj_interpolation as interpol


class Walking:
    """
    Assumptions:
      1) mass concentrated at com
      2) zero angular momentum
      3) point contacts

    Dynamics:
      1) input is com jerk
      2) dynamics is a triple integrator of com jerk
      3) there must be contact forces that
        - realize the motion
        - fulfil contact constraints (i.e. unilateral constraint)
    """

    def __init__(self, mass, N, dt):
        """Walking class constructor

        Args:
            mass (float): robot mass
            N (int): horizon length
            dt (float): discretization step
        """

        self._N = N 
        self._dt = dt   # dt used for optimization knots
        self._mass = mass
        self._time = [(i * dt) for i in range(self._N)]

        gravity = np.array([0, 0, -9.81])

        # define dimensions
        sym_t = cs.SX
        self._dimc = dimc = 3
        self._dimx = dimx = 3 * dimc
        self._dimu = dimu = dimc
        self._dimf = dimf = dimc
        self._ncontacts = ncontacts = 4
        self._dimf_tot = dimf_tot = ncontacts * dimf

        # define cs variables
        delta_t = sym_t.sym('delta_t', 1)   # symbolic in order to pass values for optimization/interpolation
        c = sym_t.sym('c', dimc)
        dc = sym_t.sym('dc', dimc)
        ddc = sym_t.sym('ddc', dimc)
        x = cs.vertcat(c, dc, ddc) 
        u = sym_t.sym('u', dimc)

        # expression for the integrated state
        xf = cs.vertcat(
            c + dc*delta_t + 0.5*ddc*delta_t**2 + 1.0/6.0*u*delta_t**3,  # position
            dc + ddc*delta_t + 0.5*u*delta_t**2,  # velocity
            ddc + u*delta_t  # acceleration
        )

        # wrap the expression into a function
        self._integrator = cs.Function('integrator', [x, u, delta_t], [xf], ['x0', 'u', 'delta_t'], ['xf'])

        # construct the optimization problem (variables, cost, constraints, bounds)
        X = sym_t.sym('X', N * dimx)    # state is an SX for all knots
        U = sym_t.sym('U', N * dimu)    # for all knots
        F = sym_t.sym('F', N * (ncontacts * dimf))  # for all knots
        P = list()
        g = list()  # list of constraint expressions
        J = list()  # list of cost function expressions

        self._trj = {
            'x': X,
            'u': U,
            'F': F
        }

        # iterate over knots starting from k = 0
        for k in range(self._N):

            # slice indices for variables at knot k
            x_slice1 = k * dimx
            x_slice2 = (k + 1) * dimx
            u_slice1 = k * dimu
            u_slice2 = (k + 1) * dimu
            f_slice1 = k * dimf_tot
            f_slice2 = (k + 1) * dimf_tot

            # dynamics constraint
            if k > 0:
                x_old = X[(k - 1) * dimx : x_slice1]  # save previous state
                u_old = U[(k - 1) * dimu : u_slice1]  # prev control
                dyn_k = self._integrator(x0=x_old, u=u_old, delta_t=dt)['xf'] - X[x_slice1:x_slice2]
                g.append(dyn_k)

            # contact points
            p_k = sym_t.sym('p_' + str(k), ncontacts * dimc)
            P.append(p_k)

            # cost  function

            # horizontal distance of CoM from the mean of contact points
            h_horz = X[x_slice1:x_slice2][0:2] - 0.25 * (p_k[0:2] + p_k[3:5] + p_k[6:8] + p_k[9:11])  # xy

            # vertical distance between CoM and mean of feet
            h_vert = X[x_slice1:x_slice2][2] - 0.25 * (p_k[2] + p_k[5] + p_k[8] + p_k[11]) - 0.68

            j_k = 1e2 * cs.sumsqr(h_horz) + 1e3 * cs.sumsqr(h_vert) + \
                  1e-0 * cs.sumsqr(U[u_slice1:u_slice2]) + 1e-3 * cs.sumsqr(F[f_slice1:f_slice2][0::3]) + \
                  1e-3 * cs.sumsqr(F[f_slice1:f_slice2][1::3])

            J.append(j_k)

            # newton
            ddc_k = X[x_slice1:x_slice2][6:9]
            newton = mass*ddc_k - mass*gravity
            for i in range(ncontacts):
                f_i_k = F[f_slice1:f_slice2][3*i:3*(i+1)]  # force of i-th contact
                newton -= f_i_k

            g.append(newton)

            # euler
            c_k = X[x_slice1:x_slice2][0:3]
            euler = np.zeros(dimf)
            for i in range(ncontacts):
                f_i_k = F[f_slice1:f_slice2][3*i:3*(i+1)]  # force of i-th contact
                p_i_k = p_k[3*i:3*(i+1)]  # contact of i-th contact
                euler += cs.cross(p_i_k - c_k, f_i_k)

            g.append(euler)

        # construct the solver
        self._nlp = {
            'x': cs.vertcat(X, U, F),
            'f': sum(J),
            'g': cs.vertcat(*g),
            'p': cs.vertcat(*P)
            }

        # save dimensions
        self._nvars = self._nlp['x'].size1()
        self._nconstr = self._nlp['g'].size1()
        self._nparams = self._nlp['p'].size1()

        solver_options = {
            'ipopt.linear_solver': 'ma57'
        }

        self._solver = cs.nlpsol('solver', 'ipopt', self._nlp, solver_options)

    def solve(self, x0, contacts, swing_id, swing_tgt, swing_clearance, swing_t, min_f=50):
        """Solve the stepping problem

        Args:
            x0 ([type]): initial state (com position, velocity, acceleration)
            contacts ([type]): list of contact point positions
            swing_id ([type]): the index of the swing leg from 0 to 3
            swing_tgt ([type]): the target foothold for the swing leg
            swing_clearance: clearance achieved from the highest point between initial and target position
            swing_t ([type]): pair (t_lift, t_touch) in secs
            min_f: minimum threshold for forces in z direction
        """

        # lists for assigning bounds
        Xl = [0] * self._dimx * self._N     # state lower bounds (for all knots)
        Xu = [0] * self._dimx * self._N     # state upper bounds
        Ul = [0] * self._dimu * self._N     # control lower bounds
        Uu = [0] * self._dimu * self._N     # control upper bounds
        Fl = [0] * self._dimf_tot * self._N     # force lower bounds
        Fu = [0] * self._dimf_tot * self._N     # force upper bounds
        gl = list()  # constraint lower bounds
        gu = list()  # constraint upper bounds
        P = list()  # parameter values

        # time that maximum clearance occurs
        clearance_time = 0.5 * (swing_t[0] + swing_t[1]) #??????????????????????? is wrong

        # swing foot position at maximum clearance
        if contacts[swing_id][2] >= swing_tgt[2]:
            clearance_swing_position = contacts[swing_id][0:2].tolist() + [contacts[swing_id][2] + swing_clearance]
        else:
            clearance_swing_position = swing_tgt[0:2].tolist() + [swing_tgt[2] + swing_clearance]

        # iterate over knots starting from k = 0
        for k in range(self._N):

            # slice indices for bounds at knot k
            x_slice1 = k * self._dimx
            x_slice2 = (k + 1) * self._dimx
            u_slice1 = k * self._dimu
            u_slice2 = (k + 1) * self._dimu
            f_slice1 = k * self._dimf_tot
            f_slice2 = (k + 1) * self._dimf_tot

            # state bounds
            if k == 0:
                x_max = x0 
                x_min = x0

            else:
                x_max = np.concatenate([[cs.inf], [cs.inf], [cs.inf], np.full(6, cs.inf)])
                x_min = -np.concatenate([[cs.inf], [cs.inf], [cs.inf], np.full(6, cs.inf)])

            Xu[x_slice1:x_slice2] = x_max
            Xl[x_slice1:x_slice2] = x_min

            # ctrl bounds
            u_max = np.full(self._dimu, cs.inf)     # do not bound control
            u_min = -u_max 

            Uu[u_slice1:u_slice2] = u_max
            Ul[u_slice1:u_slice2] = u_min

            # force bounds
            f_max = np.full(self._dimf * self._ncontacts, cs.inf)
            f_min = np.array([-cs.inf, -cs.inf, min_f] * self._ncontacts)   # bound only the z component

            # swing phase
            is_swing = k >= swing_t[0]/self._dt and k <= swing_t[1]/self._dt
            if is_swing:
                # we are in swing phase
                f_max[3*swing_id:3*(swing_id+1)] = np.zeros(self._dimf) # overwrite forces for the swing leg
                f_min[3*swing_id:3*(swing_id+1)] = np.zeros(self._dimf)

            Fu[f_slice1:f_slice2] = f_max
            Fl[f_slice1:f_slice2] = f_min

            # contact positions
            p_k = np.hstack(contacts)  # start with initial contacts (4x3)

            # time region around max clearance time
            clearance_region = (clearance_time / self._dt - 4 <= k <= clearance_time / self._dt + 4)

            if clearance_region:
                p_k[3 * swing_id:3 * (swing_id + 1)] = clearance_swing_position

            elif k > clearance_time / self._dt + 4:
                # after the swing, the swing foot is now at swing_tgt
                p_k[3*swing_id:3*(swing_id+1)] = swing_tgt

            P.append(p_k)
            
            # dynamics bounds
            if k > 0:
                gl.append(np.zeros(self._dimx))
                gu.append(np.zeros(self._dimx))

            # constraint bounds (newton-euler eq.)
            gl.append(np.zeros(6)) 
            gu.append(np.zeros(6)) 

        # final constraints
        Xl[-6:] = [0.0 for i in range(6)]  # zero velocity and acceleration
        Xu[-6:] = [0.0 for i in range(6)]

        # initial guess
        v0 = np.zeros(self._nvars)

        # format bounds and params according to solver
        lbv = cs.vertcat(Xl, Ul, Fl)
        ubv = cs.vertcat(Xu, Uu, Fu)
        lbg = cs.vertcat(*gl)
        ubg = cs.vertcat(*gu)
        params = cs.vertcat(*P)

        # compute solution-call solver
        sol = self._solver(x0=v0, lbx=lbv, ubx=ubv, lbg=lbg, ubg=ubg, p=params)

        # plot state, forces, control input, quantities to be computed by evaluate function
        x_trj = cs.horzcat(self._trj['x']) # pack states in a desired matrix
        f_trj = cs.horzcat(self._trj['F']) # pack forces in a desired matrix
        u_trj = cs.horzcat(self._trj['u']) # pack control inputs in a desired matrix

        # return values of the quantities *_trj
        return {
            'x': self.evaluate(sol['x'], x_trj),
            'F': self.evaluate(sol['x'], f_trj),
            'u': self.evaluate(sol['x'], u_trj)
        }
        
    def evaluate(self, solution, expr):
        """ Evaluate a given expression

        Args:
            solution: given solution
            expr: expression to be evaluated

        Returns:
            Numerical value of the given expression

        """

        # casadi function that symbolically maps the _nlp to the given expression
        expr_fun = cs.Function('expr_fun', [self._nlp['x']], [expr], ['v'], ['expr'])

        expr_value = expr_fun(v=solution)['expr'].toarray()

        # make it a flat list
        expr_value_flat = [i for sublist in expr_value for i in sublist]

        return expr_value_flat

    def interpolate(self, solution, sw_curr, sw_tgt, clearance, sw_t, resol):
        """ Interpolate the trajectories generated by the solution of the problem

        Args:
            solution: solution of the problem (numerical values) is a directory
                solution['x'] 9x30 --> optimized states
                solution['f'] 12x30 --> optimized forces
                solution['u'] 9x30 --> optimized control
            sw_curr: current position of the foot to be swinged
            sw_tgt: target position of the foot to be swinged
            clearance: swing clearance
            sw_t: (start, stop) period of foot swinging in a global manner wrt to optimization problem
            resol: interpolation resolution (points per second)

        Returns: a dictionary with:
            time list for interpolation times (in sec)
            list of list with state trajectory points
            list of lists with forces' trajectory points
            list of lists with the swinging foot's trajectory points

        """

        # start and end times of optimization problem
        t_tot = [0.0, self._N * self._dt]

        delta_t = 1.0 / resol  # dt for interpolation

        # -------- state trajectory interpolation ------------
        # intermediate points between two knots --> time interval * resolution
        self._n = int(self._dt * resol)

        x_old = solution['x'][0:9]  # initial state
        x_all = []  # list to append all states

        for ii in range(self._N):   # loop for knots

            # control input to change in every knot
            u_old = solution['u'][self._dimu*ii:self._dimu*(ii + 1)]

            for j in range(self._n):     # loop for interpolation points

                x_all.append(x_old)  # storing state in the list 600x9

                x_next = self._integrator(x_old, u_old, delta_t)    # next state
                x_old = x_next     # refreshing the current state

        # initialize state and time lists to gather the data
        int_state = [[] for i in range(self._dimx)]     # primary dimension = number of state components
        self._t = [(ii*delta_t) for ii in range(self._N * self._n)]

        for i in range(self._dimx):     # loop for every component of the state vector
            for j in range(self._N * self._n):    # loop for every point of interpolation

                # append the value of x_i component on j point of interpolation
                # in the element i of the list int_state
                int_state[i].append(x_all[j][i])

        # ----------- force trajectory interpolation --------------
        force_func = [[] for i in range(self._dimf_tot)]    # list to store the splines
        int_force = [[] for i in range(self._dimf_tot)]     # list to store lists of points

        test = solution['F'][1::self._dimf_tot]
        for i in range(self._dimf_tot): # loop for each component of the force vector

            # append the spline (by casadi) in the i element of the list force_func
            force_func[i].append(cs.interpolant('X_CONT', 'linear',
                                                [self._time],
                                                solution['F'][i::self._dimf_tot]))

            # store the interpolation points for each force component in the i element of the list int_force
            # primary dimension = number of force components
            int_force[i] = force_func[i][0](self._t)

        # ----------- swing leg trajectory interpolation --------------
        # swing trajectory with intemediate point
        sw_interpl = interpol.swing_trj_triangle(sw_curr, sw_tgt, clearance, sw_t, t_tot, resol)

        # swing trajectory with spline optimization for z coordinate
        #sw_interpl = interpol.swing_trj_optimal_spline(sw_curr, sw_tgt, clearance, sw_t, t_tot, resol)

        return {
            't': self._t,
            'x': int_state,
            'f': int_force,
            'sw': sw_interpl
        }

    def print_trj(self, solution, results, resol, t_exec=0):
        '''

        Args:
            t_exec: time that trj execution stopped (because of early contact or other)
            results: results from interpolation
            resol: interpolation resol
            publish_freq: publish frequency - applies only when trajectories are published through rostopics,
                else publish_freq = resol

        Returns: prints the nominal interpolated trajectories

        '''

        # Interpolated state plot
        state_labels = ['CoM Position', 'CoM Velocity', 'CoM Acceleration']
        plt.figure()
        for i, name in enumerate(state_labels):
            plt.subplot(3, 1, i + 1)
            for j in range(self._dimc):
                plt.plot(results['t'], results['x'][self._dimc * i + j], '-')
                #plt.plot(self._time, solution['x'][self._dimc * i + j::self._dimx], 'o')
            plt.grid()
            plt.legend(['x', 'y', 'z'])
            #plt.legend(['x', 'xopt', 'y', 'yopt', 'z', 'zopt'])
            plt.title(name)
        plt.xlabel('Time [s]')
        #plt.savefig('../plots/step_state_trj.png')

        feet_labels = ['front left', 'front right', 'hind left', 'hind right']

        # Interpolated force plot
        plt.figure()
        for i, name in enumerate(feet_labels):
            plt.subplot(2, 2, i + 1)
            for k in range(3):
                plt.plot(results['t'], results['f'][3 * i + k], '-')
            plt.grid()
            plt.title(name)
            plt.legend([str(name) + '_x', str(name) + '_y', str(name) + '_z'])
        plt.xlabel('Time [s]')
        #plt.savefig('../plots/step_forces.png')

        # plot swing trajectory
        # All points to be published
        N_total = int(self._N * self._dt * resol)  # total points --> total time * frequency
        s = np.linspace(0, self._dt * self._N, N_total)
        coord_labels = ['x', 'y', 'z']
        plt.figure()
        for i, name in enumerate(coord_labels):
            plt.subplot(3, 1, i + 1)
            plt.plot(s, results['sw'][name])   # nominal trj
            plt.plot(s[0:t_exec], results['sw'][name][0:t_exec])   # executed trj
            plt.grid()
            plt.legend(['nominal', 'real'])
            plt.title('Trajectory ' + name)
        plt.xlabel('Time [s]')
        #plt.savefig('../plots/step_swing.png')

        # plot swing trajectory in two dimensions Z- X
        plt.figure()
        plt.plot(results['sw']['x'], results['sw']['z'])    # nominal trj
        plt.plot(results['sw']['x'][0:t_exec], results['sw']['z'][0:t_exec])    # real trj
        plt.grid()
        plt.legend(['nominal', 'real'])
        plt.title('Trajectory Z- X')
        plt.xlabel('X [m]')
        plt.ylabel('Z [m]')
        #plt.savefig('../plots/step_swing_zx.png')
        plt.show()


if __name__ == "__main__":

    w = Walking(mass=95, N=40, dt=0.2)

    # initial state =
    c0 = np.array([0.107729, 0.0000907, -0.02118])
    dc0 = np.zeros(3)
    ddc0 = np.zeros(3)
    x_init = np.hstack([c0, dc0, ddc0])

    foot_contacts = [
        np.array([0.35, 0.35, -0.7187]),   # fl
        np.array([0.35, -0.35, -0.7187]),  # fr
        np.array([-0.35, 0.35, -0.7187]),    # hl
        np.array([-0.35, -0.35, -0.7187])   # hr
    ]

    # swing id from 0 to 3
    sw_id = 0

    step_clear = 0.05

    # swing_target = np.array([-0.35, -0.35, -0.719])
    dx = 0.1
    dy = 0.0
    dz = -0.05
    swing_target = np.array([foot_contacts[sw_id][0] + dx, foot_contacts[sw_id][1] + dy, foot_contacts[sw_id][2] + dz])

    # swing_time = (1.5, 3.0)
    swing_time = [2.0, 5.0]

    # sol is the directory returned by solve class function contains state, forces, control values
    sol = w.solve(x0=x_init, contacts=foot_contacts, swing_id=sw_id, swing_tgt=swing_target,
                  swing_clearance=step_clear, swing_t=swing_time, min_f=100)

    # debug
    print("X0 is:", x_init)
    print("contacts is:", foot_contacts)
    print("swing id is:", sw_id)
    print("swing target is:", swing_target)
    print("swing time:", swing_time)

    # interpolate the values, pass values and interpolation resolution
    res = 300
    interpl = w.interpolate(sol, foot_contacts[sw_id], swing_target, step_clear, swing_time, res)

    # print the results
    w.print_trj(sol, interpl, res)


