import casadi as cs 
import numpy as np
from matplotlib import pyplot as plt

class Walking:
    """
    Assumptions:
      1) mass concentrated at com
      2) zero angular momentum
      3) point contacts

    Dynamics:
      1) input is com jerk
      2) dyamics is a triple integrator of com jerk
      3) there must be contact forces that
        - realize the motion
        - fulfil contact constraints (i.e. unilateral constraint)

    TODO:
      0) play around, change cost, etc
      1) interpolate motion to a higher frequency (small dt)
      2) swing leg trajectory (at least continuous acceleration)
      3) send to cartesio for ik (using ros topics)
      4) gazebo simulation
      5) contact detection
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
        self._time = [(i * dt) for i in range(N)]

        gravity = np.array([0, 0, -9.81])

        # define dimensions
        sym_t = cs.SX
        self._dimc = dimc = 3
        self._dimx = dimx = 3 * dimc
        self._dimu = dimu = dimc
        self._dimf = dimf = dimc
        self._ncontacts = ncontacts = 4

        # define cs variables
        delta_t = sym_t.sym('delta_t', 1)   #symbolic in order to pass values for optimization/interpolation
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
        X = list()
        U = list()
        F = list()
        P = list()
        g = list()  # list of constraint expressions
        J = list()  # list of cost function expressions

        self._trj = {
            'x': X,
            'u': U,
            'F': F
        }

        # iterate over knots starting from k = 0
        for k in range(N):

            # create k-th state
            x_k = sym_t.sym('x_' + str(k), dimx)
            X.append(x_k)

            # create k-th control
            u_k = sym_t.sym('u_' + str(k), dimu)
            U.append(u_k)

            # dynamics constraint
            if k > 0:
                x_old = X[-2]  # save previous state
                u_old = U[-2]  # prev control
                dyn_k = self._integrator(x0=x_old, u=u_old, delta_t=dt)['xf'] - x_k
                g.append(dyn_k)

            # forces
            f_k = sym_t.sym('f_' + str(k), ncontacts * dimf)
            F.append(f_k)

            # contact points
            p_k = sym_t.sym('p_' + str(k), ncontacts * dimc)
            P.append(p_k)

            # cost
            j_k = 0.5 * cs.sumsqr(u_k) + 1e-12 * cs.sumsqr(f_k) + 5e-0 * cs.sumsqr(x_k[0:2])  # 1/2 |u_k|^2
            J.append(j_k)

            # newton
            ddc_k = x_k[6:9]
            newton = mass*ddc_k - mass*gravity
            for i in range(ncontacts):
                f_i_k = f_k[3*i:3*(i+1)]  # force of i-th contact
                newton -= f_i_k

            g.append(newton)

            # euler
            c_k = x_k[0:3]
            euler = np.zeros(dimf)
            for i in range(ncontacts):
                f_i_k = f_k[3*i:3*(i+1)]  # force of i-th contact
                p_i_k = p_k[3*i:3*(i+1)]  # contact of i-th contact
                euler += cs.cross(p_i_k - c_k, f_i_k)

            g.append(euler)

        # construct the solver
        self._nlp = {
            'x': cs.vertcat(*X, *U, *F),
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

    def solve(self, x0, contacts, swing_id, swing_tgt, swing_t, min_f=50):
        """Solve the stepping problem

        Args:
            x0 ([type]): initial state (com position, velocity, acceleration)
            contacts ([type]): list of contact point positions
            swing_id ([type]): the index of the swing leg
            swing_tgt ([type]): the target foothold for the swing leg
            swing_t ([type]): pair (t_lift, t_touch) in secs
        """
        
        Xl = list()  # state lower bounds
        Xu = list()  # state upper bounds
        Ul = list()  # control lower bounds
        Uu = list()  # control upper bounds
        Fl = list()  # force lower bounds
        Fu = list()  # force upper bounds
        gl = list()  # constraint lower bounds
        gu = list()  # constraint upper bounds
        P = list()  # parameter values

        # iterate over knots starting from k = 0
        for k in range(self._N):

            # state bounds
            if k == 0:
                x_max = x0 
                x_min = x0
            else:
                x_max = np.full(self._dimx, cs.inf) #do not bound state
                x_min = -x_max 

            Xu.append(x_max)
            Xl.append(x_min)

            # ctrl bounds
            u_max = np.full(self._dimu, cs.inf) #do not bound control
            u_min = -u_max 
            Uu.append(u_max)
            Ul.append(u_min)

            # force bounds
            f_max = np.full(self._dimf * self._ncontacts, cs.inf)
            f_min = np.array([-cs.inf, -cs.inf, min_f] * self._ncontacts)   #bound only the z component

            # swing phase
            is_swing = k >= swing_t[0]/self._dt and k <= swing_t[1]/self._dt
            if is_swing:
                # we are in swing phase
                f_max[3*swing_id:3*(swing_id+1)] = np.zeros(self._dimf) #overwrite forces for the lifted leg
                f_min[3*swing_id:3*(swing_id+1)] = np.zeros(self._dimf)
            Fl.append(f_min)
            Fu.append(f_max)

            # contact positions
            p_k = np.hstack(contacts)  # start with initial contacts (4x3)
            if k >= swing_t[0]/self._dt:
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
        Xl[-1][3:] = 0  #zero velocity and acceleration
        Xu[-1][3:] = 0

        # initial guess
        v0 = np.zeros(self._nvars)

        # format bounds and params according to solver
        lbv = cs.vertcat(*Xl, *Ul, *Fl)
        ubv = cs.vertcat(*Xu, *Uu, *Fu)
        lbg = cs.vertcat(*gl)
        ubg = cs.vertcat(*gu)
        params = cs.vertcat(*P)

        # compute solution-call solver
        sol = self._solver(x0=v0, lbx=lbv, ubx=ubv, lbg=lbg, ubg=ubg, p=params)

        # plot state, forces, control input, quantities to be computed by evaluate function
        x_trj = cs.horzcat(*self._trj['x']) # pack states in a desired matrix
        f_trj = cs.horzcat(*self._trj['F']) # pack forces in a desired matrix
        u_trj = cs.horzcat(*self._trj['u']) # pack control inputs in a desired matrix

        # return values of the quantities *_trj
        return {
            'x': self.evaluate(sol['x'], x_trj),
            'F': self.evaluate(sol['x'], f_trj),
            'u': self.evaluate(sol['x'], u_trj)
        }
        
    def evaluate(self, solution, expr):
        """ Evaluate the given expression

        Args:
            solution: given solution
            expr: expression to be evaluated

        Returns:
            Numerical value of the given expression

        """

        # casadi function that symbolically maps the _nlp to the given expression
        expr_fun = cs.Function('expr_fun', [self._nlp['x']], [expr], ['v'], ['expr'])

        expr_value = expr_fun(v=solution)['expr'].toarray()

        return expr_value

    def interpolate(self, solution, pos_curr, pos_tgt, swing_t, resol):
        """ Interpolate the solution of the problem

        Args:
            solution: solution of the problem (values) is a directory
                solution['x'] 9x30 --> optimized states
                solution['f'] 12x30 --> optimized forces
                solution['u'] 9x30 --> optimized control
            pos_curr: current position of the foot to be swinged
            pos_tgt: target position of the foot to be swinged
            swing_t: (start, stop) period of foot swinging in a global manner wrt to optimization problem
            resol: interpolation resolution (points per second)

        Returns: a dictionary with:
            time list for interpolation times (in sec)
            list of list with state trajectory points
            list of lists with forces' trajectory points
            list of lists with the swinging foot's trajectory points

        """

        # -------- state trajectory interpolation ------------
        delta_t = 1 / resol # delta_t for interpolation

        # intermediate points between two knots --> time interval * resolution
        n = int(self._dt * resol)

        x_old = solution['x'][0:9, 0]  #i nitial state
        x_all = []  # list to append all states

        for ii in range(self._N):   # loop for knots

            # control input to change in every knot
            u_old = solution['u'][0:3, ii]

            for j in range(n):     # loop for interpolation points

                x_all.append(x_old)  # storing state in the list 600x9

                x_next = self._integrator(x_old, u_old, delta_t)    # next state
                x_old = x_next     # refreshing the current state

        # initialize state and time lists to gather the data
        int_state = [[] for i in range(self._dimx)]
        t = [(ii*delta_t) for ii in range(self._N * n)]

        for i in range(self._dimx): # loop for every element of the state vector
            for j in range(self._N * n):    # loop for every point of interpolation

                # store the value of x_i component on ii point of interpolation
                # in the element i of the list int_state
                int_state[i].append(x_all[j][i])

        # ----------- force trajectory interpolation --------------
        force_func = [[] for i in range(len(solution['F']))] # list to store the splines
        int_force = [[] for i in range(len(solution['F']))]  # list to store lists of points

        for i in range(len(solution['F'])): # loop for each component of the force vector

            # store the spline (by casadi) in the i element of the list force_func
            force_func[i].append(cs.interpolant('X_CONT', 'linear', [self._time], solution['F'][i]))

            # store the interpolation points for each force component in the i element of the list int_force
            int_force[i] = force_func[i][0](t)

        # ----------- swing leg trajectory interpolation --------------

        # list of first and last point of swing phase
        sw_points = [pos_curr] + [pos_tgt]

        # list of the two points for each coordinate
        x = [sw_points[0][0], sw_points[1][0]]
        y = [sw_points[0][1], sw_points[1][1]]
        z = [sw_points[0][2], sw_points[1][2]]

        # conditions, initial point of swing phase
        init_x = [x[0], 0, 0]
        init_y = [y[0], 0, 0]
        init_z = [z[0], 0, 0]

        # conditions, final point of swing phase
        fin_x = [x[1], 0, 0]
        fin_y = [y[1], 0, 0]
        fin_z = [z[1], 0, 0]

        # save polynomial coefficients in one list for each coordinate
        # inverse the elements as [0, 1, 1] is translated into 1 + x from polynomial function
        coeff_x = self.splines(swing_t, init_x, fin_x)[::-1]
        coeff_y = self.splines(swing_t, init_y, fin_y)[::-1]
        coeff_z = self.splines(swing_t, init_z, fin_z)[::-1]

        # convert to polynomial functions
        poly_x = np.poly1d(coeff_x)
        poly_y = np.poly1d(coeff_y)
        poly_z = np.poly1d(coeff_z)

        # construct list of interpolated points according to specified resolution
        sw_dt = swing_t[1] - swing_t[0]
        sw_interpl_t = np.linspace(swing_t[0], swing_t[1], int(resol * sw_dt))
        sw_interpl_x = poly_x(sw_interpl_t)
        sw_interpl_y = poly_y(sw_interpl_t)
        sw_interpl_z = poly_z(sw_interpl_t)

        # add points for non swing phase
        t_start = 0.0  # will be removed
        t_end = 3.0

        # number of interpolation points in non swing phases
        sw_t1 = int(resol * (swing_t[0] - t_start))
        sw_t2 = int(resol * (t_end - swing_t[1]))

        # add points for non swing phases
        sw_interpl_x = [pos_curr[0]] * sw_t1 + [sw_interpl_x[i] for i in range(len(sw_interpl_x))] + [pos_tgt[0]] * sw_t2
        sw_interpl_y = [pos_curr[1]] * sw_t1 + [sw_interpl_y[i] for i in range(len(sw_interpl_y))] + [pos_tgt[1]] * sw_t2
        sw_interpl_z = [pos_curr[2]] * sw_t1 + [sw_interpl_z[i] for i in range(len(sw_interpl_z))] + [pos_tgt[2]] * sw_t2

        # include all coordinates in a single list
        sw_interpl = [sw_interpl_x, sw_interpl_y, sw_interpl_z]

        return {
            't': t,
            'x': int_state,
            'f': int_force,
            'sw': sw_interpl
        }

    def splines(self, dt, init_cond, fin_cond):
        """
        This function computes the polynomial of 5th order between two points with 6 given conditions
        Args:
            dt: time interval that the polynomial applies
            init_cond: list with initial position, velocity and acceleration conditions
            fin_cond: list with final position, velocity and acceleration conditions

        Returns:
            a list with the coefficient of the polynomial of the spline
        """

        # symbolic polynomial coefficients
        sym_t = cs.SX
        a0 = sym_t.sym('a0', 1)
        a1 = sym_t.sym('a1', 1)
        a2 = sym_t.sym('a2', 1)
        a3 = sym_t.sym('a3', 1)
        a4 = sym_t.sym('a4', 1)
        a5 = sym_t.sym('a5', 1)

        # time
        t = sym_t.sym('t', 1)

        # initial and final conditions
        p0 = init_cond[0]
        v0 = init_cond[1]
        ac0 = init_cond[2]
        p1 = fin_cond[0]
        v1 = fin_cond[1]
        ac1 = fin_cond[2]
        print('Initial and final conditions are:', p0, v0, ac0, p1, v1, ac1)

        # the 5th order polynomial expression
        spline = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5

        # wrap the polynomial expression in a function
        p = cs.Function('p', [t, a0, a1, a2, a3, a4, a5], [spline], ['t', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5'],
                        ['spline'])

        # symbolic velocity - 1st derivative
        first_der = cs.jacobian(spline, t)
        dp = cs.Function('dp', [t, a1, a2, a3, a4, a5], [first_der], ['t', 'a1', 'a2', 'a3', 'a4', 'a5'], ['first_der'])

        # symbolic acceleration - 2nd derivative
        sec_der = cs.jacobian(first_der, t)
        ddp = cs.Function('ddp', [t, a2, a3, a4, a5], [sec_der], ['t', 'a2', 'a3', 'a4', 'a5'], ['sec_der'])

        # construct the system of equations Ax=B, with x the list of coefficients to be computed
        A = np.array([[p(dt[0], 1, 0, 0, 0, 0, 0), p(dt[0], 0, 1, 0, 0, 0, 0), p(dt[0], 0, 0, 1, 0, 0, 0),
                       p(dt[0], 0, 0, 0, 1, 0, 0), p(dt[0], 0, 0, 0, 0, 1, 0), p(dt[0], 0, 0, 0, 0, 0, 1)],\
                      [0, dp(dt[0], 1, 0, 0, 0, 0), dp(dt[0], 0, 1, 0, 0, 0), dp(dt[0], 0, 0, 1, 0, 0),
                       dp(dt[0], 0, 0, 0, 1, 0), dp(dt[0], 0, 0, 0, 0, 1)],\
                      [0, 0, ddp(dt[0], 1, 0, 0, 0), ddp(dt[0], 0, 1, 0, 0), ddp(dt[0], 0, 0, 1, 0),
                       ddp(dt[0], 0, 0, 0, 1)],\
                      [p(dt[1], 1, 0, 0, 0, 0, 0), p(dt[1], 0, 1, 0, 0, 0, 0), p(dt[1], 0, 0, 1, 0, 0, 0),
                       p(dt[1], 0, 0, 0, 1, 0, 0), p(dt[1], 0, 0, 0, 0, 1, 0), p(dt[1], 0, 0, 0, 0, 0, 1)],\
                      [0, dp(dt[1], 1, 0, 0, 0, 0), dp(dt[1], 0, 1, 0, 0, 0), dp(dt[1], 0, 0, 1, 0, 0),
                       dp(dt[1], 0, 0, 0, 1, 0), dp(dt[1], 0, 0, 0, 0, 1)],\
                      [0, 0, ddp(dt[1], 1, 0, 0, 0), ddp(dt[1], 0, 1, 0, 0), ddp(dt[1], 0, 0, 1, 0),
                       ddp(dt[1], 0, 0, 0, 1)]])

        B = np.array([p0, v0, ac0, p1, v1, ac1])

        # coefficients
        coeffs = np.linalg.inv(A).dot(B)
        print(coeffs)

        return coeffs


if __name__ == "__main__":

    w = Walking(mass=90, N=30, dt=0.1)

    # initial state
    c0 = np.array([0.1, 0.0, 0.64])
    dc0 = np.zeros(3)
    ddc0 = np.zeros(3)
    x0 = np.hstack([c0, dc0, ddc0])

    contacts = [
        np.array([0.35, 0.35, 0.0]),   # fl
        np.array([0.35, -0.35, 0.0]),  # fr
        np.array([-0.35, -0.35, 0.0]), # hr
        np.array([-0.35, 0.35, 0.0])   # hl
    ]

    swing_id = 0

    swing_tgt = np.array([0.45, 0.35, 0.1])

    swing_t = (1.0, 2.0)

    # sol is the directory returned by solve class function contains state, forces, control values
    sol = w.solve(x0=x0, contacts=contacts, swing_id=swing_id, swing_tgt=swing_tgt, swing_t=swing_t)

    # interpolate the values, pass values and interpolation resolution
    res = 100
    interpl = w.interpolate(sol, contacts[0], swing_tgt, swing_t, res)

    # plot com position from optimization
    plt.figure()
    plt.plot(np.arange(w._N)*w._dt, sol['x'][0:3, :].transpose(), '-o')
    plt.grid()
    plt.title('State trajectory')
    plt.legend(['x', 'y', 'z'])
    plt.xlabel('Time [s]')

    # plot forces from optimization problem
    plt.figure()
    feet_labels = ['front left', 'front right', 'hind right', 'hind left']
    for i, name in enumerate(feet_labels):
        plt.subplot(2, 2, i+1)
        plt.plot(np.arange(w._N)*w._dt, sol['F'][3*i:3*(i+1), :].transpose(), 'o-')
        plt.grid()
        plt.title('Force trajectory ({})'.format(name))
        plt.legend(['x', 'y', 'z'])
        plt.xlabel('Time [s]')

    # Interpolated state plot
    state_labels = ['CoM Position', 'CoM Velocity', 'CoM Acceleration']
    plt.figure()
    for i, name in enumerate(state_labels):
        plt.subplot(3, 1, i+1)
        for j in range(w._dimc):
            plt.plot(interpl['t'], interpl['x'][w._dimc*i + j], '-')
        plt.grid()
        plt.legend(['x', 'y', 'z'])
        plt.title(name)
    plt.xlabel('Time [s]')
  
    # Interpolated force plot
    s = np.linspace(0, int(w._N*w._dt), int(w._N*w._dt)*res)
    plt.figure()
    for i, name in enumerate(feet_labels):
        plt.subplot(2, 2, i + 1)
        for k in range(3):
            plt.plot(s, interpl['f'][3*i + k], '-')
        plt.grid()
        plt.title(name)
        plt.legend([str(name) + '_x', str(name) + '_y', str(name) + '_z'])
    plt.xlabel('Time [s]')

    plt.show()


