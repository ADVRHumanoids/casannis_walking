import casadi as cs 
import numpy as np

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
       - fulfil contact constraints (i.e. unilateral constr)
    """

    def __init__(self, mass, N, dt):
        """Walking class constructor

        Args:
            mass (float): robot mass
            N (int): horizon length
            dt (float): discretization step
        """

        self._N = N 
        self._dt = dt
        self._mass = mass 

        gravity = np.array([0, 0, -9.81])

        sym_t = cs.SX 
        self._dimc = dimc = 3
        self._dimx = dimx = 3 * dimc
        self._dimu = dimu = dimc
        self._dimf = dimf = dimc
        self._ncontacts = ncontacts = 4

        # define cs variables
        c = sym_t.sym('c', dimc)
        dc = sym_t.sym('dc', dimc)
        ddc = sym_t.sym('ddc', dimc)
        x = cs.vertcat(c, dc, ddc) 
        u = sym_t.sym('u', dimc)

        # expression for the integrated state
        xf = cs.vertcat(
            c + dc*dt + 0.5*ddc*dt**2 + 1.0/6.0*u*dt**3,  # pos
            dc + ddc*dt + 0.5*u*dt**2,  # vel
            ddc + u*dt  # acc
        )

        # wrap the expression into a function
        self._integrator = cs.Function('integrator', {'x0': x, 'u': u, 'xf': xf}, ['x0', 'u'], ['xf'])

        # construct the optimization problem (variables, cost, constraints, bounds)
        X = list()
        U = list()
        F = list()
        P = list()
        g = list()  # list of constraint expressions
        J = list()  # list of cost function expressions

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
                dyn_k = self._integrator(x0=x_old, u=u_old)['xf'] - x_k
                g.append(dyn_k)

            # cost
            j_k = 0.5 * cs.sumsqr(u_k)  # 1/2 |u_k|^2
            J.append(j_k)

            # forces
            f_k = sym_t.sym('f_' + str(k), ncontacts*dimf)
            F.append(f_k)

            # contact points
            p_k = sym_t.sym('p_' + str(k), ncontacts*dimc)
            P.append(p_k)

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

        # final constraint (vel and acc should vanish)
        x_final = X[-1]
        dc_final = x_final[3:6]
        ddc_final = x_final[6:9]
        g.append(dc_final)
        g.append(ddc_final)
        
        # construct the solver
        nlp = {
            'x': cs.vertcat(*X, *U, *F),
            'f': sum(J),
            'g': cs.vertcat(*g),
            'p': cs.vertcat(*P) 
            }

        self._solver = cs.nlpsol('solver', 'ipopt', nlp)

    def solve(self, x0, contacts, swing_id, swing_tgt, swing_t):
        """Solve the stepping problem

        Args:
            x0 ([type]): initial state (com position, velocity, acceleration)
            contacts ([type]): list of contact point positions
            swing_id ([type]): the index of the swing leg
            swing_tgt ([type]): the target foothold for the swing leg
            swing_t ([type]): pair (t_lift, t_touch) in secs
        """
        
        Xl = list()
        Xu = list()
        Ul = list()
        Uu = list()
        Fl = list()
        Fu = list()
        gl = list()
        gu = list()
        P = list()

        # iterate over knots starting from k = 0
        for k in range(self._N):

            # state bounds
            if k == 0:
                x_max = x0 
                x_min = x0
            else:
                x_max = np.full(self._dimx, cs.inf)
                x_min = -x_max 

            Xu.append(x_max)
            Xl.append(x_min)

            # ctrl bounds
            u_max = np.full(self._dimu, cs.inf)
            u_min = -u_max 
            Uu.append(u_max)
            Ul.append(u_min)

            # force bounds
            f_max = np.full(self._dimf * self._ncontacts, cs.inf)
            f_min = np.array([-cs.inf, -cs.inf, 0.0] * self._ncontacts)

            # swing phase
            is_swing = k >= swing_t[0]/self._dt and k <= swing_t[1]/self._dt
            if is_swing:
                # we are in swing phase
                f_max[3*swing_id:3*(swing_id+1)] = np.zeros(self._dimf)
                f_min[3*swing_id:3*(swing_id+1)] = np.zeros(self._dimf)
            Fl.append(f_min)
            Fu.append(f_max)

            # contact positions
            p_k = np.hstack(contacts)  # start with initial contacts
            if k >= swing_t[0]/self._dt:
                # after the swing, the swing foot is now at swing_tgt
                p_k[3*swing_id:3*(swing_id+1)] = swing_tgt
            P.append(p_k)

            
            # dynamics boudns
            if k > 0:
                gl.append(np.zeros(self._dimx))
                gu.append(np.zeros(self._dimx))

            # constraint bounds (ne eq.)
            gl.append(np.zeros(6)) 
            gu.append(np.zeros(6)) 

            





            


w = Walking(mass=90, N=50, dt=0.1)

w.solve(x0=np.zeros(9), contacts=[np.ones(3) for _ in range(4)], swing_id=1, swing_tgt=[0, 1, 2], swing_t=(2, 3))