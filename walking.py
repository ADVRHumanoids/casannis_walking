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

    def __init__(mass, N, dt):
        
        self._N = N 
        self._dt = dt
        self._mass = mass 

        gravity = np.array([0, 0, -9.81])

        sym_t = cs.SX 
        dimc = 3
        dimx = 3 * dimc
        dimu = dimc
        dimf = dimc
        ncontacts = 4

        # define cs variables
        c = sym_t.sym('c', dimc)
        dc = _sym_t.sym('dc', dimc)
        ddc = _sym_t.sym('ddc', dimc)
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

            # newton
            ddc_k = x_k[6:9]
            newton = mass*ddc_k - mass*gravity
            for i in ncontacts:
                f_i_k = f_k[3*i:3*(i+1)]  # force of i-th contact
                newton -= f_i_k

            g.append(newton)

            # euler         
            c_k = x_k[0:3]   
            euler = np.zeros(dimf)
            for i in ncontacts:
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
        

            


