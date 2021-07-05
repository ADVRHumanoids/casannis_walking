import casadi as cs
import numpy as np
from matplotlib import pyplot as plt
import trj_interpolation as interpol
import time
import cubic_hermite_polynomial as cubic_spline


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
        print("This class is a first trial towards walking with payload modeled as a force applied to a moving contact point")

        self._N = N
        self._dt = dt  # dt used for optimization knots
        self._mass = mass
        self._time = [(i * dt) for i in range(self._N)]         # time junctions w/o the last one
        self._tjunctions = [(i * dt) for i in range(self._N + 1)]   # time junctions from first to last


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
        delta_t = sym_t.sym('delta_t', 1)  # symbolic in order to pass values for optimization/interpolation
        c = sym_t.sym('c', dimc)
        dc = sym_t.sym('dc', dimc)
        ddc = sym_t.sym('ddc', dimc)
        x = cs.vertcat(c, dc, ddc)
        u = sym_t.sym('u', dimc)

        # expression for the integrated state
        xf = cs.vertcat(
            c + dc * delta_t + 0.5 * ddc * delta_t ** 2 + 1.0 / 6.0 * u * delta_t ** 3,  # position
            dc + ddc * delta_t + 0.5 * u * delta_t ** 2,  # velocity
            ddc + u * delta_t  # acceleration
        )

        # wrap the expression into a function
        self._integrator = cs.Function('integrator', [x, u, delta_t], [xf], ['x0', 'u', 'delta_t'], ['xf'])

        # construct the optimization problem (variables, cost, constraints, bounds)
        X = sym_t.sym('X', N * dimx)  # state is an SX for all knots
        U = sym_t.sym('U', N * dimu)  # for all knots
        F = sym_t.sym('F', N * (ncontacts * dimf))  # for all knots
        P = list()
        g = list()  # list of constraint expressions
        J = list()  # list of cost function expressions

        # extra moving contact to model payload
        f_pay = [0, 0, -200]    # virtual force
        P_mov = sym_t.sym('P_mov', N * 3)   # position knots for the virtual contact
        DP_mov = sym_t.sym('DP_mov', N * 3)   # velocity knots for the virtual contact

        self._trj = {
            'x': X,
            'u': U,
            'F': F,
            'P_mov': P_mov,
            'DP_mov': DP_mov
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
                x_old = X[(k - 1) * dimx: x_slice1]  # save previous state
                u_old = U[(k - 1) * dimu: u_slice1]  # prev control
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
            newton = mass * ddc_k - mass * gravity
            for i in range(ncontacts):
                f_i_k = F[f_slice1:f_slice2][3 * i:3 * (i + 1)]  # force of i-th contact
                newton -= f_i_k
            newton -= f_pay
            g.append(newton)

            # euler
            p_mov = P_mov[u_slice1:u_slice2]     # moving contact at knot k
            c_k = X[x_slice1:x_slice2][0:3]
            #euler = np.zeros(dimf)
            euler = cs.cross(f_pay, c_k - p_mov)    # payload addition

            for i in range(ncontacts):
                f_i_k = F[f_slice1:f_slice2][3 * i:3 * (i + 1)]  # force of i-th contact
                p_i_k = p_k[3 * i:3 * (i + 1)]  # contact of i-th contact

                euler += cs.cross(f_i_k, c_k - p_i_k)
            g.append(euler)

        # construct the solver
        self._nlp = {
            'x': cs.vertcat(X, U, F, P_mov, DP_mov),
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
        Xl = [0] * self._dimx * self._N  # state lower bounds (for all knots)
        Xu = [0] * self._dimx * self._N  # state upper bounds
        Ul = [0] * self._dimu * self._N  # control lower bounds
        Uu = [0] * self._dimu * self._N  # control upper bounds
        Fl = [0] * self._dimf_tot * self._N  # force lower bounds
        Fu = [0] * self._dimf_tot * self._N  # force upper bounds
        P_movl = [0] * self._dimu * self._N  # position of moving contact lower bounds
        P_movu = [0] * self._dimu * self._N  # position of moving contact upper bounds
        DP_movl = [0] * self._dimu * self._N  # velocity of moving contact lower bounds
        DP_movu = [0] * self._dimu * self._N  # velocity of moving contact upper bounds
        gl = list()  # constraint lower bounds
        gu = list()  # constraint upper bounds
        P = list()  # parameter values

        # time that maximum clearance occurs
        clearance_time = 0.5 * (swing_t[0] + swing_t[1])  # not accurate

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
            u_max = np.full(self._dimu, cs.inf)  # do not bound control
            u_min = -u_max

            Uu[u_slice1:u_slice2] = u_max
            Ul[u_slice1:u_slice2] = u_min

            # force bounds
            f_max = np.full(self._dimf * self._ncontacts, cs.inf)
            f_min = np.array([-cs.inf, -cs.inf, min_f] * self._ncontacts)  # bound only the z component

            # swing phase
            is_swing = k >= swing_t[0] / self._dt and k <= swing_t[1] / self._dt
            if is_swing:
                # we are in swing phase
                f_max[3 * swing_id:3 * (swing_id + 1)] = np.zeros(self._dimf)  # overwrite forces for the swing leg
                f_min[3 * swing_id:3 * (swing_id + 1)] = np.zeros(self._dimf)

            Fu[f_slice1:f_slice2] = f_max
            Fl[f_slice1:f_slice2] = f_min

            # Moving contact position bounds
            if k == 0:
                p_mov_max = np.array([0.53, 0.0, 0.3])
                p_mov_min = np.array([0.53, 0.0, 0.3])
            else:
                p_mov_max = np.array([0.7, 0.2, 0.3])
                p_mov_min = np.array([0.53, -0.2, 0.3])

            P_movu[u_slice1:u_slice2] = p_mov_max
            P_movl[u_slice1:u_slice2] = p_mov_min

            # Moving contact velocity bounds
            if k == 0:
                dp_mov_max = np.zeros(3)
                dp_mov_min = np.zeros(3)
            else:
                dp_mov_max = np.full(3, cs.inf)
                dp_mov_min = np.full(3, -cs.inf)

            DP_movu[u_slice1:u_slice2] = dp_mov_max
            DP_movl[u_slice1:u_slice2] = dp_mov_min

            # contact positions
            p_k = np.hstack(contacts)  # start with initial contacts (4x3), but not the moving one

            # time region around max clearance time
            clearance_region = (clearance_time / self._dt - 4 <= k <= clearance_time / self._dt + 4)

            if clearance_region:
                p_k[3 * swing_id:3 * (swing_id + 1)] = clearance_swing_position

            elif k > clearance_time / self._dt + 4:
                # after the swing, the swing foot is now at swing_tgt
                p_k[3 * swing_id:3 * (swing_id + 1)] = swing_tgt

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
        # check improvement by excellent initial guess
        # v0 = np.array([0.107729, 9.07e-05, -0.02118, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10677777023401071, 4.3290169388274255e-05, -0.021567533860261792, -0.014268446489839501, -0.0007111474591758862, -0.005813007903926874, -0.14268446489839498, -0.00711147459175886, -0.05813007903926873, 0.1009028810128242, -0.00029579972490668835, -0.02380796215755845, -0.0453179988482791, -0.002952906036896781, -0.016167400747669227, -0.16781105868600094, -0.015306111185450087, -0.04541384939815477, 0.08870965337284191, -0.0012427977380125947, -0.027725048856599908, -0.07548131103457582, -0.006768547004250019, -0.021880114050467893, -0.1338220631769664, -0.022850298488082302, -0.0117132836298319, 0.07131010448960602, -0.0030878555932826327, -0.03214731679938409, -0.09664840486169013, -0.011853743971742306, -0.021402462677843712, -0.07784887509417657, -0.028001671186840563, 0.01648979735607372, 0.05080254443256604, -0.006025315644003575, -0.03600702017366406, -0.1065317036228019, -0.01755424569864546, -0.016739604994119455, -0.02098411251694111, -0.02900334608219101, 0.030138779481168842, 0.02939700350277575, -0.010084434139412823, -0.03874768171936947, -0.10592129544955642, -0.022877951425628693, -0.010644591145459055, 0.027088194249395774, -0.024233711187641317, 0.03081135900543517, 0.009000577123630716, -0.01506679808511874, -0.04030116575462649, -0.09681262421300221, -0.02655618521556719, -0.0050942141384807986, 0.06399851811614629, -0.012548626711743612, 0.024692411064347383, -0.008901267325710668, -0.020504254598577482, -0.040869395763390576, -0.08130227012573091, -0.027194614599572415, -0.0008042629609343766, 0.09110502275656672, 0.006164332871691354, 0.018207100711116836, -0.02321105268723193, -0.02565957456694497, -0.04068427726477496, -0.061152742447013755, -0.023557003613536607, 0.002564593329991491, 0.11039025403060479, 0.030211776988666703, 0.015481462198141843, -0.03315772982063682, -0.029600799924430536, -0.03984912983952484, -0.03793369751010625, -0.015025550834076931, 0.005849878498954587, 0.12180019533847025, 0.05510275080593003, 0.01737138949148911, -0.03859080772342539, -0.03168246802086549, -0.038334246801273945, -0.017808793055463065, -0.006684194858963399, 0.009286349626705477, 0.07944884920796158, 0.028310808945205285, 0.01699332178601979, -0.04081221969962987, -0.032602664163715274, -0.036152013251156655, -0.005648478452937328, -0.003265633319340514, 0.012461471819746483, 0.04215429681729581, 0.005874806451023565, 0.014757900144390278, -0.041261140635838196, -0.03321837062114428, -0.0334156070272378, 0.0003477131810201894, -0.0032918108678564786, 0.014647359704850748, 0.017807619522279352, -0.006136581936183205, 0.007100978706652352, -0.040891563908992276, -0.03399603795487701, -0.03042690224730128, 0.003067462588420468, -0.0044677300766596305, 0.014825754418681187, 0.009389874551723434, -0.005622610151848318, -0.0053170315683479415, -0.0401142446568528, -0.03498958722292035, -0.027651195112012233, 0.004585876150078873, -0.005405517852146112, 0.012515801348808123, 0.005794261064860614, -0.0037552676030164957, -0.017782499130382688, -0.03909206828877031, -0.036139537744756645, -0.025572317564937212, 0.005581467114593635, -0.006062695362950485, 0.007929810421547424, 0.004161648580287002, -0.0028165075050272308, -0.028077410142224302, -0.0378951890601704, -0.037406402829559315, -0.024602701270573183, 0.006374089341782768, -0.006595934795636282, 0.0014923645865879632, 0.0037645736916043404, -0.002515886821830739, -0.03629704820737031, -0.03654408669376995, -0.03877446141701753, -0.025067249543173395, 0.007141899443280821, -0.007077420538417686, -0.006323248441442012, 0.003913527323376183, -0.0022989706059833117, -0.04185908207292944, -0.03503724277281138, -0.04023124169221281, -0.027170270968998603, 0.007927507195479347, -0.007466965990495459, -0.014712916297201195, 0.003942550198609058, -0.0015964839147944096, -0.0420375964846624, -0.033376731029506765, -0.041745551705919065, -0.030891763628117335, 0.008658406738749687, -0.007621069833123435, -0.022192797643912326, 0.003366445234094346, 5.544548851464458e-05, -0.03276121698244892, -0.03158511123288835, -0.04324875955057485, -0.0358469483188291, 0.009220838948367413, -0.007311522552441208, -0.026666053374606907, 0.0022578768620829287, 0.003040027318307629, -0.011971340324496928, -0.029697548272281417, -0.04461759406165347, -0.04124282724787176, 0.009645978826160796, -0.006213475293127643, -0.026408943153976325, 0.0019935219158508937, 0.007940445274828022, 0.014542442530802729, -0.02764787370948048, -0.04559164864643284, -0.04610976571424006, 0.011253808598107401, -0.00297791271291819, -0.021640434940652133, 0.01408477580361515, 0.024415180527266506, 0.0331426396024392, -0.024936578029359, -0.045502602021783745, -0.049735745898890384, 0.016753340425245882, 0.004850006742846173, -0.014423096848694463, 0.04091054246776967, 0.05386401403037712, 0.03903074131713749, -0.020541287316057175, -0.04322335309794054, -0.051863434835746874, 0.028331625602258617, 0.019102318968918067, -0.006972214487172247, 0.07487230930235768, 0.08865910823034179, 0.035478082298084676, -0.013620534917922374, -0.03787580171989934, -0.05264005432650609, 0.03966080383726901, 0.033142721909747555, -0.0012526716168521927, 0.03841947304774631, 0.051744921177953006, 0.021717346405115866, -0.0051480395617769644, -0.030452479700854695, -0.05253997017422306, 0.04392387536286847, 0.039889894348379244, 0.001834870877438383, 0.004211242208248238, 0.015726803208363886, 0.009158078537789894, 0.003547905098163042, -0.022349855954852834, -0.052048498133516766, 0.04217029495253831, 0.04018688717243305, 0.0027865310019386503, -0.021747046311549917, -0.012756874967825831, 0.00035852270721278265, 0.011441838229213562, -0.014690839475679162, -0.05151416269203861, 0.03624311169183617, 0.03578716033952154, 0.002406117347573805, -0.037524786295471485, -0.031240393361289313, -0.004162659250861236, 0.017898614928731436, -0.008216228208696977, -0.05112332782769728, 0.02811790573864296, 0.028668887661818597, 0.0014665541950584673, -0.04372727323646062, -0.03994233341574013, -0.005232972274292139, 0.022657140644506402, -0.0032855492070494466, -0.05092792894186883, 0.019514801582984662, 0.02061664304264978, 0.0005211721247390033, -0.0423037683201224, -0.040580112775948037, -0.0042208484289025005, 0.02575807058270846, 6.005181559400562e-05, -0.0508961335664119, 0.011714722739073782, 0.013008740531947031, -0.00014332877473381914, -0.03569702011898639, -0.03549891233107945, -0.0024241605658257234, 0.02744971394529875, 0.002007843414672087, -0.050962115431876086, 0.005514906972605471, 0.00674928415538511, -0.0004606543759125554, -0.02630113754569671, -0.02709565143453977, -0.0007490954459616398, 0.02809418131991653, 0.002879791049273753, -0.05106184576278725, 0.001267310428625408, 0.0022902113517087423, -0.0004997366672460542, -0.016174827894103915, -0.0174950766022239, 0.00035827253262665163, 0.028085666681096175, 0.0030484235767519055, -0.0511515262312008, -0.0010448576501457412, -0.0003014271310228111, -0.0003815609449737332, -0.006946852893607575, -0.008421308225091632, 0.0008234846900965579, 0.027785074504270854, 0.0028678273613232364, -0.05121155323039772, -0.0017244820627275962, -0.001263958146875253, -0.00021963156701602595, 0.00015060876778902462, -0.001204001933432787, 0.0007958090894805143, 0.027470046337370314, 0.0026199028656255234, -0.05124146889317, -0.0012915192548318274, -0.001070550948371908, -8.90527165001694e-05, 0.004179019311168662, 0.0031380739184662335, 0.000509979415678051, 0.027296583321343364, 0.002472858446748201, -0.05125121121842991, -0.00043680866185748056, -0.0003783717782626423, -1.902738746618215e-05, 0.004368086618574806, 0.0037837177826264235, 0.0001902738746618215, 0.0272674627438862, 0.0024476336615306916, -0.05125247971092766, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.713422324491975, -0.0355573729587943, -0.2906503951963436, -0.12563296893802972, -0.04097318296845614, 0.06358114820556982, 0.1699449775451726, -0.037720936513161066, 0.16850282884161433, 0.2798659404139492, -0.025756863493791308, 0.1410154049295281, 0.28432381288617725, -0.0050083744767522335, 0.0682449106254756, 0.24036153383168443, 0.023848174472748455, 0.0033628976213316386, 0.18455161933375253, 0.05842542237948853, -0.030594739705438925, 0.13553252320210218, 0.09356479791717483, -0.03242655176615273, 0.09642615637019031, 0.12023722058487674, -0.013628192564874964, 0.057049706539327366, 0.12445486908631663, 0.00944963646673633, -0.21175673065254336, -0.13395970930362372, -0.0018903385273465988, -0.1864727619533288, -0.11218001247090859, -0.011177108208147568, -0.12173338647508229, -0.06005694193603384, -0.038284607188689625, -0.04208872485277959, 0.0025698589216744372, -0.06209005137500147, -0.017978067434314106, 0.00933671274415911, -0.06232733781017373, -0.008163062422868055, 0.004693800489946325, -0.051474555059208046, -0.0019853744434133063, 0.0015031034159824588, -0.04109819032573003, 0.00074476815885921, 0.0010845810792371347, -0.02781016932779567, 0.00014511437616437684, 0.0035124334559445123, -0.0008925720586647945, -0.002880524822573561, 0.00825964701654527, 0.046381897511067416, -0.005542841860057087, 0.01492290914896492, 0.10394938328975994, -0.0013217747311601758, 0.02450208978260197, 0.13256891427649828, 0.06045626943882128, 0.08237367626219243, 0.09300098535818235, 0.1341288333207726, 0.14724416751555305, 0.02944050857349144, 0.16980883417293996, 0.17397547099982336, -0.01776329509526405, -0.1822641812730568, -0.18457093526194393, -0.06880367946484403, -0.17104115419749036, -0.18009058984794557, -0.06279633933662986, -0.12979144259899078, -0.14241839088094857, -0.04399777915288556, -0.07888869991960781, -0.0924175919673174, -0.022605909790370093, -0.03101243470494568, -0.04350970027225411, -0.0053515651171545135, 0.007117524581691079, -0.0031888968010394822, 0.00506061922694819, 0.03303374100568004, 0.025406002224342913, 0.008983439315383887, 0.0469794128664484, 0.04201630448269839, 0.008375325599320417, 0.05063154825796395, 0.048002874161579334, 0.005536839892941457, 0.046139875002481706, 0.04536884188566134, 0.0023260607873495315, 0.035487308306983, 0.03608653145829423, -0.00013837800308021765, 0.020142052716898187, 0.021710379259495097, -0.0014291483690123165, 0.0009453365370307167, 0.003228219320800951, -0.0015985277050811475, -0.021840433092874027, -0.01891858891313212, -0.0009513693733091076, 0.0, 0.0, 0.0, -5.925093137244243e-25, -1.9732975840131673e-25, 304.81121072768497, 5.925098262190022e-25, -1.9765278269911862e-25, 304.5895629151722, 1.9752663279199465e-25, 1.9749127055021915e-25, 161.28454336517223, -1.9752742512577745e-25, 1.9749127055021915e-25, 161.26468299197063, -3.363202040013208, -0.194451522877946, 309.69925893782573, -3.414310042660554, -0.194451522877946, 308.3310271268682, -3.363202040013208, -0.14334352023059985, 154.24467801134261, -3.414310042660554, -0.14334352023059985, 154.152678415233, -3.9314281537327136, -0.41760463071424814, 307.6565384908878, -4.039597133852331, -0.41760463071424814, 305.70289950336263, -3.9314281537327136, -0.30943565059463096, 157.21278169857547, -4.039597133852331, -0.30943565059463096, 157.06346461434956, -3.1038565709897927, -0.6171120185551138, 298.4046807235394, -3.252691429916111, -0.6171120185551138, 297.526337765673, -3.1038565709897927, -0.46827715962879557, 167.50409695378866, -3.252691429916111, -0.46827715962879557, 167.40212261216493, -1.7730061844339389, -0.7409442897402181, 283.29622694105075, -1.9248153825394483, -0.7409442897402181, 285.81440984490376, -1.7730061844339389, -0.5891350916347087, 181.95314542680694, -1.9248153825394483, -0.5891350916347087, 182.45274853606554, -0.44409104029575913, -0.7431111014336287, 264.6086974331214, -0.5526543042589437, -0.7431111014336287, 272.58622376036124, -0.44409104029575913, -0.6345478374704443, 197.43859697835995, -0.5526543042589437, -0.6345478374704443, 200.1796658788685, 0.6582470388772375, -0.5904530661605691, 244.97107047145303, 0.6284421879690618, -0.5904530661605691, 259.2286264009164, 0.6582470388772375, -0.5606482152523936, 211.23558570567494, 0.6284421879690618, -0.5606482152523936, 219.44179652747195, 1.491081145461914, -0.2691462246073506, 226.72528362531799, 1.5488484650550345, -0.2691462246073506, 246.5435084826768, 1.491081145461914, -0.32691354420047103, 221.46815825717815, 1.5488484650550345, -0.32691354420047103, 239.55882868594009, 2.1022260155869406, 0.20792118058418885, 211.4195714268085, 2.2252625653499787, 0.20792118058418885, 235.16663617240485, 2.1022260155869406, 0.08488463082115048, 227.50404979841554, 2.2252625653499787, 0.08488463082115048, 259.5894171699274, 2.549455896907639, 0.789842339800059, 199.77891454056345, 2.694081169546088, 0.789842339800059, 225.82277048846422, 2.549455896907639, 0.6452170671616094, 229.93550529338816, 2.694081169546088, 0.6452170671616094, 277.8835485864076, 0.0, 0.0, 0.0, 3.6533306833092425, 1.3375694373698188, 411.35576681906025, 4.264357190536191, 1.948595944596767, 422.2445160886402, 3.6533306833092425, 1.948595944596767, 100.0, 0.0, 0.0, 0.0, 2.361793412253781, 0.5883353246014954, 407.9790982681291, 2.8240538502487884, 1.0505957625965032, 421.91434690127977, 2.361793412253781, 1.0505957625965032, 103.67092040026301, 0.0, 0.0, 0.0, 1.2297743772081906, -0.024187839729940517, 408.3538026961823, 1.545109443226721, 0.2911472262885896, 422.66071973647195, 1.2297743772081906, 0.2911472262885896, 102.33747808106288, 0.0, 0.0, 0.0, 0.49778972618327993, -0.3265615453569339, 409.68305870848144, 0.6961444022499785, -0.12820686929023534, 422.6254886080846, 0.49778972618327993, -0.12820686929023534, 100.3160456605661, 0.0, 0.0, 0.0, 0.25655300389975877, -0.25963536861816333, 410.43357852286886, 0.3789320746142087, -0.1372562979037134, 421.0113040187196, 0.25655300389975877, -0.1372562979037134, 100.0, 0.0, 0.0, 0.0, 0.15767434464782148, -0.170537985574385, 411.2773385113459, 0.23510611186611535, -0.09310621835609106, 418.98332466199025, 0.15767434464782148, -0.09310621835609106, 100.0, 0.0, 0.0, 0.0, 0.11300217059828685, -0.12675613988079867, 412.3533040570928, 0.16935227393069147, -0.07040603654839411, 416.9293425516322, 0.11300217059828685, -0.07040603654839411, 100.0, 0.0, 0.0, 0.0, 0.10236388719294971, -0.11336497544034888, 413.6309135635756, 0.15290672631651292, -0.06282213631678567, 414.8708674202852, 0.10236388719294971, -0.06282213631678567, 100.0, 0.0, 0.0, 0.0, 0.10733618333781807, -0.10598509966099366, 415.1726618444159, 0.15711272904510123, -0.05620855395371047, 412.80072592413154, 0.10733618333781807, -0.05620855395371047, 100.0, 0.0, 0.0, 0.0, 0.11020145745496916, -0.0798472549704583, 417.1610116300805, 0.15413935395792216, -0.035909358467505294, 410.79541728170574, 0.11020145745496916, -0.035909358467505294, 100.0, 0.0, 0.0, 0.0, 0.09820339228459202, -0.015045639787160774, 419.8166492019371, 0.12340551266977882, 0.010156480598026004, 409.0210357833074, 0.09820339228459202, 0.010156480598026004, 100.0, 0.0, 0.0, 0.0, 0.0746507759698763, 0.10257021575424202, 423.19736961669383, 0.06519674995812562, 0.09311618974249133, 407.61535367340423, 0.0746507759698763, 0.09311618974249133, 100.0, 0.0, 0.0, 0.0, 0.08168515878616057, 0.2885613632713186, 426.8858257769254, 0.026014264433513717, 0.23289046891867177, 406.4457059857599, 0.08168515878616057, 0.23289046891867177, 100.0000002777411, 0.0, 0.0, 0.0, 0.48579352530496855, 0.8526986330777503, 429.33006325273595, 0.36646665073350215, 0.733371758506284, 404.41682007357053, 0.48579352530496855, 0.733371758506284, 101.35166743592524, 0.0, 0.0, 0.0, 1.3643616106903511, 1.843415976050565, 430.7831925818252, 1.1577783130574166, 1.6368326784176306, 402.11751095733104, 1.3643616106903511, 1.6368326784176306, 102.75721688597189, 0.0, 0.0, 0.0, 2.473540172874527, 3.0127058505605575, 433.43762773303735, 2.165789037974925, 2.704954715660956, 401.8827910186663, 2.473540172874527, 2.704954715660956, 100.0, 0.9295433380065805, 1.208200841616395, 189.55016761714637, 0.8953816316418641, 1.2130810856938958, 228.66458994288985, 0.929543338245591, 1.2472427922976224, 222.15483108286097, 0.8953816316418641, 1.2472427922976224, 293.6435592655889, 0.11089248831533369, 0.3603056287555034, 198.70762736577274, 0.08914151648929929, 0.3634129106794546, 232.06665131872944, 0.11089248848965032, 0.3851638826798056, 225.67385013807856, 0.08914151648929929, 0.3851638826798056, 276.37188863850923, -0.5098146287648615, -0.3110844420727563, 208.85323272964058, -0.5231700710902173, -0.30917652158241343, 234.03028289636683, -0.509814628651946, -0.29582107914414213, 228.55845677453325, -0.5231700710902173, -0.29582107914414213, 260.5420872566446, -0.8871982597817974, -0.7468352031987928, 218.64932328460867, -0.8952290892836896, -0.7456879417499476, 234.58015694933306, -0.8871982597206144, -0.737657112186872, 230.459553975456, -0.8952290892836896, -0.737657112186872, 247.86551316177048, -1.0361448576763286, -0.9515178464370769, 226.92928076270894, -1.0409006210666487, -0.9508384516277555, 234.23110960560535, -1.0361448576541332, -0.94608268821524, 231.55098265160711, -1.0409006210666487, -0.94608268821524, 238.74149461402087, -1.0033336364014571, -0.9654544384583216, 232.9851758465175, -1.0060953588024826, -0.9650599066837651, 233.56300152524526, -1.0033336364052057, -0.9622981842864882, 232.16355934648666, -1.0060953588024826, -0.9622981842864882, 232.83728268100484, -0.8470255014839396, -0.8440447641068981, 236.66707597047665, -0.8485829541588022, -0.8438222708863092, 232.96365263330983, -0.847025501502163, -0.8422648182296701, 232.55028102328075, -0.8485829541588022, -0.8422648182296701, 229.53869511917938, -0.62422540962208, -0.6440397444226947, 238.28153698249648, -0.625078623786619, -0.643917856713361, 232.58828143407408, -0.6242254096458694, -0.6430646425726112, 232.83228684576432, -0.625078623786619, -0.6430646425726112, 228.1767306702988, -0.3839146862620687, -0.41579643324662824, 238.40043767422827, -0.3843896386961962, -0.41572858292873777, 232.43212173007024, -0.38391468628541087, -0.4152536305179524, 233.04064493982304, -0.3843896386961962, -0.4152536305179524, 228.11083154647795, -0.16483657010631192, -0.20018965345220155, 237.6761064808202, -0.16513894233020307, -0.20014645744663545, 232.4206014910592, -0.16483657012600167, -0.19984408524243408, 233.17412519111073, -0.16513894233020307, -0.19984408524243408, 228.75739788256908, 0.003695971396175203, -0.0287395618721263, 236.7060162627924, 0.0034579450814412236, -0.02870555813430179, 232.47259045414435, 0.0036959713808996882, -0.028467531834843334, 233.23631745712177, 0.0034579450814412236, -0.028467531834843334, 229.61067768944213, 0.09935224271116154, 0.07440717850015795, 235.95191469143055, 0.0991511745753391, 0.07443590250214871, 232.528589151136, 0.09935224269918315, 0.07463697062599277, 233.24594403951915, 0.0991511745753391, 0.07463697062599277, 230.27200016240374, 0.10380846402902735, 0.08978266048362912, 235.69811459187497, 0.10367565035875158, 0.0898016338490774, 232.55388386831328, 0.10380846401807604, 0.08993444750840184, 233.22909398860267, 0.10367565035875158, 0.08993444750840184, 230.48698356930205, 1.0161344161353494e-11, 1.1009878869710798e-11, 236.02675361003256, -3.921747131476577e-12, -4.739224445287759e-12, 232.53287002305456, -2.31784989840034e-12, -3.1353272122115193e-12, 233.2069209483431, -3.921747131476578e-12, -3.1353272122115197e-12, 230.1834554185698])
        v0 = np.zeros(self._nvars)

        # format bounds and params according to solver
        lbv = cs.vertcat(Xl, Ul, Fl, P_movl, DP_movl)
        ubv = cs.vertcat(Xu, Uu, Fu, P_movu, DP_movu)
        lbg = cs.vertcat(*gl)
        ubg = cs.vertcat(*gu)
        params = cs.vertcat(*P)

        # compute solution-call solver
        sol = self._solver(x0=v0, lbx=lbv, ubx=ubv, lbg=lbg, ubg=ubg, p=params)

        # plot state, forces, control input, quantities to be computed by evaluate function
        x_trj = cs.horzcat(self._trj['x'])  # pack states in a desired matrix
        f_trj = cs.horzcat(self._trj['F'])  # pack forces in a desired matrix
        u_trj = cs.horzcat(self._trj['u'])  # pack control inputs in a desired matrix
        P_mov_trj = cs.horzcat(self._trj['P_mov'])  # pack moving contact trj in a desired matrix
        DP_mov_trj = cs.horzcat(self._trj['DP_mov'])  # pack moving contact trj in a desired matrix

        # return values of the quantities *_trj
        return {
            'x': self.evaluate(sol['x'], x_trj),
            'F': self.evaluate(sol['x'], f_trj),
            'u': self.evaluate(sol['x'], u_trj),
            'P_mov': self.evaluate(sol['x'], P_mov_trj),
            'DP_mov': self.evaluate(sol['x'], DP_mov_trj)
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

        for ii in range(self._N):  # loop for knots

            # control input to change in every knot
            u_old = solution['u'][self._dimu * ii:self._dimu * (ii + 1)]

            for j in range(self._n):  # loop for interpolation points

                x_all.append(x_old)  # storing state in the list 600x9

                x_next = self._integrator(x_old, u_old, delta_t)  # next state
                x_old = x_next  # refreshing the current state

        # initialize state and time lists to gather the data
        int_state = [[] for i in range(self._dimx)]  # primary dimension = number of state components
        self._t = [(ii * delta_t) for ii in range(self._N * self._n)]

        for i in range(self._dimx):  # loop for every component of the state vector
            for j in range(self._N * self._n):  # loop for every point of interpolation

                # append the value of x_i component on j point of interpolation
                # in the element i of the list int_state
                int_state[i].append(x_all[j][i])

        # ----------- force trajectory interpolation --------------
        force_func = [[] for i in range(self._dimf_tot)]  # list to store the splines
        int_force = [[] for i in range(self._dimf_tot)]  # list to store lists of points

        #test = solution['F'][1::self._dimf_tot]
        for i in range(self._dimf_tot):  # loop for each component of the force vector

            # append the spline (by casadi) in the i element of the list force_func
            force_func[i].append(cs.interpolant('X_CONT', 'linear',
                                                [self._time],
                                                solution['F'][i::self._dimf_tot]))

            # store the interpolation points for each force component in the i element of the list int_force
            # primary dimension = number of force components
            int_force[i] = force_func[i][0](self._t)

        # ----------- moving contact trajectory interpolation --------------
        mov_cont_splines = []
        mov_cont_polynomials = []
        mov_cont_points = []
        aa = []
        bb = []
        cc = []
        for i in range(3):

            # debug
            aa.append(solution['P_mov'][i::self._dimu])
            bb.append(solution['DP_mov'][i::self._dimu])
            cc.append(self._time)

            mov_cont_splines.append(cubic_spline.CubicSpline(solution['P_mov'][i::self._dimu],
                                                             solution['DP_mov'][i::self._dimu],
                                                             self._tjunctions))

            mov_cont_polynomials.append(mov_cont_splines[i].get_polys())
            mov_cont_points.append(mov_cont_splines[i].get_point_list(resol))


        # dense spline
        # poly_object = CubicSpline(randomlist, deriv, tim)
        # polynomials = poly_object.get_polys()
        # points = poly_object.get_point_list(300)
        # plot_spline(points)

        # ----------- swing leg trajectory interpolation --------------
        # swing trajectory with intemediate point
        sw_interpl = interpol.swing_trj_triangle(sw_curr, sw_tgt, clearance, sw_t, t_tot, resol)

        # swing trajectory with spline optimization for z coordinate
        # sw_interpl = interpol.swing_trj_optimal_spline(sw_curr, sw_tgt, clearance, sw_t, t_tot, resol)

        return {
            't': self._t,
            'x': int_state,
            'f': int_force,
            'p_mov': [i['p'] for i in mov_cont_points],
            'sw': sw_interpl
        }

    def print_trj(self, solution, results, resol, contacts, swing_id, t_exec=0):
        '''

        Args:
            t_exec: time that trj execution stopped (because of early contact or other)
            results: results from interpolation
            resol: interpolation resol
            contacts: contact points
            swing_id: the id of the leg to be swinged
        Returns: prints the nominal interpolated trajectories

        '''

        # Interpolated state plot
        state_labels = ['CoM Position', 'CoM Velocity', 'CoM Acceleration']
        plt.figure()
        for i, name in enumerate(state_labels):
            plt.subplot(3, 1, i + 1)
            for j in range(self._dimc):
                plt.plot(results['t'], results['x'][self._dimc * i + j], '-')
                # plt.plot(self._time, solution['x'][self._dimc * i + j::self._dimx], 'o')
            plt.grid()
            plt.legend(['x', 'y', 'z'])
            # plt.legend(['x', 'xopt', 'y', 'yopt', 'z', 'zopt'])
            plt.title(name)
        plt.xlabel('Time [s]')
        # plt.savefig('../plots/step_state_trj.png')

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
        # plt.savefig('../plots/step_forces.png')

        # Interpolated moving contact trajectory
        mov_contact_labels = ['position', 'velocity']
        plt.figure()
        #for i, name in enumerate(mov_contact_labels):
        plt.subplot(2, 1, 1)
        for k in range(3):
            plt.plot(results['t'], results['p_mov'][k], '-')
            plt.grid()
        plt.subplot(2, 1, 2)
        for k in range(3):
            plt.plot(self._time, solution['DP_mov'][k::self._dimu], '-')
            plt.grid()
        plt.title('Moving Contact trajectory')
        plt.legend(['x', 'y', 'z'])
        plt.xlabel('Time [s]')
        # plt.savefig('../plots/mov_contact.png')

        # plot swing trajectory
        # All points to be published
        N_total = int(self._N * self._dt * resol)  # total points --> total time * frequency
        s = np.linspace(0, self._dt * self._N, N_total)
        coord_labels = ['x', 'y', 'z']
        plt.figure()
        for i, name in enumerate(coord_labels):
            plt.subplot(3, 1, i + 1)
            plt.plot(s, results['sw'][name])  # nominal trj
            plt.plot(s[0:t_exec], results['sw'][name][0:t_exec])  # executed trj
            plt.grid()
            plt.legend(['nominal', 'real'])
            plt.title('Trajectory ' + name)
        plt.xlabel('Time [s]')
        # plt.savefig('../plots/step_swing.png')

        # plot swing trajectory in two dimensions Z- X
        plt.figure()
        plt.plot(results['sw']['x'], results['sw']['z'])  # nominal trj
        plt.plot(results['sw']['x'][0:t_exec], results['sw']['z'][0:t_exec])  # real trj
        plt.grid()
        plt.legend(['nominal', 'real'])
        plt.title('Trajectory Z- X')
        plt.xlabel('X [m]')
        plt.ylabel('Z [m]')
        # plt.savefig('../plots/step_swing_zx.png')

        # Support polygon and CoM motion in the plane
        SuP_x_coords = [contacts[k][1] for k in range(4) if k not in [swing_id]]
        SuP_x_coords.append(SuP_x_coords[0])
        SuP_y_coords = [contacts[k][0] for k in range(4) if k not in [swing_id]]
        SuP_y_coords.append(SuP_y_coords[0])
        plt.figure()
        plt.plot(results['x'][1], results['x'][0], '-')
        plt.plot(SuP_x_coords, SuP_y_coords, 'ro-')
        plt.grid()
        plt.title('Support polygon and CoM')
        plt.xlabel('Y [m]')
        plt.ylabel('X [m]')
        plt.xlim(0.5, -0.5)
        # plt.savefig('../plots/mov_contact.png')
        plt.show()


if __name__ == "__main__":
    start_time = time.time()

    w = Walking(mass=95, N=40, dt=0.2)

    # initial state =
    c0 = np.array([0.107729, 0.0000907, -0.02118])
    dc0 = np.zeros(3)
    ddc0 = np.zeros(3)
    x_init = np.hstack([c0, dc0, ddc0])

    foot_contacts = [
        np.array([0.35, 0.35, -0.7187]),  # fl
        np.array([0.35, -0.35, -0.7187]),  # fr
        np.array([-0.35, 0.35, -0.7187]),  # hl
        np.array([-0.35, -0.35, -0.7187])  # hr
    ]

    # swing id from 0 to 3
    sw_id = 1

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
    # check time needed
    end_time = time.time()
    print('Total time is:', (end_time - start_time) * 1000, 'ms')

    # debug
    print("X0 is:", x_init)
    print("contacts is:", foot_contacts)
    print("swing id is:", sw_id)
    print("swing target is:", swing_target)
    print("swing time:", swing_time)

    # interpolate the values, pass values and interpolation resolution
    res = 300
    interpl = w.interpolate(sol, foot_contacts[sw_id], swing_target, step_clear, swing_time, res)

    # check time needed
    end_time = time.time()
    print('Total time for nlp formulation, solution and interpolation:', (end_time - start_time) * 1000, 'ms')

    print("Solution is:")
    print("State is:", sol['x'])
    print("Control is:", sol['u'])
    print("Forces are:", sol['F'])
    print("Moving contact is:", sol['P_mov'])

    # print the results
    w.print_trj(sol, interpl, res, foot_contacts, sw_id)


