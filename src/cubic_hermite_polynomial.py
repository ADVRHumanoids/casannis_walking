import casadi as cs
import numpy as np
import matplotlib.pyplot as plt


def plot_poly(point_list):
    '''
    Prints points that have been generated by sampling a cubic polynomial and its derivative
    :param point_list: dictionary with list of points and list of timings
    :return: t-value plot of the points
    '''

    subplot_labels = ['p', 'dp', 'ddp']
    plt.figure()
    for i, name in enumerate(subplot_labels):
        plt.subplot(3, 1, i + 1)
        plt.plot(point_list['t'], point_list[name])
        plt.ylabel(name)
        plt.grid()
        plt.title('Cubic Polynomial and Derivative')
    plt.xlabel('Time [s]')
    plt.show()


def plot_spline(point_list):
    '''
    Prints points that have been generated by sampling a cubic spline and its derivative
    :param point_list: dictionary with list of points and list of timings
    :return: t-value plot of the points
    '''

    subplot_labels = ['p', 'dp', 'ddp']
    plt.figure()
    for i, name in enumerate(subplot_labels):
        plt.subplot(3, 1, i+1)
        plt.plot(point_list['t'], point_list[name])
        plt.ylabel(name)
        plt.grid()
        plt.title('Spline and Derivative')
    plt.xlabel('Time [s]')
    plt.show()


def integrate_numerically(poly_obj, t_junctions):

    # simpson's 1/3 rule
    # integral(f) from a to b = (b-a)/6 * (f(a) + 4 * f((a+b)/2) + f(b))
    numerical_integration = (t_junctions[1]-t_junctions[0])/6.0 * (
            poly_obj(t_junctions[0]) + 4.0 * poly_obj((t_junctions[0]+t_junctions[1])/2) + poly_obj(t_junctions[1])
    )

    return numerical_integration


class CubicPolynomial:
    '''
    Construct a class of a cubic polynomial from zero and first derivative values
    at the beginning and at the end and the duration
    '''

    def __init__(self, p_list, v_list, T, t0=0):
        '''
        Construct the object of the class
        :param p_list: list of position values
        :param v_list: list of first derivative values
        :param T: duration of the polynomial
        :param t0: global starting time of the polynomial (default zero)
        '''

        # save in the class
        self._p0 = p_list[0]
        self._p1 = p_list[1]
        self._v0 = v_list[0]
        self._v1 = v_list[1]
        self._t0 = t0
        self._T = T

        # poly coefficients w/o time offset
        # p(t) = d * t^3 + c * t^2 + b * t + a
        a = self._p0
        b = self._v0
        c = - (3*self._p0 - 3*self._p1 + 2*T*self._v0 + T*self._v1)/T**2
        d = (2*self._p0 - 2*self._p1 + T*self._v0 + T*self._v1)/T**3

        # poly coefficients / general case
        # p(t) = d * (t-t0)^3 + c * (t-t0)^2 + b * (t-t0) + a
        self._a = a - b*t0 + c*t0**2 - d*t0**3
        self._b = b - 2*c*t0 + 3*t0**2*d
        self._c = c - 3*d*t0
        self._d = d

    def get_poly_from_coeffs(self):
        '''
        Get the polynomial from the coefficients
        :return: cubic polynomial object
        '''

        coeff_list = [self._a, self._b, self._c, self._d]

        # convert to polynomial function
        self._cubic_polynomial_object = np.polynomial.polynomial.Polynomial(coeff_list)

        return self._cubic_polynomial_object

    def get_first_deriv_poly(self):
        '''
        Get the polynomial of the derivative from the coefficients
        :return: quadratic polynomial object
        '''

        coeff_list = [self._b, 2.0 * self._c, 3.0 * self._d]

        # convert to polynomial function
        polynomial_object = np.polynomial.polynomial.Polynomial(coeff_list)

        return polynomial_object

    def get_second_deriv_poly(self):
        '''
        Get the polynomial of the derivative from the coefficients
        :return: quadratic polynomial object
        '''

        coeff_list = [2.0 * self._c, 6.0 * self._d]

        # convert to polynomial function
        polynomial_object = np.polynomial.polynomial.Polynomial(coeff_list)

        return polynomial_object

    def evaluate_second_derivative(self, t):

        coeff_list = [2.0 * self._c, 6.0 * self._d]

        polynomial_object = np.polynomial.polynomial.Polynomial(coeff_list)

        value = polynomial_object(t)

        return value

    def get_trajectory(self, timings):
        '''
        Generate a list of points (pairs of time and value) by sampling the cubic polynomial
        :param polynomial: the polynomial object
        :param timings: timings at which the polynomial should be sampled
        :return: dictionary with list of timings and list of values of the generated points
        '''

        coeff_cubic = [self._a, self._b, self._c, self._d]
        coeff_first_deriv = [self._b, 2.0 * self._c, 3.0 * self._d]
        coeff_second_deriv = [2.0 * self._c, 6.0 * self._d]

        # find polynomials for cubic and derivatives
        cubic_polynomial = np.polynomial.polynomial.Polynomial(coeff_cubic)
        first_derivative = np.polynomial.polynomial.Polynomial(coeff_first_deriv)
        second_derivative = np.polynomial.polynomial.Polynomial(coeff_second_deriv)

        # sample polynomials
        cubic_trj = cubic_polynomial(np.array(timings))
        first_derivative_trj = first_derivative(np.array(timings))
        second_derivative_trj = second_derivative(np.array(timings))

        return {
            't': timings,
            'p': cubic_trj,
            'dp': first_derivative_trj,
            'ddp': second_derivative_trj
        }

    def get_coeffs_of_squared_deriv(self, desired_derivative=0):
        """
        This function returns the coefficients of the squared cubic polynomial or its squared derivatives
        i.e. poly ^ 2 = (d*x^3 + c*x^2 + b*x + a) ^ 2
        :param desired_derivative: the derivative of the cubic polynomial that we want
        :return: list of coefficients of the resulted polynomial
        """
        # f^2 = a^2 + 2*a*b*x + 2*a*c*x^2 + 2*a*d*x^3 + b^2*x^2 + 2*b*c*x^3 + 2*b*d*x^4 + c^2*x^4 + 2*c*d*x^5 + d^2*x^6
        # df^2 = b^2 + 4*b*c*x + 6*b*d*x^2 + 4*c^2*x^2 + 12*c*d*x^3 + 9*d^2*x^4
        # ddf^2 = 4*c^2 + 24*c*d*x + 36*d^2*x^2

        # dictionary from which we select the desired polynomial coeffs
        # these have been computed in matlab symbolically
        switcher = {
            0: [self._a**2.0,
                2.0*self._a*self._b,
                2.0*self._a*self._c + self._b**2.0,
                2.0*self._a*self._d + 2.0*self._b*self._c,
                2.0*self._b*self._d + self._c**2.0,
                2.0*self._c*self._d,
                self._d**2.0],

            1: [self._b**2.0,
                4.0*self._b*self._c,
                6.0*self._b*self._d + 4.0*self._c**2.0,
                12.0*self._c*self._d,
                9.0*self._d**2.0],

            2: [4.0*self._c**2.0,
                24.0*self._c*self._d,
                36.0*self._d**2.0],
        }

        # Get the function from switcher dictionary
        desired_poly_coeffs = switcher.get(desired_derivative, lambda: "Invalid derivative")

        return desired_poly_coeffs

    def integrate_squared_deriv(self, desired_derivative=0):
        """
        This function computes the numerical integral of the squared cubic polynomial or its squared derivatives
        :param desired_derivative: the order of the derivative to integrate
        :return: Value of the integral computed numerically (see integrate_numerically function)
        """

        # Get the function from switcher dictionary
        desired_poly_object = np.polynomial.polynomial.Polynomial(self.get_coeffs_of_squared_deriv(desired_derivative))

        timings = [self._t0, self._t0 + self._T]

        numerical_integral = integrate_numerically(desired_poly_object, timings)

        return numerical_integral


class CubicSpline:
    '''
    Construct a class for Cubic Spline, that is a sequence of cubic polynomials
    '''
    def __init__(self, p_list, v_list, T_list):
        '''
        Create the object from the list of zero and first derivative values polynomials' junctions
        :param p_list:  list of points at junctions
        :param v_list: list of first derivative values at junctions
        :param T_list: list of global times at junctions (not durations)
        '''

        self._junction_num = len(p_list)
        self._poly_num = len(p_list) - 1    # number of polynomials
        self._p_list = p_list
        self._v_list = v_list
        self._T_list = T_list
        self._t_total = T_list[-1] - T_list[0]
        self._polynomials = []      # list to host polynomial objects

        self._durations = []    # list to host polynomials' durations
        self._frac_dur = []     # list to host fraction of polynomial durations wrt total spline duration
        for i in range(self._poly_num):
            self._durations.append(self._T_list[i+1] - self._T_list[i])
            self._frac_dur.append(self._durations[i] / self._t_total)

    def get_poly_objects(self):
        '''
        Get the list of polynomial objects that comprise the spline
        :return: list of polynomial objects
        '''
        # loop over polynomials
        for i in range(self._poly_num):

            # construct using the CubicPolynomial class
            self._polynomials.append(CubicPolynomial(self._p_list[i:i + 2], self._v_list[i:i + 2],
                                                     self._durations[i], self._T_list[i]))

        return self._polynomials

    def get_spline_trajectory(self, resolution):
        '''
        Generate points by sampling the spline, in particular sampling each polynomial of the spline
        :param resolution: desired points per second to be generated
        :return: dictionary that includes list of points and list of timings
        '''

        # number of points to be generated
        point_num = int(resolution * self._t_total)

        # sample evenly the spline time
        point_tlist = np.linspace(self._T_list[0], self._T_list[-1], point_num).tolist()

        # find indices to split point_tlist based on the corresponding polynomial
        indice_list = []
        for i in range(self._poly_num):
            indice_list.append(next(x[0] for x in enumerate(point_tlist) if x[1] >= self._T_list[i]))
        indice_list.append(point_num)

        # list of of lists for timings, #sublists = #polynomials
        timings_lists = []
        for i in range(self._poly_num):
            timings_lists.append(point_tlist[(indice_list[i]):(indice_list[i+1])])

        # list of points using the CubicSpline class
        trajectory = []
        trajectory_cubic = []
        trajectory_first_derivative = []
        trajectory_second_derivative = []
        for i in range(self._poly_num):

            # Use the methods of CubicPolynomial class
            trajectory.append(self._polynomials[i].get_trajectory(timings_lists[i]))

            # save points from cubics and derivatives
            trajectory_cubic.append(trajectory[i]['p'])
            trajectory_first_derivative.append(trajectory[i]['dp'])
            trajectory_second_derivative.append(trajectory[i]['ddp'])

        # convert them to flat lists
        trajectory_cubic = [i.tolist() for i in trajectory_cubic]
        trajectory_cubic = [item for sublist in trajectory_cubic for item in sublist]

        trajectory_first_derivative = [i.tolist() for i in trajectory_first_derivative]
        trajectory_first_derivative = [item for sublist in trajectory_first_derivative for item in sublist]

        trajectory_second_derivative = [i.tolist() for i in trajectory_second_derivative]
        trajectory_second_derivative = [item for sublist in trajectory_second_derivative for item in sublist]

        return {
           't': point_tlist,
           'p': trajectory_cubic,
           'dp': trajectory_first_derivative,
           'ddp': trajectory_second_derivative
        }


if __name__ == "__main__":

    # symbolic
    '''sym_t = cs.SX
    p0 = sym_t.sym('p0', 1)
    v0 = sym_t.sym('v0', 1)
    p1 = sym_t.sym('p1', 1)
    v1 = sym_t.sym('v1', 1)

    poly_object = CubicPolynomial([p0, p1], [v0, v1], 1)
    polynomial = poly_object.get_poly_from_coeffs()
    points = poly_object.get_point_list(polynomial, 0.0, 100)
    '''

    # dense polynomial
    poly_object = CubicPolynomial([0.0, 1.0], [0.0, 0.0], 1.0)
    polynomial = poly_object.get_poly_from_coeffs()
    timelist = np.linspace(0, 1, 10).tolist()
    points = poly_object.get_trajectory(timelist)
    plot_poly(points)
    value = poly_object.evaluate_second_derivative(0.0)
    #print(value)
    integration = poly_object.integrate_squared_deriv(2)
    print("Computed numerical integral: ", integration)

    # dense spline
    # poly_object = CubicSpline([0.0, 2.0, 3.5, 8.0], [0.0, 0.0, 1.0, 0.0], [1.0, 2.0, 3.5, 4])
    # polynomials = poly_object.get_poly_objects()
    # points = poly_object.get_spline_trajectory(300)
    # plot_spline(points)