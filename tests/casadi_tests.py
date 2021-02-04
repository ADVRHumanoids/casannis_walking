import casadi as cs
import numpy as np
import scipy.interpolate as ip
from matplotlib import pyplot as plt
from scipy.stats import norm

a = "[0.5,2.5]"
a = a.rstrip(']').lstrip('[').split(',')
a = [float(i) for i in a]
print(a)
x=a[0]
print(x)
'''
#initialize a normal distribution with frozen in mean=-1, std. dev.= 1
rv = norm(loc = -1, scale = 55.0)
rv1 = norm(loc = 0., scale = 2.0)
rv2 = norm(loc = 2., scale = 3.0)
print(rv.pdf(-10))

x = np.arange(-10, 10, .1)

#plot the pdfs of these normal distributions
plt.figure()
plt.plot(x, rv.pdf(x))
plt.show()
'''
'''
# select a sym type
sym_t = cs.SX
a0 = sym_t.sym('a0', 1)
a1 = sym_t.sym('a1', 1)
a2 = sym_t.sym('a2', 1)
a3 = sym_t.sym('a3', 1)
a4 = sym_t.sym('a4', 1)
a5 = sym_t.sym('a5', 1)

t = sym_t.sym('t', 1)

# initial and final position, vel, acc


# the 5th order polynomial expression
spline = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5

# wrap the polynomial expression in a function
p = cs.Function('p', [t, a0, a1, a2, a3, a4, a5], [spline], ['t', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5'], ['spline'])
#spline = cs.Function('spline', [t], [p], ['t'], ['p'])

# initial and final conditions
p0 = sym_t.sym('p0', 1)
v0 = sym_t.sym('v0', 1)
ac0 = sym_t.sym('ac0', 1)
p1 = sym_t.sym('p1', 1)
v1 = sym_t.sym('v1', 1)
ac1 = sym_t.sym('ac1', 1)

# velocity
first_der = cs.jacobian(spline, t)
dp = cs.Function('dp', [t, a1, a2, a3, a4, a5], [first_der], ['t', 'a1', 'a2', 'a3', 'a4', 'a5'], ['first_der'])
print(cs.jacobian(spline, t))
print(dp)

# acceleration
sec_der = cs.jacobian(first_der, t)
ddp = cs.Function('ddp', [t, a2, a3, a4, a5], [sec_der], ['t', 'a2', 'a3', 'a4', 'a5'], ['sec_der'])
print(cs.jacobian(first_der, t))
print(ddp)

print("-------->", dp(1, a1, a2, a3, a4, a5))
'''
'''d_spl = spline.jacobian()
print(d_spl(t=t))
dd_spl = d_spl.jacobian()
print(dd_spl)
#sec_der = cs.Function('sec_der', [t], [dd_spl], ['t'], ['dd_spl'])
#print(sec_der(t=0))
#print("---------", dd_spl(t=0))'''



'''
dimp = 3
p = sym_t.sym('p', dimp)

f = cs.sumsqr(p)  # symbolic expression
print(f)

F = cs.Function('f', # function name
                {  # dict of variables (inputs, outputs)
                    'p': p,
                    'f': f
                }, 
                ['p'],  # list of inputs
                ['f']  # list of outputs
                )

print(F)
print(float(F(p=[1, 1, 1])['f']))

jacF = F.jacobian()
print(jacF(p=p)['jac'])

hesF = jacF.jacobian()
print(hesF)
print(hesF(p=p)['jac'])

# ---------------------------------------------

xgrid = np.linspace(1,6,6)
V = [-1, -1, -2, -3, 0, 2]
print(V)

# interpolation
lut = cs.interpolant('LUT', 'bspline', [xgrid], V)
print (lut(2.5))

interp = ip.InterpolatedUnivariateSpline(xgrid, V)
print(interp(2.5))

# evaluation points
N=100
x = np.linspace(1, 6, N)
print(x[1])
y = []
y1 = []

for i in range(0, N):
    y.append(lut(x[i]))
for i in range(0, N):
    y1.append(interp(x[i]))

plt.figure()
plt.plot(x, y, x, y1, xgrid, V, 'o-')
plt.grid()
plt.title('State trajectory')
plt.legend(['x', 'y', 'z'])
plt.xlabel('Time [s]')
plt.show()'''