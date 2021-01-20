import casadi as cs
import numpy as np
import scipy.interpolate as ip
from matplotlib import pyplot as plt

# select a sym type
sym_t = cs.SX

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
plt.show()