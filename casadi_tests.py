import casadi as cs
import numpy as np

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
