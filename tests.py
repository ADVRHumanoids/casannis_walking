
def f(t, table):
    a0 = table[0]
    a1 = table[1]
    a2 = table[2]
    a3 = table[3]
    a4 = table[4]
    a5 = table[5]
    value = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + a5 * t ** 5
    deriv = a1 + 2*a2 * t + 3*a3 * t ** 2 + 4*a4 * t ** 3 + 5*a5 * t ** 4
    return deriv



coef = [ 0.15078891, -0.1793868,   0.09056,    -0.01559885, -0.00025998,  0.00389672]
print(f(2, coef))