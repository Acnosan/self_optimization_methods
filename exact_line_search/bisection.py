import sympy as sp
from sympy.abc import x,y,alpha
import numpy as np

from plot_functions.script import bisection_plot

def bisection(phi_func,a,b,l,epsilon):
    while(b-a >= l):
        lambdaa=(a+b)/2 - epsilon
        mu=(a+b)/2 + epsilon
        print(f'lambda is {lambdaa:.6f}, mu is {mu:.6f}')
        phi_lambda=phi_func.evalf(subs={alpha: lambdaa})
        phi_mu=phi_func.evalf(subs={alpha: mu})
        if phi_lambda>phi_mu:
            a = lambdaa
        else:
            b = mu
    print(f'[a b] = [{a:.6f} {b:.6f}], alpha = {(a+b)/2}')
    phi_func = sp.lambdify(sp.symbols('alpha'), phi_func, modules='numpy')
    alpha_values = np.linspace(a,b,50)
    bisection_plot(phi_func,alpha_values,a,b)

function = sp.Pow(x-1,2)+sp.Pow(y,3)-x*y
dk = sp.Matrix([1,-2])
xy = sp.Matrix([1,1])

xy_alpha = xy+alpha*dk
phi=function.subs({x:xy_alpha[0],y:xy_alpha[1]})
print(f'xy_alpha : {xy_alpha}')
print(f'phi : {phi}')
epsilon = 1e-6
b=0.4
a=0
l=0.001
bisection(phi,a,b,l,epsilon)
