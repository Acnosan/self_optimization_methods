import sympy as sp
from sympy.abc import x,y,alpha
import numpy as np
from math import log
from plot_functions.script import bisection_plot

def bisection(phi_func,a,b,l,epsilon):
    step=0
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
        step+=1
    alpha_star = (a+b)/2
    print(f'step {step} : [a b] = [{a:.6f} {b:.6f}], alpha = {alpha_star}')
    phi_func = sp.lambdify(sp.symbols('alpha'), phi_func, modules='numpy')
    alpha_values = np.linspace(a,b,50)
    bisection_plot(phi_func,alpha_values,alpha_star,a,b)

## -----------------------------------------------------------------
## YOUR CHANGES HERE 
function = sp.Pow(x-1,2)+sp.Pow(y,3)-x*y
dk = sp.Matrix([1,-2]) ## direction of gradient
xy = sp.Matrix([1,1]) ## TEST WITH AN INITIAL POINT

xy_alpha = xy+alpha*dk
phi=function.subs({x:xy_alpha[0],y:xy_alpha[1]})
print(f'xy_alpha : {xy_alpha}')
print(f'phi func: {phi}')
epsilon = 1e-6
b=0.4 # upper bound
a=0 # lower bound
l=0.001 # l (threshold)
bisection(phi,a,b,l,epsilon)

N = log((b-a)/l)/log(2)
print(f'The Minimal Number of Iterations is around {N}')
