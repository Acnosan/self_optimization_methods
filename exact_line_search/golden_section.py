import sympy as sp
from sympy.abc import x,y,alpha
import numpy as np

from plot_functions.script import golden_section_plot

def golden_section(phi_func,a,b,epsilon,golden_alpha):
    step=0
    lambdaa=a+(1-golden_alpha)*(b-a)
    mu=a+(golden_alpha)*(b-a)
    while(b-a >= epsilon):
        phi_lambda=phi_func.evalf(subs={alpha: lambdaa})
        phi_mu=phi_func.evalf(subs={alpha: mu})
        print(f'lambda: {lambdaa:.6f}, mu: {mu:.6f}; phi(lambda): {phi_lambda:.6f}, phi(mu):{phi_mu:.6f}')
        if phi_lambda>phi_mu:
            a = lambdaa
            lambdaa = mu
            mu = a+golden_alpha*(b-a)
        else:
            b = mu
            mu = lambdaa
            lambdaa = a+(1-golden_alpha)*(b-a)
        step+=1
    alpha_star = (a+b)/2
    print(f'step {step} : [a b] = [{a:.6f} {b:.6f}], alpha = {alpha_star}')
    phi_func = sp.lambdify(sp.symbols('alpha'), phi_func, modules='numpy')
    alpha_values = np.linspace(a,b,50)
    golden_section_plot(phi_func,alpha_values,alpha_star,a,b)

## -----------------------------------------------------------------
## YOUR CHANGES HERE 
"""
function = sp.Pow(x-1,2)+sp.Pow(y,3)-x*y
dk = sp.Matrix([1,-2]) ## direction of gradient
xy = sp.Matrix([1,1]) ## TEST WITH AN INITIAL POINT

xy_alpha = xy+alpha*dk
phi_func=function.subs({x:xy_alpha[0],y:xy_alpha[1]})
print(f'xy_alpha : {xy_alpha}')
print(f'phi func: {phi_func}')
"""
## THIS PHI FUNC WAS GIVEN ..
phi_func = 100*(alpha**2) - (100*alpha) + 25
b=0.8 # upper bound
a=0.2 # lower bound
epsilon=0.05 # (threshold)
golden_alpha = 0.61
golden_section(phi_func,a,b,epsilon,golden_alpha)

