import sympy as sp
from sympy import hessian
from sympy.abc import x,y,alpha
import numpy as np

from plot_functions.script import armijo_plot

def armijo(function,x0_y0,alpha_star,beta,pii):
    derivatives = [function.diff(x)[0],function.diff(y)[0]]
    gradient = np.array([d.evalf(subs={x:x0_y0[0],y:x0_y0[1]})for d in derivatives],dtype=float)
    direction = -alpha*gradient
    x0_y0_alpha = x0_y0+direction

    phi_func = s.subs({x:x0_y0_alpha[0],y:x0_y0_alpha[1]})
    phi_func_prime = phi_func.diff(alpha)
    
    phi_evaluated=float(phi_func.evalf(subs={alpha:alpha_star})[0])
    phi_init_evaluated=float(phi_func.evalf(subs={alpha:0})[0])
    phi_prime_evaluated=float(phi_func_prime.evalf(subs={alpha:0})[0])
    step=0
    print(f"Step {step}, phi func: {phi_evaluated}, phi(0): {phi_init_evaluated}, phi'(0): {phi_prime_evaluated}")
    while(phi_evaluated>phi_init_evaluated+(alpha_star*beta*phi_prime_evaluated)):
        alpha_star = pii*alpha_star
        phi_evaluated=float(phi_func.evalf(subs={alpha:alpha_star})[0])
        step+=1
        print(f"Step {step}, phi func: {phi_evaluated}, phi(0): {phi_init_evaluated}, phi'(0): {phi_prime_evaluated}")
    else:
        print(f'Step {step}, alpha*:{alpha_star} != alpha* real BUT close enough')
    alpha_values = np.linspace(float(alpha_star)-float(alpha_star/0.8), float(alpha_star)+float(alpha_star/0.8), 50)
    phi_func = sp.lambdify(sp.symbols('alpha'), phi_func, modules='numpy')
    
    lower_bound = [phi_init_evaluated+(alpha*phi_prime_evaluated) for alpha in alpha_values]
    upper_bound = [phi_init_evaluated+(alpha*beta*phi_prime_evaluated) for alpha in alpha_values]
    armijo_plot(phi_func,alpha_values,alpha_star,lower_bound,upper_bound)

function = sp.Pow(x,2)+sp.Pow(y,2) ## x^2 + y^2
hess_matrix = hessian(function,(x,y))
arrx = sp.Matrix([x,y])
b = sp.Matrix([0,0])
s = 1/2*sp.transpose(arrx)*hess_matrix*arrx - sp.transpose(b)*arrx
sp.pprint(f'Quadratic form :{s}')

x0_y0 = np.array([1,1],dtype=float)
beta=1e-4
pii=0.25
alpha_star=0.4
armijo(s,x0_y0,alpha_star,beta,pii)