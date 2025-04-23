import sympy as sp
from sympy.abc import x,y,alpha
from sympy import hessian
import numpy as np

from plot_functions.script import newton_plot

def newton(function,hess_matrix,x0_y0,epsilon):
    for i in range(1,hess_matrix.shape[-1]+1):
        det = sp.det(hess_matrix[:i,:i])
        if det<=0:
            print(f'the hessien is not definite positive')
    step = 0
    x_y_k = x0_y0.copy()
    derivatives = [function.diff(x)[0],function.diff(y)[0]]
    while(step < 500):
        gradient = np.array([d.subs({x:x_y_k[0],y:x_y_k[1]}) for d in derivatives],dtype=np.float32)
        if np.linalg.norm(gradient) < epsilon:
            print(f"Stopping at step {step}, point: {x_y_k}, alpha = {alpha_star}")
            break
        alpha_star = 1.0  
        dk = -gradient
        x_y_alpha = x_y_k + alpha*dk
        phi_func = function.subs({x:x_y_alpha[0],y:x_y_alpha[1]})
        phi_alpha_prime = sp.diff(phi_func,alpha) 
        phi_prime_prime = sp.diff(phi_alpha_prime,alpha)
        
        for i in range(10):
            num = phi_alpha_prime.evalf(subs={alpha:alpha_star})[0]
            denom = phi_prime_prime.evalf(subs={alpha:alpha_star})[0]
            if abs(denom) < 1e-8:
                break
            alpha_star = alpha_star - ( num / denom)
        x_y_k = x_y_k + alpha_star*dk
        step+=1
        print(f"Step {step} -> point: {x_y_k}, alpha: {alpha_star:.4f}")
        
    alpha_values = np.linspace(float(alpha_star)-5, float(alpha_star)+5, 50)
    phi_func = sp.lambdify(sp.symbols('alpha'), phi_func, modules='numpy')
    
    newton_plot(phi_func,alpha_values,alpha_star)

print(f'---- QUADRATIC FORM')
function = sp.Pow(x,2)+sp.Pow(y,2) ## x^2 + y^2
hess_matrix = hessian(function,(x,y))
arrx = sp.Matrix([x,y])
b = sp.Matrix([0,0])
s = 1/2*sp.transpose(arrx)*hess_matrix*arrx - sp.transpose(b)*arrx
sp.pprint(f'Quadratic form :{s}')

x0_y0 = np.array([1,1],dtype=float)
epsilon = 1e-5

newton(s,hess_matrix,x0_y0,epsilon)


