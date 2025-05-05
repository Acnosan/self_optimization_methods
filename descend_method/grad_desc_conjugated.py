import sympy as sp
from sympy.abc import x,y,alpha
from sympy import hessian
import numpy as np

from plot_functions.script import grad_desc_plot

def grad_desc_conjugated(function,x0y0):
    derivatives = [function.diff(x)[0],function.diff(y)[0]]
    x_y_step = x0y0.copy()
    hess_matrix = np.array(hessian(function,(x,y)))
    gradient = np.array(
        [d.evalf(subs={x:x_y_step[0],y:x_y_step[1]})for d in derivatives]
        ,dtype=float)
    step = 0
    tol = 1e-6
    while (np.linalg.norm(gradient) > tol or step != 3):
        dk = -gradient
        hess = np.array(
            [[h.subs({x:x_y_step[0],y:x_y_step[1]}) for h in hess_row] for hess_row in hess_matrix]
        ,dtype=float)
        numerator = np.dot(gradient.T, dk)
        denominator = np.dot(dk.T, np.dot(hess, dk))
        if abs(denominator) < 1e-12:
            print(f"Step {step}: denominator too small, stopping to avoid NaN.")
            break
        alpha_k = -numerator / denominator
        x_y_step = x_y_step + alpha_k*dk
        gradient_old = gradient.copy()
        gradient = np.array(
            [d.evalf(subs={x:x_y_step[0],y:x_y_step[1]})for d in derivatives]
            ,dtype=float)
        beta_k = np.dot(gradient.T,np.dot(hess,gradient_old))/denominator
        dk = -gradient + beta_k*dk
        step+=1
        print(f'step {step} : x{step}_y{step} = {x_y_step}, alpha = {alpha_k}')
    else:
        print(f'Final step {step} : x{step}_y{step} = {x_y_step}, alpha* = {alpha_k}')

## -----------------------------------------------------------------
## YOUR CHANGES HERE 
print(f'---- QUADRATIC FORM')
function = sp.Pow(x,2)+sp.Pow(y,2) ## x^2 + y^2
hess_matrix = hessian(function,(x,y))
arrx = sp.Matrix([x,y])
b = sp.Matrix([0,0])
s = 1/2*sp.transpose(arrx)*hess_matrix*arrx - sp.transpose(b)*arrx
sp.pprint(f'Quadratic form :{s}')

print(f'---- USING GRADIENT DESC CONJUGATED')
x0y0 = np.array([1,1],dtype=float) ## TEST WITH AN INITIAL POINT
grad_desc_conjugated(s,x0y0)
