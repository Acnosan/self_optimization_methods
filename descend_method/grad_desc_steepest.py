import sympy as sp
from sympy.abc import x,y,alpha
from sympy import hessian
import numpy as np

from plot_functions.script import grad_desc_plot

def grad_desc(function,x0y0,epsilon):
    derivatives = [function.diff(x)[0],function.diff(y)[0]]
    x_y_step = x0y0.copy()
    step = 0
    while(step < 500):
        gradient = np.array([d.evalf(subs={x:x_y_step[0],y:x_y_step[1]}) for d in derivatives],dtype=np.float32)
        if np.linalg.norm(gradient) < epsilon:
            break
        dk = -gradient
        opt_history = grad_desc_solve_for_alpha(function,x_y_step,dk)
        alpha_star = opt_history['alpha*']
        phi_alpha = opt_history['phi_alpha']
        x_y_step = x_y_step + alpha_star*dk
        if step%20 == 0:
            print(f'step {step} : x{step}_y{step} = {x_y_step}')
        step+=1
    else:
        print(f'step 500 , the function did not converge')
    print(f'Final step {step} : x{step}_y{step} = {x_y_step}, alpha* = {alpha_star}')
    alpha_values = np.linspace(float(alpha_star)-5, float(alpha_star)+5, 50)
    phi_alpha = sp.lambdify(alpha,phi_alpha,modules='numpy')
    grad_desc_plot(phi_alpha,alpha_values,alpha_star)

def grad_desc_solve_for_alpha(function,x0y0,dk):
    print(f'---- USING THE PHI FUNCTION WITH RESPECT TO ALPHA')
    x0y0_alpha = x0y0+alpha*dk
    phi_alpha = function.subs({x: x0y0_alpha[0], y: x0y0_alpha[1]})
    phi_prime = phi_alpha.diff(alpha)
    sp.pprint(f"phi (alpha): {phi_alpha}")
    sp.pprint(f"phi prime (alpha): {phi_prime}")
    
    print(f'---- FINDING OPTIMAL ALPHA AND VALUES X Y')
    
    opt_history = []
    optimal_alphas = sp.solve(phi_prime, alpha)
    if optimal_alphas:
        for alpha_star in optimal_alphas.values():
            alpha_star = float(alpha_star)
            xstar_ystar = x0y0 + alpha_star*dk
            #optimal_expr = function.subs({x: xstar_ystar[0], y: xstar_ystar[1]})
            evaluated = function.evalf(subs={x: xstar_ystar[0], y: xstar_ystar[1]})
            
            opt_history.append({
                'alpha*' : alpha_star,
                'phi_alpha': phi_alpha,
                'x*_y*' : xstar_ystar,
                #'function_with_substitution' : optimal_expr,
                'evaluated' : evaluated
            })
            
    opt_history = min(opt_history,key=lambda x: x['evaluated'])
    return opt_history

## -----------------------------------------------------------------
## YOUR CHANGES HERE 
print(f'---- QUADRATIC FORM')
function = sp.Pow(x,2)+sp.Pow(y,2) ## x^2 + y^2
hess_matrix = hessian(function,(x,y))
arrx = sp.Matrix([x,y])
b = sp.Matrix([0,0])
s = 1/2*sp.transpose(arrx)*hess_matrix*arrx - sp.transpose(b)*arrx
sp.pprint(f'Quadratic form :{s}')

print(f'---- USING GRADIENT DESC (Steepest)')
epsilon = 1e-5
x0y0 = np.array([1,1],dtype=float) ## TEST WITH AN INITIAL POINT
grad_desc(s,x0y0,epsilon)