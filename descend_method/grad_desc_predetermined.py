import sympy as sp
from sympy.abc import x,y,alpha
from sympy import hessian
import numpy as np

from plot_functions.script import grad_desc_plot

def grad_desc(function,alphas,x0y0,epsilon):
    derivatives = [function.diff(x)[0],function.diff(y)[0]]
    for a in alphas:
        print(f'ALPHA = {a}')
        x0y0 = x0y0
        x_y_step = x0y0.copy()
        step = 0
        while(step < 500):
            gradient = np.array([d.subs({x:x_y_step[0],y:x_y_step[1]}) for d in derivatives],dtype=np.float32)
            if np.linalg.norm(gradient) < epsilon:
                break
            direction = -a * (gradient/np.linalg.norm(gradient))
            x_y_step = x_y_step + direction
            if step%20 == 0:
                print(f'step {step} : x{step}_y{step} = {x_y_step}')
            step+=1
        else:
            print(f'step 500 : for alpha {a}, the function did not converge')
        print(f'Final step {step} : x{step}_y{step} = {x_y_step}')
    opt_history = grad_desc_solve_for_alpha(s,x0y0)

    print(f"Best Alpha = {opt_history['alpha*']}")
    print(f"x*_y* = {opt_history['x*_y*']}")
    print(f"Evaluated = {opt_history['evaluated'][0]}")

def grad_desc_solve_for_alpha(function,x0y0):
    print(f'---- USING THE PHI FUNCTION WITH RESPECT TO ALPHA')

    derivatives = [function.diff(x)[0],function.diff(y)[0]]
    grad_vals = [g.evalf(subs={x: x0y0[0], y: x0y0[1]}) for g in derivatives]
    dk = -np.array(grad_vals, dtype=np.float64)
    x0y0_alpha = x0y0+alpha*dk
    phi_alpha = s.subs({x: x0y0_alpha[0], y: x0y0_alpha[1]})
    phi_prime = sp.diff(phi_alpha, alpha)
    sp.pprint(f"phi (alpha): {phi_alpha}")
    sp.pprint(f"phi prime (alpha): {phi_prime}")
    
    print(f'---- FINDING OPTIMAL ALPHA AND VALUES X Y')
    
    opt_history = []
    optimal_alphas = sp.solve(phi_prime, alpha)
    if optimal_alphas:
        for alphaStar in optimal_alphas.values():
            alpha_value = float(alphaStar)
            xstar_ystar = x0y0 + alpha_value*dk
            evaluated = s.evalf(subs={x: xstar_ystar[0], y: xstar_ystar[1]})
            
            opt_history.append({
                'alpha*' : alpha_value,
                'x*_y*' : xstar_ystar,
                'evaluated' : evaluated
            })
            
    opt_history = min(opt_history,key=lambda x: x['evaluated'])
    alpha_star = opt_history['alpha*']
    alpha_values = np.linspace(float(alpha_star)-5, float(alpha_star)+5, 50)
    phi_alpha = sp.lambdify(alpha,phi_alpha,modules='numpy')
    grad_desc_plot(phi_alpha,alpha_values,alpha_star)

## -----------------------------------------------------------------
## YOUR CHANGES HERE 
print(f'---- QUADRATIC FORM')
function = sp.Pow(x,2)+sp.Pow(y,2) ## x^2 + y^2
hess_matrix = hessian(function,(x,y))
arrx = sp.Matrix([x,y])
b = sp.Matrix([0,0])
s = 1/2*sp.transpose(arrx)*hess_matrix*arrx - sp.transpose(b)*arrx
sp.pprint(f'Quadratic form :{s}')

print(f'---- USING GRADIENT DESC WITH PREDETERMINED ALPHA')
alphas = [0.05,0.3] ## TEST ALPHA VALUES
epsilon = 1e-5
x0y0 = np.array([1,1],dtype=float) ## TEST WITH AN INITIAL POINT
grad_desc(s,alphas,x0y0,epsilon)
