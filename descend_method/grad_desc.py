import sympy as sp
from sympy.abc import x,y,alpha
from sympy import hessian
import numpy as np

from plot_functions.script import grad_desc_plot

print(f'1---- QUADRATIC FORM')
function = sp.Pow(x,2)+sp.Pow(y,2) ## x^2 + y^2
hess_matrix = sp.Matrix(hessian(function,(x,y)))
arrx = sp.Matrix([x,y])
b = sp.Matrix([0,0])
s = 1/2*sp.transpose(arrx)*hess_matrix*arrx - sp.transpose(b)*arrx
sp.pprint(f'Quadratic form :{s}')


print(f'3-1---- USING GRADIENT DESC')
alphas = [0.05,3]
epsilon = 1e-5
x0y0 = np.array([1,1],dtype=float)
x_y_step = x0y0.copy()
derivatives = [s.diff(x)[0],s.diff(y)[0]]
step = 0
for a in alphas:
    print(f'ALPHA = {a}')
    while(np.linalg.norm(x0y0 - x_y_step) < epsilon):
        gradient = np.array([d.subs({x:x_y_step[0],y:x_y_step[1]}) for d in derivatives],dtype=np.float32)
        direction = -a * gradient
        x_y_step = x_y_step + direction
        print(f'step {step} : x0y0new = {x_y_step}')
        step+=1
        x0y0 = x_y_step.copy()


print(f'4---- USING THE PHI FUNCTION WITH RESPECT TO ALPHA')
x0y0 = np.array([1,1],dtype=float)
grad_vals = [g.evalf(subs={x: x0y0[0], y: x0y0[1]}) for g in derivatives]
dk = -np.array(grad_vals, dtype=np.float64)
x0y0_phi = x0y0+alpha*dk
phi_alpha = s.subs({x: x0y0_phi[0], y: x0y0_phi[1]})
phi_prime = sp.diff(phi_alpha, alpha)
sp.pprint(f"phi (alpha): {phi_alpha}")
sp.pprint(f"phi prime (alpha): {phi_prime}")

## 5 
print(f'5---- FINDING OPTIMAL ALPHA AND VALUES X Y')
opt_history = []
optimal_alphas = sp.solve(phi_prime, alpha)
if optimal_alphas:
    for alphaStar in optimal_alphas.values():
        alpha_value = float(alphaStar)
        xstar_ystar = x0y0 + alpha_value*dk
        optimal_expr = s.subs({x: xstar_ystar[0], y: xstar_ystar[1]})
        evaluated = s.evalf(subs={x: xstar_ystar[0], y: xstar_ystar[1]})
        opt_history.append({
            'alphaStar' : alpha_value,
            'x*_y*' : xstar_ystar,
            'optimal_f' : optimal_expr,
            'evaluated' : evaluated
        })

best_optim = min(opt_history,key=lambda x: x['evaluated'])
print(f"best alpha = {best_optim['alphaStar']}")
print(f"x*_y* = {best_optim['x*_y*']}")
print(f"Opt Function = {best_optim['optimal_f']}")
print(f"Evaluated = {best_optim['evaluated']}")


alpha_values = np.linspace(-2, 2, 50)
phi_alpha = sp.lambdify(alpha,phi_alpha,modules='numpy')
alpha_star = best_optim['alphaStar']

grad_desc_plot(phi_alpha,alpha_values,alpha_star)