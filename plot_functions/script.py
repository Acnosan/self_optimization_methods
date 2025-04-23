import matplotlib.pyplot as plt
import numpy as np

def grad_desc_plot(phi_func,alpha_values,alpha_star):
    phi_values=np.array([float(phi_func(v)) for v in alpha_values])
    plt.figure(figsize=(12,8))
    plt.plot(alpha_values,phi_values, c='purple', label='Phi(Alpha)')
    plt.scatter(alpha_star, phi_func(alpha_star), c='black', marker='x', label='alpha*')
    plt.axvline(x=alpha_star, c='red', linestyle='--', label='alpha*')
    plt.axhline(y=phi_func(alpha_star), c='green', linestyle='--', label='phi(alpha*)')
    
    plt.title('Function Phi Alpha Grad Desc')
    plt.xlabel('Alpha')
    plt.ylabel('Phi(Alpha)')
    plt.legend()
    plt.grid(True)
    plt.show()


def bisection_plot(phi_func,alpha_values,alpha_star,a,b):
    plt.figure(figsize=(12, 8))
    plt.plot(alpha_values,phi_func(alpha_values), c='purple',label='function')
    plt.axvline(x=a, c='red', linestyle='--', label='a')
    plt.axvline(x=b, c='yellow', linestyle='--', label='b')
    plt.axvline(x=alpha_star, c='green', linestyle='--', label='alpha')
    plt.axhline(y=phi_func(alpha_star), c='cyan', linestyle='--', label='phi(alpha*)')
    plt.scatter(alpha_star, phi_func(alpha_star), c='black', marker='x', label='alpha')
    
    plt.title('Function Phi Alpha Bisection')
    plt.xlabel('alpha')
    plt.ylabel('Phi(alpha)')
    plt.legend()
    plt.grid(True)
    plt.show()


def golden_section_plot(phi_func,alpha_values,alpha_star,a,b):
    plt.figure(figsize=(12, 8))
    plt.plot(alpha_values,phi_func(alpha_values), c='purple',label='function')
    plt.axvline(x=a, c='red', linestyle='--', label='a')
    plt.axvline(x=b, c='yellow', linestyle='--', label='b')
    plt.axvline(x=alpha_star, c='green', linestyle='--', label='alpha')
    plt.axhline(y=phi_func(alpha_star), c='cyan', linestyle='--', label='phi(alpha*)')
    plt.scatter(alpha_star, phi_func(alpha_star), c='black', marker='x', label='alpha')
    
    plt.title('Function Phi Alpha Bisection')
    plt.xlabel('alpha')
    plt.ylabel('Phi(alpha)')
    plt.legend()
    plt.grid(True)
    plt.show()


