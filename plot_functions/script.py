import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

def grad_desc_plot(phi_func,alpha_values,alpha_star):
    plt.figure(figsize=(12,8))
    plt.plot(alpha_values,phi_func(alpha_values), c='purple', label='Phi(Alpha)')
    plt.scatter(alpha_star, phi_func(alpha_star), c='black', marker='x', label='alpha*')
    plt.axvline(x=alpha_star, c='red', linestyle='--', label='alpha*')
    plt.axhline(y=phi_func(alpha_star), c='red', linestyle='--', label='phi(alpha*)')
    plt.title('Function Phi Alpha TP2 EX2')
    plt.xlabel('Alpha')
    plt.ylabel('Phi(Alpha)')
    plt.legend()
    plt.grid(True)
    plt.show()

def bisection_plot(phi_func,alpha_values,a,b):
    alpha_star = (a + b) / 2

    plt.figure(figsize=(12, 8))
    plt.plot(alpha_values,phi_func(alpha_values), c='purple',label='function')
    plt.axvline(x=a, c='red', linestyle='--', label='a')
    plt.axvline(x=b, c='red', linestyle='--', label='b')
    plt.axvline(x=alpha_star, c='green', linestyle='--', label='alpha')
    plt.axhline(y=phi_func(alpha_star), c='green', linestyle='--', label='phi(alpha*)')
    plt.scatter(alpha_star, phi_func(alpha_star), c='black', marker='x', label='alpha')
    
    plt.title('Function Phi Alpha Bisection')
    plt.xlabel('alpha')
    plt.ylabel('Phi(alpha)')
    plt.grid(True)
    plt.legend()
    plt.show()

