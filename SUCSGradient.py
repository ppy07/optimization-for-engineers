# Optimization for Engineers - Dr.Johannes Hild
# scaled unit central simplex gradient

# Purpose: Approximates gradient of f on a scaled unit central simplex

# Input Definition:
# f: objective class with methods .objective()
# x: column vector in R ** n(domain point)
# h: simplex edge length
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# grad_f_h: simplex gradient
# stenFail: 0 by default, but 1 if stencil failure shows up

# Required files:
# < none >

# Test cases:
# myObjective = multidimensionalObjective()
# x = np.array([[1.02614],[0],[0],[0],[0],[0],[0],[0]], dtype=float)
# h = 1.0e-6
# myGradient = SUCSGradient(myObjective, x, h)
# should return
# myGradient close to [[0],[0],[0],[0],[0],[0],[0],[0]]


import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23393224
    return matrnr


def SUCSGradient(f, x: np.array, h: float, verbose=0):

    if verbose: # print information
        print('Start SUCSGradient...') # print start

    grad_f_h = x.copy() # initialize simplex gradient of f
    # INCOMPLETE CODE STARTS
    n = x.shape[0]       # dimension of the input
    grad_f_h = np.zeros((n, 1))  # Initialize gradient vector
    fx = f.objective(x)  # evaluate function at x

    for j in range(n):
        e_j = np.zeros((n, 1))       # Create unit vector e_j
        e_j[j, 0] = 1.0
        x_plus = x + h * e_j         # Forward vertex
        x_minus = x - h * e_j        # Reflected vertex
        f_plus = f.objective(x_plus)
        f_minus = f.objective(x_minus)
        grad_f_h[j, 0] = (f_plus - f_minus) / (2 * h)  # Central difference


    # INCOMPLETE CODE ENDS

    if verbose: # print information
        print('SUCSGradient terminated with gradient =', grad_f_h) # print termination

    return grad_f_h


def SUCSStencilFailure(f, x: np.array, h: float, verbose=0):

    if verbose: # print information
        print('Check for SUCSStencilFailure...') # print start of check

    f_x = f.objective(x)
    n = x.shape[0]
    stenFail = 1 # initialize stencil failure with true

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    for j in range(n):
        e_j = np.zeros((n, 1))
        e_j[j, 0] = 1.0
        x_plus = x + h * e_j
        x_minus = x - h * e_j
        if f.objective(x_plus) < f_x or f.objective(x_minus) < f_x:
            stenFail = 0  # If any direction is better, no failure
            break
    # INCOMPLETE CODE ENDS
    
    if verbose: # print information
        print('SUCSStencilFailure check returns ', stenFail) # print termination

    return stenFail
