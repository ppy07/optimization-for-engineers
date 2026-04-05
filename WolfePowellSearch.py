# Optimization for Engineers - Dr.Johannes Hild
# Wolfe-Powell line search

# Purpose: Find t to satisfy f(x+t*d)<=f(x) + t*sigma*gradf(x).T@d
# and gradf(x+t*d).T@d >= rho*gradf(x).T@d

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# sigma: value in (0, 1 / 2), marks quality of decrease. Default value: 1.0e-3
# rho: value in (sigma, 1), marks quality of steepness. Default value: 1.0e-2
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set, such that t satisfies both Wolfe - Powell conditions

# Required files:
# < none >

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.01], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=1

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.2], [1]])
# d = np.array([[0.1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=16

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-0.2], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=0.25

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23393224
    return matrnr


def WolfePowellSearch(f, x: np.array, d: np.array, sigma=1.0e-3, rho=1.0e-2, verbose=0):
    fx = f.objective(x) # store objective
    gradx = f.gradient(x) # store gradient
    descent = gradx.T @ d # store descent value

    if descent >= 0: # if not a descent direction
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 0.5: # if sigma is out of range
        raise TypeError('range of sigma is wrong!')

    if rho <= sigma or rho >= 1: # if rho does not fit to sigma
        raise TypeError('range of rho is wrong!')

    if verbose: # print information
        print('Start WolfePowellSearch...') # print start

    #Wolfe-Powell condtion 1: Armijo condition (bool valued function)
    def WP1(ft, t):
        return ft <= fx + t * sigma * descent
        
    # Wolfe-Powell condition 2: Curvature(bool valued function)
    def WP2(gradft: np.array):
        return gradft.T @ d >= rho * descent
        

    t = 1.0 # initial step size guess

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    
    #Case 1: W1 fails -> backtracking
    if not WP1(f.objective(x+t*d),t):
        t=t/2  # Reduce t
        while not WP1(f.objective(x+t*d),t):
            t=t/2 # Continue reducing until W1 is satisfied
        t_m1=t   # Set t⁻ (lower bound)
        t_pp=2*t # Set t⁺ (upper bound)
    
    #Case 2: W1 and W2 pass -> accept step
    elif WP2(f.gradient(x+t*d)):
        return t 
    
    #case 3: W1 passes, W2 fails -> forward tracking
    else:
        t=2*t   # Double t
        while WP1(f.objective(x+t*d),t): # As long as W1 holds
            t=2*t # Keep doubling t
        t_m1=t/2  # Last good t becomes lower bound
        t_pp=t    # First violating t becomes upper bound

    #Final refinement: bisection until both conditions are satisfied
    t=t_m1   # Start from lower bound

    while not WP2(f.gradient(x+t*d)):    # While W2 not satisfied
        t=(t_m1+t_pp)/2                   # Bisection
        if WP1(f.objective(x+t*d),t):    # Check W1 at midpoint
            t_m1=t                        # Narrow lower bound
        else:
            t_p = t                       # Narrow upper bound

    # INCOMPLETE CODE ENDS

    if verbose: # print information
        xt = x + t * d # store solution point
        fxt = f.objective(xt) # get its objective
        gradxt = f.gradient(xt) # get its gradient
        print('WolfePowellSearch terminated with t=', t) # print terminatin and step size
        print('Wolfe-Powell: ', fxt, '<=', fx+t*sigma*descent, ' and ', gradxt.T @ d, '>=', rho*descent) # print Wolfe-Powell checks

    return t
