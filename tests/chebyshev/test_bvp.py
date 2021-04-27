import sys
from os.path import join
import numpy as np
from numpy import cos, sin, exp, sqrt
from scipy.linalg import solve
import matplotlib.pyplot as plt

plt.style.use('dark_background')
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

sys.path.insert(0, join('..', '..'))
from orthopoly.chebyshev import *

#-------------------------------------------------------------------------------
#INPUTS

#low boundary of domain
xa = -6
#high boundary of domain
xb = 3
#range of numbers of points to use
pts = range(6, 33, 2)
#order of left boundary condition derivative (can be zero)
ad = 0
#order of internal derivative (1 or 2)
id = 2
#order of right boundary condition derivative (can be zero)
bd = 1
#the solution function
F = lambda x: sin(x) + exp(x/2) - x/3
#the source function (whichever derivative of F)
S = lambda x: -sin(x) + exp(x/2)/4
#value of low boundary condition
va = F(xa)
#value of high boundary condition
vb = cos(xb) + exp(xb/2)/2 - 1/3
#extra points to use in the expansion
xextra = None

#-------------------------------------------------------------------------------
#FUNCTIONS

def cheby_bvp_test(xa, xb, pts, aderiv, ideriv, bderiv, va, vb, F, S, xextra=None):
    """Test the convergence of the chebyshev spectral solver for a boundary
    value problem
    args:
        xa - float, value of left (lower) domain boundary
        xb - float, value of right (higher) domain boundary
        pts - iterable, different numbers of points to use
        aderiv - int, order of derivative for left boundary condition
        ideriv - int, order of derivative for internal nodes
                 (must be 1 or 2)
        bderiv - int, order of derivative for right boundary condition
        va - value for the left boundary
        vb - value for the right boundary
        F - solution function
        S - source function (whichever derivative of F)
    optional args:
        xextra - extra points within the domain to include as
                 collocation points
    returns:
        err - maximum relative error for each of the numerical solutions
        xs - array of grid points for each solution
        qexs - array of exact solution values for each solution
        qsps - array of numerical solution values for each solution"""

    #number of resolutions to test
    L = len(pts)
    #exact and spectral solution arrays
    qexs, qsps = [], []
    #grids
    xs = []
    #max relative error
    err = np.empty((L,))
    #do the tests
    for i in range(L):
        #set up the grid/solver
        xhat, theta, x, A = cheby_bvp_setup(xa, xb, pts[i], aderiv,
                ideriv, bderiv, xextra=xextra)
        #create the source array
        b = np.empty((len(x),))
        b[0] = va
        b[1:-1] = np.array([S(_x) for _x in x[1:-1]])
        b[-1] = vb
        #solve for the coefficients of the Cheb expansion
        coef = solve(A, b)
        #evaluate the expansion
        qsp = cheby_hat_sum(xhat, coef)
        #evaluate the exact solution
        qex = np.array([F(_x) for _x in x])
        #store the error
        err[i] = np.max(np.abs(qsp - qex)/np.abs(qex.max()))
        #store the solution stuff
        qexs.append(qex)
        qsps.append(qsp)
        xs.append(x)

    return(err, xs, qexs, qsps)

#-------------------------------------------------------------------------------
#MAIN

print('TESTING CHEBYSHEV SPECTRAL CONVERGENCE FOR BOUNDARY VALUE PROBLEM')

#test the boundary value convergence
err, xs, qexs, qsps = cheby_bvp_test(xa, xb, pts, ad, id, bd, va, vb, F, S, xextra)

assert err[-1] < 1e-10

figa, axa = plt.subplots(1,1)
figb, axb = plt.subplots(1,1)

axa.loglog(pts, err, ':.')
axa.set_xlabel('Number of Expansion/Collocation Points')
axa.set_ylabel('Maximum Relative Error')
axa.set_title('Chebyshev Collocation Convergence on Second Order BVP')

x = np.linspace(xa, xb, 1000)
axb.plot(x, F(x))
axb.set_xlabel('x')
axb.set_ylabel('y')
axb.set_title('Exact Solution to BVP')

figa.tight_layout()
figb.tight_layout()

plt.show()
