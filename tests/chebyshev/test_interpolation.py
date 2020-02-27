import sys
from os.path import join
import numpy as np
from numpy import cos, sin, exp, sqrt
from scipy.linalg import solve
from scipy.integrate import quad
import matplotlib.pyplot as plt

plt.style.use('dark_background')
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

sys.path.append(join('..', '..'))
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

def cheby_coef_test(xa, xb, pts, F, da=None, db=None, xextra=None):
    """Test the convergence of chebyshev interpolation
    args:
        xa - float, value of left (lower) domain boundary
        xb - float, value of right (higher) domain boundary
        pts - iterable, different numbers of points to use
        F - function to interpolate
    optional args:
        da - value of left deriv or None to use the value, not the deriv
        db - value of right deriv or None to use the value, not the deriv
        xextra - extra points within the domain to include as collocation
                 points
    returns:
        err - L2 error for each of the numerical solutions
        xs - array of grid points for each solution
        coefs - expansion coefficients for each solution"""

    #number of resolutions to test
    L = len(pts)
    #grids
    xs = []
    #coefficents
    coefs = []
    #L2 error
    err = np.empty((L,))

    for i in range(L):

        if da is not None and db is None:
            xhat, theta, x, A, scale = cheby_coef_setup(xa, xb, pts[i], da=True, xextra=xextra)
            f = np.zeros((len(x),))
            f[0], f[-1] = da, F(xb)

        elif da is None and db is not None:
            xhat, theta, x, A, scale = cheby_coef_setup(xa, xb, pts[i], db=True, xextra=xextra)
            f = np.zeros((len(x),))
            f[0], f[-1] = F(xa), db

        elif da is not None and db is not None:
            xhat, theta, x, A, scale = cheby_coef_setup(xa, xb, pts[i], da=True, db=True, xextra=xextra)
            f = np.zeros((len(x),))
            f[0], f[-1] = da, db

        else:
            xhat, theta, x, A, scale = cheby_coef_setup(xa, xb, pts[i], xextra=xextra)
            f = np.zeros((len(x),))

        f[0], f[-1] = F(xa), F(xb)
        f[1:-1] = np.array([F(_x) for _x in x[1:-1]])
        coef = solve(A, f)
        err[i] = quad(lambda x: (F(x) - cheby_hat_sum(x2xhat(x, xa, xb), coef))**2, xa, xb)[0]
        xs.append(x)
        coefs.append(coef)

    return(err, xs, coefs)

#-------------------------------------------------------------------------------
#MAIN

print('TESTING CHEBYSHEV SPECTRAL CONVERGNECE FOR INTERPOLATION PROBLEM')

#now test simple interpolation
err, xs, coefs = cheby_coef_test(xa, xb, pts, F)

figa, axa = plt.subplots(1,1)
figb, axb = plt.subplots(1,1)

axa.loglog(pts, err, ':.')
axa.set_xlabel('Number of Expansion/Collocation Points')
axa.set_ylabel(r'$L_2$ Error')
axa.set_title('Chebyshev Interpolation Convergence')

x = np.linspace(xa, xb, 1000)
axb.plot(x, F(x))
axb.set_xlabel('x')
axb.set_ylabel('y')
axb.set_title('Interpolated Function')

figa.tight_layout()
figb.tight_layout()

plt.show()
