import sys
from os.path import join
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, join('..', '..'))
from orthopoly.legendre import *

plt.style.use('dark_background')
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

#-------------------------------------------------------------------------------
#FUNCTIONS

#5 pt finite difference first deriv of Legendre type function
_fd1 = lambda y, h: (y[0]/12 - 2*y[1]/3 + 2*y[3]/3 - y[4]/12)/h
fd1 = lambda f, x, h, n, m: _fd1((f(x - 2*h,n,m),
                                  f(x -   h,n,m),
                                  None,
                                  f(x +   h,n,m),
                                  f(x + 2*h,n,m)), h)
#5 pt finite difference second deriv of Legendre type function
_fd2 = lambda y, h: (-y[0]/12 + 4*y[1]/3 - 5*y[2]/2 + 4*y[3]/3 - y[4]/12)/(h*h)
fd2 = lambda f, x, h, n, m: _fd2((f(x - 2*h,n,m),
                                  f(x - h,n,m),
                                  f(x,n,m),
                                  f(x + h,n,m),
                                  f(x + 2*h,n,m)), h)

def test_dlegen_theta(nmax, N=100, h=1e-3):
    """compare the results of dlegen_theta and a numerical evaluation of the
    first derivative of legen_theta
    args:
        nmax - maximum degree
    optional args:
        N - number of points to evaluate
        h - spacing of finite diff appx
    returns:
        err - 2D matrix with maximum abs error for each function
        fig, ax - plot of err"""

    theta = np.linspace(2*h, np.pi-2*h, N)
    err = np.nan*np.zeros((nmax+1,nmax+1))
    for n in range(nmax+1):
        for m in range(n+1):
            #numerical derivative of P_n^m
            dP_fd = [fd1(legen_theta, i, h/((n-m+1)**2), n, m) for i in theta]
            #recursive derivative of P_n^m
            dP = dlegen_theta(theta, n, m)
            #track maximum error
            err[n,m] = np.abs(dP - dP_fd).max()
            #print('(%d,%d)   %g' % (n,m,err[n,m]))
    fig, ax = plt.subplots(1,1)
    r = ax.pcolormesh(range(nmax+2), range(nmax+2), np.log10(err))
    cb = plt.colorbar(r, ax=ax)
    cb.set_label('log$_{10}$(Maximum Absolute Error)')
    ax.set_ylabel("$n$")
    ax.set_xlabel("$m$")
    ax.set_title(r'Direct Evaluation vs Finite Difference, $dP_n^m/d\theta$')
    ax.invert_yaxis()

    return(err, fig, ax)

def test_ddlegen_theta(nmax, N=100, h=1e-2):
    """compare the results of ddlegen_theta and a numerical evaluation of the
    second derivative of legen_theta
    args:
        nmax - maximum degree
    optional args:
        N - number of points to evaluate
        h - spacing of finite diff appx
    returns:
        err - 2D matrix with maximum abs error for each function
        fig, ax - plot of err"""

    theta = np.linspace(2*h, np.pi-2*h, N)
    err = np.nan*np.zeros((nmax+1,nmax+1))
    for n in range(nmax+1):
        for m in range(n+1):
            #numerical derivative of P_n^m
            ddP_fd = [fd2(legen_theta, i, h/((n-m+1)**2), n, m) for i in theta]
            #recursive derivative of P_n^m
            ddP = ddlegen_theta(theta, n, m)
            #track maximum error
            err[n,m] = np.abs(ddP - ddP_fd).max()
            #print('(%d,%d)   %g' % (n,m,err[n,m]))
    fig, ax = plt.subplots(1,1)
    r = ax.pcolormesh(range(nmax+2), range(nmax+2), np.log10(err))
    cb = plt.colorbar(r, ax=ax)
    cb.set_label('log$_{10}$(Maximum Absolute Error)')
    ax.set_ylabel("$n$")
    ax.set_xlabel("$m$")
    ax.set_title(r'Direct Evaluation vs Finite Difference, $d^2P_n^m/d\theta^2$')
    ax.invert_yaxis()

    return(err, fig, ax)

#-------------------------------------------------------------------------------
#MAIN

print('TESTING ASSOCIATED LEGENDRE POLYNOMIALS')

print('plotting first several associated Legendre polynomials, P_n^m(x)')
fig, ax = plt.subplots(1,1)
x = np.linspace(-1, 1, 1000)
for n in range(3):
    for m in range(n+1):
        ax.plot(x, legen_hat(x, n, m), label="$P_{%d}^{%d}$" % (n,m))
ax.set_xlabel('x')
ax.set_ylabel('P')
ax.set_title('Associated Legendre Polynomials, $P_n^m(x)$')
ax.legend()
plt.show()

print('testing analytical evaluation of first derivatives of associated Legendres')
err, fig, ax = test_dlegen_theta(20)
assert np.nanmax(err) < 1e-5
plt.show()

print('testing analytical evaluation of second derivative of associated Legendres')
err, fig, ax = test_ddlegen_theta(20)
assert np.nanmax(err) < 1e-2
plt.show()
