import sys
from os.path import join
import numpy as np
from numpy import pi, sin, cos
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

sys.path.insert(0, join('..', '..'))
from orthopoly.spherical_harmonic import *

#plt.style.use('dark_background')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#-------------------------------------------------------------------------------
#FUNCTIONS

#5 pt finite difference first deriv of Legendre type function
_fd1 = lambda y, h: (y[0]/12 - 2*y[1]/3 + 2*y[3]/3 - y[4]/12)/h
fd1 = lambda f, x, h, n, m: _fd1((f(x - 2*h,n,m),
                                  f(x -   h,n,m),
                                  None, f(x +   h,n,m),
                                  f(x + 2*h,n,m)), h)
#5 pt finite difference second deriv of Legendre type function
_fd2 = lambda y, h: (-y[0]/12 + 4*y[1]/3 - 5*y[2]/2 + 4*y[3]/3 - y[4]/12)/(h*h)
fd2 = lambda f, x, h, n, m: _fd2((f(x - 2*h,n,m),
                                  f(x - h,n,m),
                                  f(x,n,m),
                                  f(x + h,n,m),
                                  f(x + 2*h,n,m)), h)

f_inner_prod = lambda t, p, n1, m1, n2, m2: sph_har(t, p, n1, m1)*sph_har(t, p, n2, m2)*sin(t)

def sph_har_inner_prod(n1, m1, n2, m2):
    """Compute the inner product of two spherical harmonics
    args:
        n1 - degree of first harmonic
        m1 - order of first harmonic
        n2 - degree of second harmonic
        m2 - order of second harmonic
    returns:
        q - inner product of the two functions over the whole sphere"""

    return( dblquad(f_inner_prod, 0, 2*pi, lambda x: 0, lambda x: pi, (n1, m1, n2, m2))[0] )

def test_sph_har_orthonormal(maxn):
    """Integrate the product of various pairs of spherical harmonics over the
    sphere to test whether the set is orthonormal
    args:
        maxn - maximum degree of harmonics
    returns:
        err - maximum absolute error from expected integration value"""
    abserr = 0.0
    #loop through all harmoics with n less than maxn
    for n1 in range(maxn+1):
        for m1 in range(-n1,n1+1):
            #test each of them against all others with n less than maxn
            for n2 in range(n1,maxn+1):
                for m2 in range(-n2,n2+1):
                    #integral over the sphere of Y1*Y2
                    q = sph_har_inner_prod(n1, m1, n2, m2)
                    #error
                    if n1 == n2 and m1 == m2:
                        e = abs(q - 1.0)
                    else:
                        e = abs(q)
                    #track max error
                    if e > abserr:
                        abserr = e
    return(abserr)

def plot_sph_har_pyramid(nmax, nx=301, nz=201, cmap='RdBu'):
    """Plot a pyramid of the real spherical harmonics up to degree n
    args:
        nmax - maximum degree of harmonics
    optional args:
        nx - number of longitudinal points to use
        nz - number of latitudinal points to use
        cmap - colormap name
    returns:
        fig - a figure with plotted harmonics"""

    ni = nmax+1
    nj = 2*nmax + 1
    fig, axs = plt.subplots(ni, nj, figsize=(10,4))
    plt.subplots_adjust(left=0.05, right=0.875, bottom=0.01, top=0.95, wspace=0.04, hspace=0.04)

    for i in range(ni):
        for j in range(nj):
            axs[i][j].set_visible(False)

    phi = np.linspace(0, 2*pi, nz)
    theta = np.linspace(0, pi, nx)
    Phi, Theta = np.meshgrid(phi, theta)
    phi, theta = np.linspace(0, 2*pi, nz+1), np.linspace(0, pi, nx+1)

    Z = {}
    vmin = 0
    vmax = 0
    for n in range(nmax+1):
        for m in range(-n,n+1):
            z = sph_har(Theta, Phi, n, m)
            Z[(n,m)] = sph_har(Theta, Phi, n, m)
            if z.max() > vmax: vmax = z.max()
            if z.min() < vmin: vmin = z.min()

    for n in range(nmax+1):
        for m in range(-n,n+1):
            ax = plt.subplot(ni, nj, n*nj + nmax+m+1, projection='aitoff')
            r = ax.pcolormesh(phi-pi, theta-pi/2, Z[(n,m)], vmin=vmin, vmax=vmax, cmap=cmap)
            if m == -n:
                ax.set_ylabel("$n = %d$" % n, fontsize=10)
            if abs(m) == n:
                ax.set_title("m = %d" % m, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    cax = fig.add_axes([0.9, 0.05, 0.015, 0.9])
    cb = fig.colorbar(r, cax=cax)
    #cb.ax.tick_params(labelsize=14)
    fig.text(0.05, 0.95, 'Real\nSpherical\nHarmonics\n$Y_n^m(\\theta,\\phi)$',
            va='top', ha='left', fontsize=15)

    return(fig)

def test_grad_sph_har(ntest=100, nmax=100, h=1e-4):
    """Compute the gradient of spherical harmonics at a set of points
    randomly distributed in n, m, theta, and phi for comparison with
    a numerical approximation of the gradient and create some illustrative
    plots
    optional args:
        ntest - number of test points
        nmax - highest allowable degree
        h - spacing of finite diff appx"""

    res = []
    yn, ym = Tnm(nmax)
    yn, ym = yn[1:], ym[1:]
    idx = np.random.randint(1, len(yn), ntest)
    for i in idx:
        n, m = yn[i], ym[i]
        t = pi*np.random.rand()
        p = 2*pi*np.random.rand()
        dt, dp = grad_sph_har(t, p, n, m)
        tg = np.linspace(t - 2*h, t + 2*h, 5)
        pg = np.linspace(p - 2*h, p + 2*h, 5)
        try:
            yt = sph_har(tg, p, n, m)
            yp = sph_har(t, pg, n, m)
        except ValueError:
            pass
        else:
            fddt = _fd1(yt, h)
            fddp = _fd1(yp, h)/sin(t)
            terr = abs((fddt - dt))#/dt)
            perr = abs((fddp - dp))#/dp)
            res.append((n, m, t, p, terr, perr))

    n, m, t, p, terr, perr = zip(*res)
    n, m, t, p, terr, perr = (np.array(n), np.array(m), np.array(t),
                             np.array(p), np.array(terr), np.array(perr))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,5))
    ax1.semilogy(t[terr > 1e-20], terr[terr > 1e-20], '.')
    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel(r'Abs Error, $\partial Y/\partial\theta$')
    ax1.set_title(r'Error in Colatitude Component, $\partial Y/\partial\theta$')
    ax2.semilogy(p[perr > 1e-20], perr[perr > 1e-20], '.')
    ax2.set_xlabel(r'$\phi$')
    ax2.set_ylabel(r'Abs Error, $\partial Y/\partial\phi$')
    ax2.set_title(r'Error in Longitude Component, $\partial Y/\partial\phi$')
    fig.tight_layout()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    r = ax1.scatter(p[terr > 1e-20], t[terr > 1e-20], c=np.log10(terr[terr > 1e-20]))
    cb = plt.colorbar(r, ax=ax1)
    cb.set_label(r'log$_{10}$(Abs Error), $\partial Y_n^m(\theta,\phi)/\partial\theta$')
    ax1.set_xlabel(r'$\theta$')
    ax1.set_ylabel(r'$\phi$')
    ax1.set_title(r'Error in Colatitude Component, $\partial Y_n^m(\theta,\phi)/\partial\theta$')
    r = ax2.scatter(p[perr > 1e-20], t[perr > 1e-20], c=np.log10(perr[perr > 1e-20]))
    cb = plt.colorbar(r, ax=ax2)
    cb.set_label(r'log$_{10}$(Abs Error), $\partial Y_n^m(\theta,\phi)/\partial\phi$')
    ax2.set_xlabel(r'$\theta$')
    ax2.set_ylabel(r'$\phi$')
    ax2.set_title(r'Error in Longitude Component, $\partial Y_n^m(\theta,\phi)/\partial\phi$')
    fig.tight_layout()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    r = ax1.scatter(m[terr > 1e-20], n[terr > 1e-20], c=np.log10(terr[terr > 1e-20]))
    cb = plt.colorbar(r, ax=ax1)
    cb.set_label(r'log$_{10}$(Abs Error), $\partial Y_n^m(\theta,\phi)/\partial\theta$')
    ax1.set_xlabel('$m$')
    ax1.set_ylabel('$n$')
    ax1.set_title(r'Error in Colatitude Component, $\partial Y_n^m(\theta,\phi)/\partial\theta$')
    ax1.invert_yaxis()
    r = ax2.scatter(m[perr > 1e-20], n[perr > 1e-20], c=np.log10(perr[perr > 1e-20]))
    cb = plt.colorbar(r, ax=ax2)
    cb.set_label(r'log$_{10}$(Abs Error), $\partial Y_n^m(\theta,\phi)/\partial\phi$')
    ax2.set_xlabel('$m$')
    ax2.set_ylabel('$n$')
    ax2.set_title(r'Error in Longitude Component, $\partial Y_n^m(\theta,\phi)/\partial\phi$')
    ax2.invert_yaxis()
    fig.tight_layout()

    plt.show()

def test_lap_sph_har(ntest=200, nmax=100, h=1e-4):
    """Compute the Laplacian of spherical harmonics at a set of points
    randomly distributed in n, m, theta, and phi for comparison with
    a numerical approximation and create some illustrative plots
    optional args:
        ntest - number of test points
        nmax - highest allowable degree
        h - spacing of finite diff appx"""

    res = []
    yn, ym = Tnm(nmax)
    yn, ym = yn[1:], ym[1:]
    idx = np.random.randint(1, len(yn), ntest)
    for i in idx:
        n, m = yn[i], ym[i]
        t = pi*np.random.rand()
        p = 2*pi*np.random.rand()
        lap = lap_sph_har(t, p, n, m)
        tg = np.linspace(t - 2*h, t + 2*h, 5)
        pg = np.linspace(p - 2*h, p + 2*h, 5)
        try:
            yt = sph_har(tg, p, n, m)
            yp = sph_har(t, pg, n, m)
        except (ValueError, AssertionError):
            pass
        else:
            fdlap = (sin(t)*_fd2(yt,h) + cos(t)*_fd1(yt, h))/sin(t) + _fd2(yp,h)/(sin(t)**2)
            err = abs((fdlap - lap))
            res.append((n, m, t, p, err))

    n, m, t, p, err = zip(*res)
    n, m, t, p, err = (np.array(n), np.array(m), np.array(t), np.array(p), np.array(err))

    fig, ax = plt.subplots(1,1)
    r = ax.scatter(p[err > 1e-20], t[err > 1e-20], c=np.log10(err[err > 1e-20]))
    cb = plt.colorbar(r, ax=ax)
    cb.set_label(r'$\log_{10}$(Abs Error), $\nabla^2 Y_n^m(\theta,\phi)$')
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel(r'$\phi$')
    ax.set_title(r'Absolute Error, $\nabla^2 Y_n^m(\theta,\phi)$')
    fig.tight_layout()

    fig, ax = plt.subplots(1,1)
    r = ax.scatter(m[err > 1e-20], n[err > 1e-20], c=np.log10(err[err > 1e-20]))
    cb = plt.colorbar(r, ax=ax)
    cb.set_label(r'$\log_{10}$(Abs Error), $\nabla^2 Y_n^m(\theta,\phi)$')
    ax.set_xlabel('$m$')
    ax.set_ylabel('$n$')
    ax.set_title(r'Absolute Error, $\nabla^2 Y_n^m(\theta,\phi)$')
    ax.invert_yaxis()
    fig.tight_layout()

    plt.show()
    return(zip(*res))

#-------------------------------------------------------------------------------
#MAIN

print('TESTING SPHERICAL HARMONICS')

print('testing orthonormality of spherical harmonics')
err = test_sph_har_orthonormal(3)
print('maximum absolute error is %g' % err)

print('plotting spherical harmonics')
fig = plot_sph_har_pyramid(3)
plt.show()

print('testing accuracy of gradient of spherical harmonic with random points')
test_grad_sph_har()

print('testing accuracy of Laplacian of spherical harmonic with random points')
test_lap_sph_har()
