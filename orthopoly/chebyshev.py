r"""
The chebyshev module is a collection of functions for setting up and using Chebyshev expansions, which can be used for very high accuracy interpolation on smooth problems and for numerical solutions to boundary value problems and PDEs. Since you're reading this, I assume you already know how awesome Chebyshev expansions are! For the right problems, they have exponential, or "infinite order," convergence!

Anyway, Chebyshev polynomials (of the first kind) can be defined and evaluated in a number of ways. Here they are defined by

.. math::   T_k(\hat{x}) = \cos(k \arccos(\hat{x})) \qquad \hat{x} \in [-1,1] \, ,

where :math:`k` is the degree of the polynomial and :math:`\hat{x}`. The polynomials are orthogonal with respect to the weighted inner product

.. math:: \left< f(x),g(x) \right> = \int_{-1}^1 f(x) g(x) \frac{1}{\sqrt{1 - x^2}} dx \, .

Through a quick transformation, the polynomials can be mapped from :math:`[-1,1]` to some other interval :math:`[x_a,x_b]` using

.. math::   x(\hat{x}) = \frac{\hat{x} + 1}{2} (x_b - x_a) + x_a \qquad x \in [x_a,x_b]

and back again by

.. math::   \hat{x}(x) = 2 \frac{x - x_a}{x_b - x_a} - 1 \qquad \hat{x} \in [-1,1].

It's often convenient to use a third coordinate, :math:`\theta`, because of the trigonometric definition of the polynomials. For :math:`\theta = \cos(\hat{x})`, the Chebyshev polynomials are

.. math::   T_k(\cos(\hat{x})) = T_k(\theta) = \cos(k\theta) ,

simply cosines, where :math:`\theta` is in the interval :math:`[\pi,0]`. Using :math:`\theta` coordinates in addition to the :math:`\hat{x}` and :math:`x` coordinates simplifies derivatives of the polynomials and enables the use of discrete cosine transformations to obtain expansion coefficients, as implemeneted in the :func:`cheby_dct` function.

The first couple derivatives of the polynomials can be evaluated with respect to :math:`\theta` by using the chain rule to convert to :math:`\hat{x}` coordinates. Derivatives then must be scaled to :math:`[x_a,x_b]` by dividing by :math:`((x_b - x_a)/2)^p`, where :math:`p` is the order of the derivative.

In addition to functions for evaluating the Chebyshev polynomials and their derivatives, some higher level functions are provided by this module.
    1. :func:`cheby_bvp_setup` creates the grid and matrix used to solve boundary value problems in 1D, capable of handing boundary conditions of arbitrary derivative order and 1st or 2nd internal derivatives.
    2. :func:`cheby_coef_setup` also creates a grid and matrix, but only to solve for the expansion coefficients, without derivatives. This is just a transform from some set of points to a set of Chebyshev polynomials. It allows for the use of a first derivative on one of the boundaries and for extra points in the grid.
    3. :func:`cheby_dct_setup` creates a grid for generating chebyshev expansions using the discrete cosine transform (DCT). It does not allow for derivatives or extra points, however. The underlying function for performing a discrete chebyshev transform is :func:`cheby_dct` and the inverting function is :func:`cheby_idct`.

Tons of information on these and other methods can be found in these references:
    * Boyd, John P. *Chebyshev and Fourier spectral methods*. Courier Corporation, 2001.
    * Fornberg, Bengt. *A practical guide to pseudospectral methods. Vol. 1*. Cambridge university press, 1998.
    * Canuto, Claudio, et al. *Spectral methods*. Springer-Verlag, Berlin, 2006.
"""

from numpy import pi, cos, sin, sqrt, prod, arccos, arange, array, zeros, sort
from scipy.fftpack import dct, idct

from .util import x2xhat, xhat2x

__all__ = [
    "insert_points",
    "cheby_grid",
    "cheby",
    "cheby_hat",
    "cheby_hat_ext",
    "dcheby_t",
    "ddcheby_t",
    "dpcheby_boundary",
    "cheby_sum",
    "dcheby_t_sum",
    "cheby_hat_sum",
    "cheby_hat_ext_sum",
    "dcheby_coef",
    "cheby_hat_recur_sum",
    "dcheby_hat_recur_sum",
    "cheby_dct",
    "cheby_idct",
    "cheby_bvp_setup",
    "cheby_coef_setup",
    "cheby_dct_setup",
]

# -------------------------------------------------------------------------------
# grid functions


def insert_points(xhat, theta, x, xextra, xa, xb):
    r"""Puts some extra points into the cheby grid, ignoring duplicates and keeping things correctly sorted

    :param array xhat: grid points on :math:`[-1,1]`
    :param array theta: grid points on :math:`[\pi,0]`
    :param array x: grid points on :math:`[x_a,x_b]`
    :param array xextra: the x points to insert
    :param float xa: left boundary
    :param float xb: right boundary

    :return: tuple containing

        - grid points on :math:`[-1,1]` with points inserted
        - grid points on :math:`[\pi,0]` with points inserted
        - grid points on :math:`[x_a,x_b]` with points inserted"""

    xextra = array(xextra)
    assert all(
        (x >= xa) | (x <= xb)
    ), "extra points cannot be outside the domain [xa,xb]"

    x = sort(array(list(set(x) | set(xextra))))
    xhat = x2xhat(x, xa, xb)
    theta = arccos(xhat)

    return (xhat, theta, x)


def cheby_grid(xa, xb, n):
    r"""Computes the chebyshev "extreme points" for use with all the functions in this module

    The grid points are in :math:`[-1,1]` and they are

    .. math:: \hat{x}_i = \cos\left(\frac{\pi j}{n-1}\right) \qquad j = n-1,...,0 \, .

    These points are returned with their counterparts in :math:`\theta` and :math:`x` coordinates.

    :param float xa: value of left (lower) domain boundary
    :param float xb: value of right (higher) domain boundary
    :param int n: number of points to use in the domain (including boundary points)

    :return: tuple containing

        - array of collocation points (:math:`\hat{x}` points) in :math:`[-1,1]`
        - array of theta values, :math:`\arccos(\hat{x})`, in :math:`[0,\pi]`
        - array collocation points in :math:`[x_a,x_b]`"""

    # order of expansion
    ord = n - 1
    # grid (collocation/interpolation points)
    j = arange(0, n)
    theta = (pi * j / ord)[::-1]
    xhat = cos(theta)  # points in [-1,1]
    x = xhat2x(xhat, xa, xb)  # points in [xa, xb]

    return (xhat, theta, x)


# -------------------------------------------------------------------------------
# evaluating Chebyshev polynomials and their derivaties


def cheby(x, k, xa=-1, xb=1):
    r"Evaluates the :math:`k^{\textrm{th}}` chebyshev polynomial in :math:`[x_a,x_b]` at :math:`x`"
    return cos(k * arccos(x2xhat(x, xa, xb)))


def cheby_hat(xhat, k):
    r"Evaluates the :math:`k^{\textrm{th}}` chebyshev polynomial in :math:`[-1,1]`"
    return cos(k * arccos(xhat))


def dcheby_t(t, k):
    r"Evaluates the first derivative of the :math:`k^{\textrm{th}}` order chebyshev polynomial at theta values"
    return k * sin(k * t) / sin(t)


def ddcheby_t(t, k):
    r"Evaluates the second derivative of the :math:`k^{\textrm{th}}` order chebyshev polynomial at theta values"
    return (sin(t) * (-k * k * cos(k * t)) - cos(t) * (-k * sin(k * t))) / (sin(t) ** 3)


def cheby_hat_ext(xhat, k):
    r"""Evaluates a chebyshev polynomial in :math:`\hat{x}` space, but potentially outside of :math:`[-1,1]`

    :param float/array xhat: argument
    :param int k: polynomial degree

    :return: :math:`T_k(\hat{x})`"""

    if abs(xhat) <= 1.0:
        return cheby_hat(xhat, k)
    else:
        return (
            (xhat - sqrt(xhat**2 - 1)) ** k + (xhat + sqrt(xhat**2 - 1)) ** k
        ) / 2


def dpcheby_boundary(sign, k, p):
    r"""Evaluates the :math:`p^{\textrm{th}}` derivative of the :math:`k^{\textrm{th}}` order chebyshev polynomial at -1 or 1

    :param int sign: -1 or 1
    :param int k: polynomial degree
    :param int p: derivative order

    :return: :math:`d T_k(\pm 1) / dx`"""

    assert sign == 1 or sign == -1, "sign must be -1 or 1"
    n = arange(0, p)
    d = (sign ** (k + p)) * prod((k**2 - n**2) / (2 * n + 1))
    return d


# -------------------------------------------------------------------------------
# evaluating Chebyshev expansions and their derivaties


def cheby_sum(x, a, xa, xb):
    "Evaluates a chebyshev expansion in :math:`[x_a,x_b]` with coefficient array :math:`a_i` by direct summation of the polynomials"
    return sum(a[k] * cheby(x, k, xa, xb) for k in range(len(a)))


def dcheby_t_sum(t, a):
    r"Evaluates the first derivative of the cheby expansion at :math:`\theta` values"
    return sum(a[k] * dcheby_t(t, k) for k in range(len(a)))


def cheby_hat_sum(xhat, a):
    "Evaluates a chebyshev series in :math:`[-1,1]` with coefficient array :math:`a_i` by direct summation of the polynomials"
    return sum(a[i] * cheby_hat(xhat, i) for i in range(len(a)))


def cheby_hat_ext_sum(xhat, a):
    r"""Evaluates chebyshev expansion anywhere, including outside :math:`[-1,1]`

    :param float/array xhat: argument
    :param array a: expansion coefficients :math:`a_i`

    :return: evaluated expansion"""

    if abs(xhat) < 1.0:
        return cheby_hat_sum(xhat, a)
    else:
        return sum(
            (
                a[k]
                * (
                    (xhat - sqrt(xhat**2 - 1)) ** k
                    + (xhat + sqrt(xhat**2 - 1)) ** k
                )
                / 2
            )
            for k in range(len(a))
        )


def dcheby_coef(a):
    """Computes the coefficents of the derivative of a Chebyshev expansion using a recurrence relation. Higher order differentiation (using this function repeatedly on the same original coefficents) is mildly ill-conditioned. See these references for more info:
        * Boyd, John P. Chebyshev and Fourier spectral methods. Courier Corporation, 2001.
        * Press, William H., et al. Numerical Recipes: The Art of Scientific Computing. 3rd ed., Cambridge University Press, 2007.

    :param array a: expansion coefficients :math:`a_i`

    :return: expansion coefficients of the input expansion's derivative"""

    n = len(a)  # number of coefficients
    c = zeros((n,))
    c[n - 2] = 2 * (n - 1) * a[n - 1]
    for i in range(n - 2, 0, -1):
        c[i - 1] = c[i + 1] + 2 * i * a[i]
    c[0] /= 2
    return c[:-1]


def cheby_hat_recur_sum(xhat, a):
    r"""Evaluates a Chebyshev expansion using a recurrence relationship. This is helpful for things like root finding near the boundaries of the domain because, if the terms are evaluated with :math:`\arccos`, they are not defined outside the domain. Any tiny departure outside the boundaries triggers a NaN. This recurrence is widely cited, but of course, See
        * Boyd, John P. Chebyshev and Fourier spectral methods. Courier Corporation, 2001.

    :param float/array xhat: argument
    :param array a: expansion coefficients :math:`a_i`

    :return: evaluated expansion"""

    # first two terms
    c0, c1 = 1.0, xhat
    # apply them
    y = a[0] * c0 + a[1] * c1
    # recurr
    for i in range(2, len(a)):
        c2 = 2 * xhat * c1 - c0
        y += c2 * a[i]
        c0 = c1
        c1 = c2
    return y


def dcheby_hat_recur_sum(xhat, a):
    """Computes the derivative of a Chebyshev expansion in :math:`[-1,1]` using a recurrence relation, avoiding division by zero at the boundaries when using :func:`dcheby_t`. This is the same procedure as in :func:`dcheby_coef`, but using the derivative coefficients to evaluate the derivative as the recurrence proceeds instead of just storing and returning them.

    :param float/array xhat: argument
    :param array a: expansion coefficients :math:`a_i`

    :return: evaluated expansion"""

    n = len(a)  # number of coefficients
    dy = 0.0
    c2 = 0.0  # recurrence group
    c1 = 2 * (n - 1) * a[n - 1]  # initial, highest order coefficient
    dy += c1 * cheby_hat_ext(xhat, n - 2)
    for i in range(n - 2, 1, -1):
        c0 = c2 + 2 * i * a[i]
        dy += c0 * cheby_hat_ext(xhat, i - 1)
        c2 = c1
        c1 = c0
    dy += (c2 + 2 * a[1]) / 2.0
    return dy


"""def cheby_matrix(xhat, kmax, deriv=0):

    assert 0 <= deriv <= 2

    n, m = len(xhat), kmax
    A = zeros((n, m))

    for i in range(n):
        for k in range(kmax):
            if xhat == 1 or xhat == -1:
                if deriv == 0:
                    A[i,k] ="""

# -------------------------------------------------------------------------------
# discrete Chebyshev transform


def cheby_dct(y):
    r"""Finds Chebyshev expansion coefficients :math:`a_k` using a discrete cosine transform (DCT). For high order expansions, this should be much faster than computing coefficients by solving a linear system.

    The input vector is assumed to contain the values for points lying on a :func:`chebyshev grid <cheby_grid>`. The Chebyshev expansion is assumed to have the form

    .. math:: \sum_{k=0}^{n-1} a_k \cos(k \arccos(\hat{x})) \qquad \hat{x} \in [-1,1]

    However, because the Type I DCT in `scipy.fftpack` has a slightly different convention, some small modifications are made to the results of `scipy.fftpack.dct` to achieve coefficents appropriate for the expansion above.

    :param array y: values of points on a :func:`chebyshev grid <cheby_grid>`, in the proper order

    :return: array of Chebyshev coefficients :math:`a_k`"""

    # length of transform
    n = len(y)
    # transform
    a = dct(y, type=1, n=n) / (2 * n - 2)
    # a little bit of correction for conventions
    a[1:-1] *= 2
    a[1::2] *= -1

    return a


def cheby_idct(a):
    r"""Evaluates a chebyshev series using the inverse discrete cosine transform. For high order expansions, this should be much faster than direct summation of the polynomials.

    The input vector is assumed to contain the values for points lying on a :func:`chebyshev grid <cheby_grid>`. The Chebyshev expansion is assumed to have the form

    .. math:: \sum_{k=0}^{n-1} a_k \cos(k \arccos(\hat{x})) \qquad \hat{x} \in [-1,1]

    Because of varying conventions, this function should only be used with chebyshev expansion coefficients computed with the functions in this package (:func:`cheby_coef_setup` and :func:`cheby_dct`). The Type I IDCT from `scipy.fftpack` is used (with minor modifications).

    :param array a: chebyshev expansion coefficients :math:`a_k`

    :return: array of evaluated expansion values at :func:`chebyshev grid <cheby_grid>` points
    """

    # copy and correct for convention
    a = a.copy()
    a[1:-1] /= 2
    a[1::2] *= -1
    # transform
    y = idct(a, type=1)

    return y


# -------------------------------------------------------------------------------
# high-level functions for setting up transforms


def cheby_bvp_setup(xa, xb, n, aderiv, ideriv, bderiv, xextra=None):
    r"""Set up the grid and matrices for a Chebyshev spectral solver for boundary value problems in 1 cartesian dimension

    :param float xa: value of left (lower) domain boundary
    :param float xb: value of right (higher) domain boundary
    :param int n: number of points to use in the domain (including boundaries)
    :param int aderiv: order of derivative for left boundary condition (can be 0)
    :param int ideriv: order of derivative for internal nodes (must be 1 or 2)
    :param int bderiv: order of derivative for right boundary condition (can be 0)
    :param xextra: extra points within the domain to include as collocation points

    :return: tuple containing

        - collocation points in :math:`[-1,1]`
        - theta values, :math:`\arccos(\hat{x})`, in :math:`[0,\pi]`
        - collocation points in :math:`[x_a,x_b]`
        - matrix to solve for the expansion coefficients"""

    assert aderiv == 0 or bderiv == 0, "either aderiv or bderiv must be zero"
    assert ideriv > 0 and ideriv < 3, "ideriv must be 1 or 2"

    # scale factor for derivatives when mapping [-1,1] to [xa, xb]
    scale = (xb - xa) / 2.0  # length of [xa,xb] divided by length of [-1,1]

    # grid points
    xhat, theta, x = cheby_grid(xa, xb, n)
    if xextra is not None:
        xhat, theta, x = insert_points(xhat, theta, x, xextra, xa, xb)
        n = len(x)

    # solver matrix (j is for rows/points and k is for colums or expansion order)
    A = zeros((n, n))
    # left boundary condition/row
    if aderiv > 0:
        for k in range(n):
            A[0, k] = dpcheby_boundary(-1, k, aderiv) / (scale**aderiv)
    else:
        for k in range(n):
            A[0, k] = cheby_hat(-1, k)
    # internal nodes
    if ideriv == 1:
        dfunk = dcheby_t
    elif ideriv == 2:
        dfunk = ddcheby_t
    for j in range(1, n - 1):
        for k in range(n):
            # have to do the js backward if the x values are to be in increasing order
            A[j, k] = dfunk(theta[j], k) / (scale**ideriv)
    # right boundary condition/row
    if bderiv > 0:
        for k in range(n):
            A[-1, k] = dpcheby_boundary(1, k, bderiv) / (scale**bderiv)
    else:
        for k in range(n):
            A[-1, k] = cheby_hat(1, k)

    return (xhat, theta, x, A)


def cheby_coef_setup(xa, xb, n, da=False, db=False, xextra=None):
    r"""Constructs the grid and matrix for finding the coefficients of the chebyshev expansion on :math:`[x_a,x_b]` with :math:`n` points

    :param float xa: value of left (lower) domain boundary
    :param float xb: value of right (higher) domain boundary
    :param int n: number of points to use in the domain (including boundaries)
    :param bool da: use the first deriv at the left/low boundary as the boundary condition
    :param bool da: use the first deriv at the right/high boundary as the boundary condition
    :param array xextra: extra points within the domain to include as collocation points

    :return: tuple containing

        - collocation points in :math:`[-1,1]`
        - theta values, :math:`\arccos(\hat{x})`, in :math:`[0,\pi]`
        - collocation points in :math:`[x_a,x_b]`
        - matrix to solve for the expansion coefficients
        - scale factor for derivatives"""

    # scale factor for derivatives when mapping [-1,1] to [xa, xb]
    scale = (xb - xa) / 2.0  # length of [xa,xb] divided by length of [-1,1]

    # grid points
    xhat, theta, x = cheby_grid(xa, xb, n)
    if xextra is not None:
        xhat, theta, x = insert_points(xhat, theta, x, xextra, xa, xb)
        n = len(x)

    # solver matrix (j is for rows/points and k is for colums or expansion order)
    A = zeros((n, n))
    for j in range(n):
        for k in range(n):
            A[j, k] = cheby_hat(xhat[j], k)
    if da:
        for k in range(n):
            A[0, k] = dpcheby_boundary(-1, k, 1) / scale
    if db:
        for k in range(n):
            A[-1, k] = dpcheby_boundary(1, k, 1) / scale

    return (xhat, theta, x, A, scale)


def cheby_dct_setup(xa, xb, n):
    r"""Constructs a Chebyshev grid and return a function for computing Chebyshev expansion coefficients on the grid with a discrete cosine transform (DCT). The returned function wraps :func:`cheby_dct`.

    :param float xa: value of left (lower) domain boundary
    :param float xb: value of right (higher) domain boundary
    :param int n: number of points to use in the domain (including boundaries)

    :return: tuple containing

        - collocation points in :math:`[-1,1]`
        - theta values, :math:`\arccos(\hat{x})`, in :math:`[0,\pi]`
        - collocation points in :math:`[x_a,x_b]`
        - function for computing expansion coefficients
        - scale factor for derivatives"""

    # scale factor for derivatives when mapping [-1,1] to [xa, xb]
    scale = (xb - xa) / 2.0  # length of [xa,xb] divided by length of [-1,1]

    # grid points
    xhat, theta, x = cheby_grid(xa, xb, n)

    return (xhat, theta, x, cheby_dct, scale)
