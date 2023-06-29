"""
This module implements the Associated Legendre Polynomials, :math:`P_n^m(x)`, and their first two derivatives in support of the :mod:`~orthopoly.spherical_harmonic` module. If :math:`m=0`, they reduce to the unassociated Legendre polynomials.
"""

__all__ = [
    "legen_norm",
    "legen_hat",
    "legen",
    "legen_theta",
    "dlegen_theta",
    "ddlegen_theta",
]

from numpy import sqrt, pi, cos
import warnings

from .util import x2xhat


def _check_legen_hat_args(xhat, n, m):
    """Checks arguments to Legendre functions are valid"""

    # check points within bounds
    assert all(abs(xhat) <= 1.0), "can't evaluate associated Lege ndre outside [-1,1]"
    # check order isn't negative
    assert m >= 0, "can't have m < 0 in evaluating associated Legendre"
    # warn about order greater than degree
    if m > n:
        warnings.warn(
            "A Legendre function of order greater than degree (m > n) was evaluated. These are zero by the definition and it's unusual to need them."
        )
        return 0.0 * xhat


def legen_norm(n, m):
    r"""Evaluates the normalization factor for the associated Legendre polynomials,

    .. math::
        :nowrap:

        \begin{equation}
            \sqrt{\frac{(2n + 1) (n - m)!}{4 \pi (n + m)!}}
        \end{equation}

    with an iteration instead of direct factorials to help avoid overflow. This function is not used in the other functions in this module, which compute the normalized polynomials without directly calculating the normalization factor. However, it can be used to switch between normalized and unnormalized values.

    :param int n: degree of polynomial
    :param int m: order of polynomial (m >= 0)

    :return: normalization factor"""

    f = sqrt((2 * n + 1) / (4 * pi))
    for i in range(n - m + 1, n + m + 1):
        f /= sqrt(float(i))

    return f


def legen_hat(xhat, n, m):
    r"""Evaluates the normalized associated Legendre function of degree n and order m, :math:`P_n^m(x)`, through a three term recurrence relationship. These are **normalized** and the normalization factor is:

    .. math::
        :nowrap:

        \begin{equation}
            \sqrt{\frac{(2n + 1) (n - m)!}{4 \pi (n + m)!}}
        \end{equation}

    as defined in chapter 6 of the reference below, but without the extra factor of :math:`-1^m`
        * Press, William H., et al. Numerical recipes 3rd edition: The art of scientific computing. Cambridge university press, 2007.

    :param array/float xhat: evaluation point in :math:`[-1,1]` (can be an array)
    :param int n: degree of polynomial
    :param int m: order of polynomial (m >= 0)

    :return: evaluated function, :math:`P_n^m(\hat{x})`"""

    # check for issues with input
    _check_legen_hat_args(xhat, n, m)

    # evaluate the initial recursion value, P_m^m, with stable factorials
    pprev = 1.0 + 0.0 * xhat  # running value for factorials, which becomes P_m^m
    if m > 0:  # account for 0! and avoid division by zero
        pprev /= sqrt(2.0 * m)
        for i in range(2 * m - 1, 1, -1):
            if (2 * m - 1 - i) % 2 == 0:  # double factorial only on every other
                pprev *= i
            pprev /= sqrt(i)
    pprev *= sqrt((2.0 * m + 1.0) / (4.0 * pi)) * ((1.0 - xhat**2) ** (m / 2.0))
    if n == m:
        return pprev
    # evaluate the second value
    pcurr = xhat * sqrt(2.0 * m + 3.0) * pprev
    if n == m + 1:
        return pcurr
    # do the recursion
    i = m + 2
    while i <= n:
        # prefactors
        a = sqrt((4.0 * i**2 - 1.0) / (i**2 - m**2))
        b = sqrt(((i - 1.0) ** 2 - m**2) / (4.0 * (i - 1.0) ** 2 - 1.0))
        # evaluate and swap simultaneously
        pprev, pcurr = pcurr, a * (xhat * pcurr - b * pprev)
        # increment
        i += 1

    return pcurr


def legen(x, n, m, xa=-1, xb=1):
    r"""Evaluates the normalized associated Legendre function of degree n and order m, :math:`P_n^m(x)`, through a three term recurrence relationship, over the interval :math:`[x_a,x_b]`. These are **normalized** and the normalization factor is:

    .. math::
        :nowrap:

        \begin{equation}
            \sqrt{\frac{(2n + 1) (n - m)!}{4 \pi (n + m)!}}
        \end{equation}

    as defined in chapter 6 of the reference below, but without the extra factor of :math:`-1^m`
        * Press, William H., et al. Numerical recipes 3rd edition: The art of scientific computing. Cambridge university press, 2007.

    :param array/float x: evaluation point in :math:`[x_a,x_b]` (can be an array)
    :param int n: degree of polynomial
    :param int m: order of polynomial (m >= 0)
    :param float xa: lower limit of evaluation interval
    :param float xb: upper limit of evaluation interval

    :return: evaluated function, :math:`P_n^m(x)`"""

    # map the evaluation points to [-1,1]
    xhat = x2xhat(x, xa, xb)
    # evaluate
    return legen_hat(xhat, n, m)


def legen_theta(t, n, m):
    r"""Evaluates the normalized associated Legendre function :math:`P_n^m(cos(\theta))` with a colatitude argument in :math:`[0,\pi]` instead of :math:`[-1,1]`

    :param array/float t: colatitude evaluation point(s) in :math:`[0,\pi]` (can be an array)
    :param int n: degree of polynomial
    :param int m: order of polynomial (m >= 0)

    :return: evaluated function, :math:`P_n^m[\cos(\theta)]`"""

    # call the regular Legendre function with xhat=cos(theta)
    return legen_hat(cos(t), n, m)


def dlegen_theta(t, n, m):
    r"""Evaluates the first derivative of the normalized associated Legendre function with colatitude argument, :math:`d P_n^m / d \theta`. This can be used in computing the gradient of spherical harmonics. The algorithm is detailed in:
        * Bosch, W. "On the computation of derivatives of Legendre functions." Physics and Chemistry of the Earth, Part A: Solid Earth and Geodesy 25.9-11 (2000): 655-659.

    :param array/float t: colatitude evaluation point(s) in :math:`[0,\pi]` (can be an array)
    :param int n: degree of polynomial
    :param int m: order of polynomial (m >= 0)

    :return: evaluated first derivative, :math:`d P_n^m / d \theta`"""

    if m == 0 and n == 0:
        return 0.0 * t
    elif m == 0:
        return -sqrt(n * (n + 1.0)) * legen_theta(t, n, 1)
    elif m == n:
        return sqrt(n / 2) * legen_theta(t, n, n - 1)
    else:
        return (
            sqrt((n + m) * (n - m + 1)) * legen_theta(t, n, m - 1)
            - sqrt((n + m + 1) * (n - m)) * legen_theta(t, n, m + 1)
        ) / 2


def ddlegen_theta(t, n, m):
    r"""Evaluates the second derivative of the normalized associated Legendre function with colatitude argument, :math:`d^2 P_n^m / d \theta^2`. This can be used in computing the Laplacian of spherical harmonics. The algorithm is detailed in:
        * Bosch, W. "On the computation of derivatives of Legendre functions." Physics and Chemistry of the Earth, Part A: Solid Earth and Geodesy 25.9-11 (2000): 655-659.

    :param array/float t: colatitude evaluation point(s) in :math:`[0,\pi]` (can be an array)
    :param int n: degree of polynomial
    :param int m: order of polynomial (m >= 0)

    :return: evaluated second derivative, :math:`d^2 P_n^m / d \theta^2`"""

    if m == 0 and n == 0:
        return 0.0 * t
    elif m == 0:
        return -sqrt(n * (n + 1)) * dlegen_theta(t, n, 1)
    elif m == n:
        return sqrt(n / 2) * dlegen_theta(t, n, n - 1)
    else:
        return (
            sqrt((n + m) * (n - m + 1)) * dlegen_theta(t, n, m - 1)
            - sqrt((n + m + 1) * (n - m)) * dlegen_theta(t, n, m + 1)
        ) / 2
