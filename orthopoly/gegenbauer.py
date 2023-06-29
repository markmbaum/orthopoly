"""
Gegenbauer polynomials :math:`C_n^m(x)` are generalizations of Chebyshev and Legendre polynomials. However, chebyshev polynomials of the first kind are implemented by other methods in the :mod:`~orthopoly.chebyshev` module, and cannot be computed by the functions in this module.
"""

import warnings

from .util import x2xhat

__all__ = ["gegen_hat", "gegen"]


def gegen_hat(xhat, n, m):
    r"""Evaluates the degree :math:`n`, order :math:`m` Gegenbauer polynomial (also called "ultraspherical" polynomial) using a three term recurrence relationship. As noted in Appendix A of Boyd, this recurrence relationship is mildly unstable for :math:`m` > 0 and the instability worsens for higher :math:`m`. However, it appears that relative error better than :math:`10^{-10}` can be expected if :math:`m` is less than about 50. More info is in appendix A of
        * Boyd, John P. Chebyshev and Fourier spectral methods. Courier Corporation, 2001.

    :param array/float xhat: evaluation point in :math:`[-1,1]`
    :param int n: degree of polynomial
    :param int m: order of polynomial (called :math:`\alpha` in some sources)

    :return: evaluated function, :math:`C_n^m(\hat{x})`"""

    # instability warning
    if m > 45:
        warnings.warn(
            "WARNING: the three term recurrence relation used to evaluate the Gegenbauer polynomials is weakly unstable for high order (m) evaluations. Relative error may be ~1e-10 for m~=49.5"
        )
    # m = 0 warning
    if m == 0:
        warnings.warn(
            "WARNING: the recurrence relationship used to compute the Gegenbauer polynomials will always yield zero for m=0."
        )
    # low n cases
    if n == 0:
        return 1.0 + 0.0 * xhat  # want to return 1 in a vectorized way
    if n == 1:
        return 2.0 * m * xhat
    # higher n cases using 3 term recurrence relation
    cprev = 1.0 + 0.0 * xhat  # 1, previous recursion value
    ccurr = 2.0 * m * xhat  # current value
    i = 1  # counter
    while i < n:
        # compute the next value and swap current with previous at the same time
        cprev, ccurr = ccurr, (
            2.0 * (i + m) * xhat * ccurr - (i + 2.0 * m - 1.0) * cprev
        ) / (i + 1.0)
        # increment
        i += 1

    return ccurr


def gegen(x, n, m, xa=-1, xb=1):
    r"""Evaluates the degree :math:`n`, order :math:`m` Gegenbauer polynomial (also called "ultraspherical" polynomial), over the interval :math:`[x_a,x_b]`, using a three term recurrence relationship. As noted in Appendix A of Boyd, this recurrence relationship is mildly unstable for :math:`m` > 0 and the instability worsens for higher :math:`m`. However, it appears that relative error better than :math:`10^{-10}` can be expected if :math:`m` is less than about 50.
        * Boyd, John P. Chebyshev and Fourier spectral methods. Courier Corporation, 2001.

    :param array/float x: evaluation points in :math:`[x_a,x_b]`
    :param int n: degree of polynomial
    :param int m: order of polynomial (called :math:`\alpha` in some sources)
    :param float xa: lower limit of evaluation interval
    :param float xb: upper limit of evaluation interval

    :return: evaluated function, :math:`C_n^m(x)`"""

    # map the evaluation points to [-1,1]
    xhat = x2xhat(x, xa, xb)
    # evaluate
    return gegen_hat(xhat, n, m)
