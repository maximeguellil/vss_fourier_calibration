"""Implied volatility and skew computations."""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .call_put import call_put_price_fourier


def black_scholes_call(S0, K, T, volatility):
    """
    Compute a Black-Scholes call price with zero rates and dividends.

    Parameters
    ----------
    S0 : float
        Initial asset price. Must be finite and strictly positive.

    K : float
        Strike price. Must be finite and strictly positive.

    T : float
        Maturity in years. Must be finite and strictly positive.

    volatility : float
        Volatility expressed as a decimal annualized volatility. For a
        non-positive value, the intrinsic value is returned, consistently
        with the convention used by the calibration notebook.

    Returns
    -------
    call_price : float
        Black-Scholes call price in the same price units as S0 and K.

    Raises
    ------
    ValueError
        If S0, K, T, or volatility is non-finite, or if S0, K, or T is not
        strictly positive.

    Notes
    -----
    The formula uses zero interest rates and no dividends:

        C = S0 * N(d1) - K * N(d2),

    with d1 = (log(S0 / K) + volatility^2 * T / 2)
    / (volatility * sqrt(T)) and d2 = d1 - volatility * sqrt(T).
    """
    try:
        S0 = float(S0)
        K = float(K)
        T = float(T)
        volatility = float(volatility)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "S0, K, T, and volatility must be scalar numeric values."
        ) from exc

    if not np.isfinite(S0) or S0 <= 0.0:
        raise ValueError("S0 must be finite and strictly positive.")
    if not np.isfinite(K) or K <= 0.0:
        raise ValueError("K must be finite and strictly positive.")
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be finite and strictly positive.")
    if not np.isfinite(volatility):
        raise ValueError("volatility must be finite.")

    if volatility <= 0.0:
        return float(max(S0 - K, 0.0))

    sigma_sqrt_T = volatility * np.sqrt(T)
    d1 = (np.log(S0 / K) + 0.5 * volatility**2 * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    return float(S0 * norm.cdf(d1) - K * norm.cdf(d2))


def implied_volatility_from_call_price(call_price, S0, K, T):
    """
    Recover the Black-Scholes implied volatility from a call price.

    Parameters
    ----------
    call_price : float or complex
        Call price to invert. If complex, only its real part is used, as in
        the Fourier-pricing calibration workflow.

    S0 : float
        Initial asset price. Must be finite and strictly positive.

    K : float
        Strike price. Must be finite and strictly positive.

    T : float
        Maturity in years. Must be finite and strictly positive.

    Returns
    -------
    implied_volatility : float
        Annualized decimal Black-Scholes implied volatility. A call price at
        the intrinsic value, up to the inversion tolerance, returns zero.

    Raises
    ------
    ValueError
        If S0, K, or T is not finite and strictly positive, if the call price
        lies outside the admissible interval beyond the price tolerance, or
        if the scalar conversion is invalid.

    Notes
    -----
    With zero rates and dividends, the admissible call-price interval is
    [max(S0 - K, 0), S0]. Prices within 1e-5 of either bound are clipped to
    the interval before inversion. The Brent search uses the volatility
    interval [1e-8, 5.0], matching the calibration notebook.
    """
    try:
        call_price = float(np.real(call_price))
        S0 = float(S0)
        K = float(K)
        T = float(T)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "call_price, S0, K, and T must be scalar numeric values."
        ) from exc

    if not np.isfinite(S0) or S0 <= 0.0:
        raise ValueError("S0 must be finite and strictly positive.")
    if not np.isfinite(K) or K <= 0.0:
        raise ValueError("K must be finite and strictly positive.")
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be finite and strictly positive.")
    if not np.isfinite(call_price):
        raise ValueError("call_price must be finite.")

    intrinsic_value = max(S0 - K, 0.0)
    tolerance = 1e-10
    price_bound_tolerance = 1e-5

    if (
        call_price < intrinsic_value - price_bound_tolerance
        or call_price > S0 + price_bound_tolerance
    ):
        raise ValueError(
            f"Call price {call_price} is outside [{intrinsic_value}, {S0}]."
        )

    call_price = min(max(call_price, intrinsic_value), S0)
    if call_price <= intrinsic_value + tolerance:
        return 0.0

    return float(
        brentq(
            lambda volatility: black_scholes_call(S0, K, T, volatility)
            - call_price,
            1e-8,
            5.0,
        )
    )


def skew_fourier(
    fourier_laplace_values,
    S_0,
    T,
    log_moneyness,
    sample_points,
    weights,
):
    """
    Compute the two-point implied-volatility skew from Fourier values.

    Parameters
    ----------
    fourier_laplace_values : array_like, shape (n_points,)
        Complex values of the same Fourier-Laplace transform evaluated at
        the quadrature sample points. The same transform is used for both
        strikes.

    S_0 : float
        Initial asset price used to reconstruct the strikes. Must be finite
        and strictly positive.

    T : float
        Maturity in years used for the Black-Scholes inversions. Must be
        finite and strictly positive.

    log_moneyness : array_like, shape (2,)
        Two distinct log-moneyness values k_i = log(K_i / S_0). Their order
        is preserved in the returned finite difference.

    sample_points : array_like, shape (n_points,)
        Fourier or Gauss-Laguerre sample points supplied to the existing
        Fourier pricing function.

    weights : array_like, shape (n_points,)
        Quadrature weights corresponding to sample_points.

    Returns
    -------
    skew : float
        Difference of the two implied volatilities divided by the difference
        of their log-moneyness values:

            (IV_2 - IV_1) / (k_2 - k_1).

        The result is a slope in annualized decimal volatility per unit of
        log-moneyness.

    Raises
    ------
    ValueError
        If S_0 or T is not finite and strictly positive, if log_moneyness
        does not have shape (2), if its values are not finite or distinct,
        or if the quadrature and transform arrays are not one-dimensional
        arrays of the same non-zero length.

    Notes
    -----
    The strikes are reconstructed as K_i = S_0 * exp(k_i). The function
    delegates all Fourier pricing to call_put_price_fourier and does not
    receive or construct VSS model parameters, kernels, covariance matrices,
    time grids, or quadrature rules.
    """
    try:
        S_0 = float(S_0)
        T = float(T)
    except (TypeError, ValueError) as exc:
        raise ValueError("S_0 and T must be scalar numeric values.") from exc

    if not np.isfinite(S_0) or S_0 <= 0.0:
        raise ValueError("S_0 must be finite and strictly positive.")
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be finite and strictly positive.")

    try:
        fourier_laplace_values = np.asarray(
            fourier_laplace_values,
            dtype=complex,
        )
        log_moneyness = np.asarray(log_moneyness, dtype=float)
        sample_points = np.asarray(sample_points, dtype=float)
        weights = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Fourier values, log-moneyness, sample points, and weights "
            "must contain numeric values."
        ) from exc

    if log_moneyness.shape != (2,):
        raise ValueError("log_moneyness must have shape (2,).")
    if not np.all(np.isfinite(log_moneyness)):
        raise ValueError("log_moneyness must be finite.")
    if log_moneyness[0] == log_moneyness[1]:
        raise ValueError("The two log-moneyness values must be distinct.")

    if (
        fourier_laplace_values.ndim != 1
        or sample_points.ndim != 1
        or weights.ndim != 1
    ):
        raise ValueError(
            "fourier_laplace_values, sample_points, and weights must be "
            "one-dimensional arrays."
        )
    if fourier_laplace_values.size == 0:
        raise ValueError(
            "fourier_laplace_values, sample_points, and weights must be "
            "non-empty."
        )
    if not (
        fourier_laplace_values.size
        == sample_points.size
        == weights.size
    ):
        raise ValueError(
            "fourier_laplace_values, sample_points, and weights must have "
            "the same length."
        )

    strikes = S_0 * np.exp(log_moneyness)
    call_prices = call_put_price_fourier(
        fourier_laplace_values=fourier_laplace_values,
        S_0=S_0,
        K_array=strikes,
        sample_points=sample_points,
        weights=weights,
        option_type="call",
    )
    call_prices = np.asarray(call_prices)
    if call_prices.shape != (2,):
        raise ValueError(
            "call_put_price_fourier must return two call prices for "
            "log_moneyness with shape (2,)."
        )

    implied_volatilities = np.array(
        [
            implied_volatility_from_call_price(call_price, S_0, strike, T)
            for call_price, strike in zip(call_prices, strikes)
        ]
    )

    return float(
        (implied_volatilities[1] - implied_volatilities[0])
        / (log_moneyness[1] - log_moneyness[0])
    )
