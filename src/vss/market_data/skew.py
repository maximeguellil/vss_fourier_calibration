"""Market implied-volatility skew computations."""

import numpy as np


def compute_market_skew(
    maturities,
    log_moneyness,
    implied_volatilities,
):
    """
    Compute the two-point implied-volatility skew term structure.

    Parameters
    ----------
    maturities : array_like, shape (n_maturities,)
        Strictly positive option maturities in years. The values are used to
        align the rows of implied_volatilities with the returned skew curve.

    log_moneyness : array_like, shape (2,)
        The two selected log-moneyness values k_i = log(K_i / S_0). They must
        be finite and distinct. The order determines the sign of the skew.

    implied_volatilities : array_like, shape (n_maturities, 2)
        Annualized decimal implied volatilities corresponding to the two
        log-moneyness values, with one row per maturity.

    Returns
    -------
    market_skews : ndarray, shape (n_maturities,)
        Implied-volatility slope at each maturity:

            (implied_volatilities[:, 1]
             - implied_volatilities[:, 0])
            / (log_moneyness[1] - log_moneyness[0]).

        The result is expressed in annualized decimal volatility per unit of
        log-moneyness.

    Raises
    ------
    ValueError
        If the inputs are not numeric, if maturities do not have shape
        (n_maturities,), if log_moneyness does not have shape (2), if
        implied_volatilities does not have shape (n_maturities, 2), if the
        two log-moneyness values are not finite and distinct, or if any
        maturity or implied volatility is non-finite or outside its domain.

    Notes
    -----
    This function does not choose the two moneyness values and does not
    interpolate the implied-volatility surface. If the source data are stored
    with moneyness columns K / S_0, convert the selected columns explicitly
    before calling this function:

        log_moneyness = np.log(selected_moneyness).

    The finite-difference convention is the same as skew_fourier.
    """
    try:
        maturities = np.asarray(maturities, dtype=float)
        log_moneyness = np.asarray(log_moneyness, dtype=float)
        implied_volatilities = np.asarray(
            implied_volatilities,
            dtype=float,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "maturities, log_moneyness, and implied_volatilities must "
            "contain numeric values."
        ) from exc

    if maturities.ndim != 1:
        raise ValueError("maturities must be one-dimensional.")

    if log_moneyness.shape != (2,):
        raise ValueError("log_moneyness must have shape (2,).")

    if implied_volatilities.shape != (maturities.size, 2):
        raise ValueError(
            "implied_volatilities must have shape "
            f"({maturities.size}, 2)."
        )

    if not np.all(np.isfinite(maturities)) or np.any(maturities <= 0.0):
        raise ValueError(
            "Maturities must be finite and strictly positive."
        )

    if not np.all(np.isfinite(log_moneyness)):
        raise ValueError("log_moneyness must be finite.")

    if log_moneyness[0] == log_moneyness[1]:
        raise ValueError("The two log-moneyness values must be distinct.")

    if (
        not np.all(np.isfinite(implied_volatilities))
        or np.any(implied_volatilities <= 0.0)
    ):
        raise ValueError(
            "Implied volatilities must be finite and strictly positive."
        )

    return (
        implied_volatilities[:, 1] - implied_volatilities[:, 0]
    ) / (log_moneyness[1] - log_moneyness[0])
