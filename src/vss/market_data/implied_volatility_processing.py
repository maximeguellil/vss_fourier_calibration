"""Cleaning and validation of implied-volatility market data."""

import numpy as np
import pandas as pd


def check_implied_volatility_data(implied_volatility_data):
    """
    Validate a maturity-by-moneyness implied-volatility matrix.

    Parameters
    ----------
    implied_volatility_data : pandas.DataFrame
        Two-dimensional matrix whose index contains maturities in years, whose
        columns contain moneyness values K / S_0, and whose cells contain
        annualized decimal implied volatilities.

    Returns
    -------
    is_valid : bool
        True when the matrix satisfies all structural and value checks.

    Raises
    ------
    TypeError
        If implied_volatility_data is not a pandas.DataFrame.

    ValueError
        If the matrix is empty, if maturities or moneyness values are not
        finite and strictly positive, if maturities or moneyness values are
        duplicated or not strictly increasing, or if an implied volatility is
        missing, non-finite, or not strictly positive.

    Notes
    -----
    The Excel representation stores log-moneyness, while the extracted
    matrix stores moneyness K / S_0. Strictly positive finite moneyness is
    equivalent to finite log-moneyness through log(K / S_0).
    """
    if not isinstance(implied_volatility_data, pd.DataFrame):
        raise TypeError("implied_volatility_data must be a pandas.DataFrame.")

    if implied_volatility_data.empty:
        raise ValueError("implied_volatility_data must not be empty.")

    if (
        implied_volatility_data.index.has_duplicates
        or implied_volatility_data.columns.has_duplicates
    ):
        raise ValueError(
            "Maturities and moneyness values must not be duplicated."
        )

    try:
        maturities = np.asarray(implied_volatility_data.index, dtype=float)
        moneyness = np.asarray(implied_volatility_data.columns, dtype=float)
        implied_volatilities = implied_volatility_data.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Maturities, moneyness values, and implied volatilities must be "
            "numeric."
        ) from exc

    if not np.all(np.isfinite(maturities)) or np.any(maturities <= 0.0):
        raise ValueError(
            "Maturities must be finite and strictly positive."
        )

    if np.any(np.diff(maturities) <= 0.0):
        raise ValueError("Maturities must be strictly increasing.")

    if (
        not np.all(np.isfinite(moneyness))
        or np.any(moneyness <= 0.0)
        or not np.all(np.isfinite(np.log(moneyness)))
    ):
        raise ValueError(
            "Moneyness values must be finite and strictly positive."
        )

    if np.any(np.diff(moneyness) <= 0.0):
        raise ValueError("Moneyness values must be strictly increasing.")

    if (
        not np.all(np.isfinite(implied_volatilities))
        or np.any(implied_volatilities <= 0.0)
    ):
        raise ValueError(
            "Implied volatilities must be finite and strictly positive."
        )

    return True


def clean_implied_volatility_data(implied_volatility_data):
    """
    Clean and order a maturity-by-moneyness implied-volatility matrix.

    Parameters
    ----------
    implied_volatility_data : pandas.DataFrame
        Two-dimensional matrix whose index contains maturities in years, whose
        columns contain moneyness values K / S_0, and whose cells contain
        annualized decimal implied volatilities.

    Returns
    -------
    cleaned_data : pandas.DataFrame
        A copy of the input matrix with numeric labels, invalid maturity
        labels and moneyness labels removed, rows containing a missing,
        non-finite, or non-positive implied volatility removed, and valid
        maturities and moneyness values sorted increasingly.

    Raises
    ------
    TypeError
        If implied_volatility_data is not a pandas.DataFrame.

    ValueError
        If duplicate maturity or moneyness labels are present, or if no valid
        implied-volatility observation remains after cleaning.

    Notes
    -----
    Cleaning concerns missing, non-finite, and out-of-domain observations
    only. It does not perform an arbitrage check. A row is removed when at
    least one implied-volatility cell in that maturity row is invalid; no
    invalid value is replaced by an inferred or interpolated value.
    """
    if not isinstance(implied_volatility_data, pd.DataFrame):
        raise TypeError("implied_volatility_data must be a pandas.DataFrame.")

    if (
        implied_volatility_data.index.has_duplicates
        or implied_volatility_data.columns.has_duplicates
    ):
        raise ValueError(
            "Maturities and moneyness values must not be duplicated."
        )

    cleaned_data = implied_volatility_data.copy()
    cleaned_data.index = pd.to_numeric(cleaned_data.index, errors="coerce")
    cleaned_data.columns = pd.to_numeric(
        cleaned_data.columns,
        errors="coerce",
    )
    cleaned_data = cleaned_data.apply(pd.to_numeric, errors="coerce")
    cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)

    maturities = np.asarray(cleaned_data.index, dtype=float)
    valid_maturities = np.isfinite(maturities) & (maturities > 0.0)
    cleaned_data = cleaned_data.loc[valid_maturities]

    moneyness = np.asarray(cleaned_data.columns, dtype=float)
    valid_moneyness = np.isfinite(moneyness) & (moneyness > 0.0)
    cleaned_data = cleaned_data.loc[:, valid_moneyness]

    valid_rows = np.all(
        np.isfinite(cleaned_data.to_numpy(dtype=float))
        & (cleaned_data.to_numpy(dtype=float) > 0.0),
        axis=1,
    )
    cleaned_data = cleaned_data.loc[valid_rows]

    cleaned_data = cleaned_data.sort_index().sort_index(axis=1)
    check_implied_volatility_data(cleaned_data)
    return cleaned_data
