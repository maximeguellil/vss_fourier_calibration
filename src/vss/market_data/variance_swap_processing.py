"""Cleaning and validation of variance-swap market data."""

import numpy as np
import pandas as pd


VARIANCE_SWAP_COLUMNS = [
    "maturity",
    "variance_swap_quote",
]


def check_variance_swap_data(variance_swap_data):
    """
    Validate the structure and domain of a variance-swap term structure.

    Parameters
    ----------
    variance_swap_data : pandas.DataFrame
        Long-format table with columns maturity and variance_swap_quote.
        Maturities are expressed in years and quotes are annualized decimal
        volatility quotes for variance swaps.

    Returns
    -------
    is_valid : bool
        True when the table satisfies all structural and value checks.

    Raises
    ------
    TypeError
        If variance_swap_data is not a pandas.DataFrame.

    ValueError
        If a required column is missing, if the table is empty, if maturities
        or quotes are non-numeric, non-finite, or out of domain, if maturity
        is duplicated, or if maturities are not strictly increasing.

    Notes
    -----
    This function checks the observations themselves. It does not check
    variance-swap arbitrage conditions; those are implemented separately by
    check_variance_swap_arbitrage.
    """
    if not isinstance(variance_swap_data, pd.DataFrame):
        raise TypeError("variance_swap_data must be a pandas.DataFrame.")

    missing_columns = [
        column
        for column in VARIANCE_SWAP_COLUMNS
        if column not in variance_swap_data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required variance-swap columns: {missing_columns}."
        )

    if variance_swap_data.columns.has_duplicates:
        raise ValueError("Variance-swap column names must not be duplicated.")

    if variance_swap_data.empty:
        raise ValueError("variance_swap_data must not be empty.")

    try:
        maturities = variance_swap_data["maturity"].to_numpy(dtype=float)
        quotes = variance_swap_data[
            "variance_swap_quote"
        ].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Maturities and variance-swap quotes must be numeric."
        ) from exc

    if not np.all(np.isfinite(maturities)) or np.any(maturities <= 0.0):
        raise ValueError(
            "Maturities must be finite and strictly positive."
        )

    if np.any(np.diff(maturities) <= 0.0):
        raise ValueError("Maturities must be strictly increasing.")

    if not np.all(np.isfinite(quotes)) or np.any(quotes < 0.0):
        raise ValueError(
            "Variance-swap quotes must be finite and non-negative."
        )

    if variance_swap_data["maturity"].duplicated().any():
        raise ValueError("Variance-swap maturities must not be duplicated.")

    return True


def clean_variance_swap_data(variance_swap_data):
    """
    Clean and order a variance-swap term structure.

    Parameters
    ----------
    variance_swap_data : pandas.DataFrame
        Long-format table with columns maturity and variance_swap_quote.
        Maturities are expressed in years and quotes are annualized decimal
        volatility quotes for variance swaps.

    Returns
    -------
    cleaned_data : pandas.DataFrame
        A copy containing the required columns and valid observations only.
        Non-numeric, missing, non-finite, non-positive maturity values and
        negative or non-finite quotes are removed. The remaining observations
        are sorted by strictly increasing maturity.

    Raises
    ------
    TypeError
        If variance_swap_data is not a pandas.DataFrame.

    ValueError
        If required columns are missing, if maturity labels are duplicated,
        or if no valid observation remains after cleaning.

    Notes
    -----
    Cleaning does not alter a quote and does not check or correct arbitrage.
    In particular, an observation is never removed because total variance is
    not increasing. Such a structure is reported by
    check_variance_swap_arbitrage.
    """
    if not isinstance(variance_swap_data, pd.DataFrame):
        raise TypeError("variance_swap_data must be a pandas.DataFrame.")

    missing_columns = [
        column
        for column in VARIANCE_SWAP_COLUMNS
        if column not in variance_swap_data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required variance-swap columns: {missing_columns}."
        )

    if variance_swap_data.columns.has_duplicates:
        raise ValueError("Variance-swap column names must not be duplicated.")

    if variance_swap_data["maturity"].duplicated().any():
        raise ValueError("Variance-swap maturities must not be duplicated.")

    cleaned_data = variance_swap_data.loc[
        :,
        VARIANCE_SWAP_COLUMNS,
    ].copy()
    cleaned_data = cleaned_data.apply(pd.to_numeric, errors="coerce")
    cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)

    valid_rows = (
        cleaned_data["maturity"].notna()
        & (cleaned_data["maturity"] > 0.0)
        & cleaned_data["variance_swap_quote"].notna()
        & (cleaned_data["variance_swap_quote"] >= 0.0)
    )
    cleaned_data = cleaned_data.loc[valid_rows]
    cleaned_data = cleaned_data.sort_values(
        "maturity",
        kind="stable",
    ).reset_index(drop=True)

    if cleaned_data.empty:
        raise ValueError(
            "No valid variance-swap observation remains after cleaning."
        )

    check_variance_swap_data(cleaned_data)
    return cleaned_data


def check_variance_swap_arbitrage(variance_swap_data):
    """
    Check monotonicity of total variance and forward variance.

    Parameters
    ----------
    variance_swap_data : pandas.DataFrame
        Long-format, validated variance-swap term structure with columns
        maturity and variance_swap_quote. Maturities are in years and quotes
        are annualized decimal volatility quotes.

    Returns
    -------
    is_arbitrage_free : bool
        True when total variance is non-decreasing and every forward variance
        between consecutive maturities is non-negative.

    Raises
    ------
    TypeError
        If variance_swap_data is not a pandas.DataFrame.

    ValueError
        If the input fails check_variance_swap_data, if total variance is not
        non-decreasing, or if a forward variance is negative.

    Notes
    -----
    For each observation at maturity T_k, the annualized variance-swap
    volatility quote sigma_VS(T_k) implies total variance

        TV(T_k) = T_k * sigma_VS(T_k)^2.

    With strictly increasing maturities, the forward variance on an interval
    is

        (TV(T_{k+1}) - TV(T_k)) / (T_{k+1} - T_k).

    No tolerance or clipping is applied. A negative value is reported as an
    error rather than corrected silently.
    """
    check_variance_swap_data(variance_swap_data)

    maturities = variance_swap_data["maturity"].to_numpy(dtype=float)
    quotes = variance_swap_data[
        "variance_swap_quote"
    ].to_numpy(dtype=float)
    total_variance = maturities * quotes**2
    forward_variance = np.diff(total_variance) / np.diff(maturities)

    if np.any(np.diff(total_variance) < 0.0):
        raise ValueError(
            "Variance-swap total variance must be non-decreasing."
        )

    if np.any(forward_variance < 0.0):
        raise ValueError(
            "Variance-swap quotes imply negative forward variance."
        )

    return True
