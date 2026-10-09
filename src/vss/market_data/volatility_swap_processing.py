"""Cleaning and validation of volatility-swap market data."""

import numpy as np
import pandas as pd

from .variance_swap_processing import check_variance_swap_data


VOLATILITY_SWAP_COLUMNS = [
    "maturity",
    "volatility_swap_quote",
]


def check_volatility_swap_data(volatility_swap_data):
    """
    Validate the structure and domain of a volatility-swap term structure.

    Parameters
    ----------
    volatility_swap_data : pandas.DataFrame
        Long-format table with columns maturity and volatility_swap_quote.
        Maturities are expressed in years and quotes are annualized decimal
        volatility-swap quotes.

    Returns
    -------
    is_valid : bool
        True when the table satisfies all structural and value checks.

    Raises
    ------
    TypeError
        If volatility_swap_data is not a pandas.DataFrame.

    ValueError
        If a required column is missing, if the table is empty, if maturities
        or quotes are non-numeric, non-finite, or out of domain, if maturity
        is duplicated, or if maturities are not strictly increasing.

    Notes
    -----
    This function checks the observations themselves. Consistency with
    variance swaps is checked separately by
    check_volatility_swap_consistency.
    """
    if not isinstance(volatility_swap_data, pd.DataFrame):
        raise TypeError("volatility_swap_data must be a pandas.DataFrame.")

    missing_columns = [
        column
        for column in VOLATILITY_SWAP_COLUMNS
        if column not in volatility_swap_data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required volatility-swap columns: {missing_columns}."
        )

    if volatility_swap_data.columns.has_duplicates:
        raise ValueError(
            "Volatility-swap column names must not be duplicated."
        )

    if volatility_swap_data.empty:
        raise ValueError("volatility_swap_data must not be empty.")

    try:
        maturities = volatility_swap_data["maturity"].to_numpy(dtype=float)
        quotes = volatility_swap_data[
            "volatility_swap_quote"
        ].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Maturities and volatility-swap quotes must be numeric."
        ) from exc

    if not np.all(np.isfinite(maturities)) or np.any(maturities <= 0.0):
        raise ValueError(
            "Maturities must be finite and strictly positive."
        )

    if np.any(np.diff(maturities) <= 0.0):
        raise ValueError("Maturities must be strictly increasing.")

    if not np.all(np.isfinite(quotes)) or np.any(quotes < 0.0):
        raise ValueError(
            "Volatility-swap quotes must be finite and non-negative."
        )

    if volatility_swap_data["maturity"].duplicated().any():
        raise ValueError("Volatility-swap maturities must not be duplicated.")

    return True


def clean_volatility_swap_data(volatility_swap_data):
    """
    Clean and order a volatility-swap term structure.

    Parameters
    ----------
    volatility_swap_data : pandas.DataFrame
        Long-format table with columns maturity and volatility_swap_quote.
        Maturities are expressed in years and quotes are annualized decimal
        volatility-swap quotes.

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
        If volatility_swap_data is not a pandas.DataFrame.

    ValueError
        If required columns are missing, if maturity labels are duplicated,
        or if no valid observation remains after cleaning.

    Notes
    -----
    Cleaning does not alter a quote and does not check consistency with
    variance swaps. No observation is removed because of a consistency
    violation.
    """
    if not isinstance(volatility_swap_data, pd.DataFrame):
        raise TypeError("volatility_swap_data must be a pandas.DataFrame.")

    missing_columns = [
        column
        for column in VOLATILITY_SWAP_COLUMNS
        if column not in volatility_swap_data.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required volatility-swap columns: {missing_columns}."
        )

    if volatility_swap_data.columns.has_duplicates:
        raise ValueError(
            "Volatility-swap column names must not be duplicated."
        )

    if volatility_swap_data["maturity"].duplicated().any():
        raise ValueError("Volatility-swap maturities must not be duplicated.")

    cleaned_data = volatility_swap_data.loc[
        :,
        VOLATILITY_SWAP_COLUMNS,
    ].copy()
    cleaned_data = cleaned_data.apply(pd.to_numeric, errors="coerce")
    cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)

    valid_rows = (
        cleaned_data["maturity"].notna()
        & (cleaned_data["maturity"] > 0.0)
        & cleaned_data["volatility_swap_quote"].notna()
        & (cleaned_data["volatility_swap_quote"] >= 0.0)
    )
    cleaned_data = cleaned_data.loc[valid_rows]
    cleaned_data = cleaned_data.sort_values(
        "maturity",
        kind="stable",
    ).reset_index(drop=True)

    if cleaned_data.empty:
        raise ValueError(
            "No valid volatility-swap observation remains after cleaning."
        )

    check_volatility_swap_data(cleaned_data)
    return cleaned_data


def check_volatility_swap_consistency(
    volatility_swap_data,
    variance_swap_data,
):
    """
    Check annualized volatility-swap consistency with variance swaps.

    Parameters
    ----------
    volatility_swap_data : pandas.DataFrame
        Long-format volatility-swap table with columns maturity and
        volatility_swap_quote. The quote is the annualized quantity
        E[sqrt(V_T)] / sqrt(T).

    variance_swap_data : pandas.DataFrame
        Long-format variance-swap table with columns maturity and
        variance_swap_quote. The quote is the annualized quantity
        sqrt(E[V_T] / T).

    Returns
    -------
    is_consistent : bool
        True when the volatility-swap quote is less than or equal to the
        variance-swap quote at every exactly matching maturity.

    Raises
    ------
    TypeError
        If either input is not a pandas.DataFrame.

    ValueError
        If either input fails its data checks, if no maturity is shared by
        the two tables, or if a volatility-swap quote is greater than the
        corresponding variance-swap quote.

    Notes
    -----
    Let V_T be the integrated variance. With the annualization conventions
    used in this project, Jensen's inequality gives

        E[sqrt(V_T)] / sqrt(T) <= sqrt(E[V_T] / T).

    The comparison is made only at exactly matching maturity values. No
    interpolation, tolerance, or correction is applied.
    """
    if not isinstance(volatility_swap_data, pd.DataFrame):
        raise TypeError("volatility_swap_data must be a pandas.DataFrame.")
    if not isinstance(variance_swap_data, pd.DataFrame):
        raise TypeError("variance_swap_data must be a pandas.DataFrame.")

    check_volatility_swap_data(volatility_swap_data)
    check_variance_swap_data(variance_swap_data)

    common_data = volatility_swap_data[
        VOLATILITY_SWAP_COLUMNS
    ].merge(
        variance_swap_data[
            ["maturity", "variance_swap_quote"]
        ],
        on="maturity",
        how="inner",
        validate="one_to_one",
    )

    if common_data.empty:
        raise ValueError(
            "The volatility-swap and variance-swap data have no common "
            "maturity."
        )

    inconsistent = (
        common_data["volatility_swap_quote"]
        > common_data["variance_swap_quote"]
    )
    if inconsistent.any():
        maturities = common_data.loc[inconsistent, "maturity"].tolist()
        raise ValueError(
            "Volatility-swap quotes exceed variance-swap quotes at maturities "
            f"{maturities}."
        )

    return True
