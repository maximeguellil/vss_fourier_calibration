"""Calibration of the deterministic G_0 curve from variance-swap quotes."""

import numbers

import numpy as np


def compute_piecewise_constant_g0_from_variance_swaps(
    maturities,
    annualized_vol_quotes,
):
    r"""
    Construct a piecewise-constant deterministic input curve G0 from
    annualized variance-swap volatility quotes in order to match the
    variance-swap term structure.

    Parameters
    ----------
    maturities : array_like, shape (n_maturities,)
        Strictly increasing variance-swap maturities in years. At least one
        maturity is required. The first interval is the interval from zero
        to the first maturity.

    annualized_vol_quotes : array_like, shape (n_maturities,)
        Non-negative annualized volatility quotes for variance swaps,
        expressed as decimal volatilities (for example, 0.20 for 20%). There
        must be one quote for each entry of maturities.

    Returns
    -------
    time_boundaries : ndarray, shape (n_maturities + 1,)
        Boundaries of the calibration intervals in years. The first value is
        zero and the remaining values are the entries of maturities.

    G0_values : ndarray, shape (n_maturities,)
        Non-negative piecewise-constant values of G_0. On the interval
        [time_boundaries[k], time_boundaries[k + 1]), the curve satisfies
        G_0(t) = G0_values[k] (right continuous).

    Raises
    ------
    ValueError
        If the inputs are not one-dimensional arrays of equal non-zero
        length, if a maturity is non-finite, non-positive, or not strictly
        increasing, if a quote is non-finite or negative, or if the quotes
        imply a negative forward variance.

    Notes
    -----
    The input quote at maturity T_k is the annualized variance-swap
    volatility sigma_VS(T_k), not a volatility-swap quote. Its mathematical
    relation with the deterministic calibration curve is

        sigma_VS(T_k) = sqrt(
            E[(1 / T_k) * integral_0^T_k sigma(s, Y_s)^2 ds]
        ) = sqrt(
            (1 / T_k) * integral_0^T_k G0(s)^2 ds
        ).

    Maturities are measured in years, variance-swap quotes are annualized
    decimal volatilities, and their squares are annualized variances. No
    clipping or numerical tolerance is applied to a negative forward
    variance.
    """
    try:
        maturities = np.asarray(maturities, dtype=float)
        annualized_vol_quotes = np.asarray(annualized_vol_quotes, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "maturities and annualized_vol_quotes must contain numeric values."
        ) from exc

    if maturities.ndim != 1 or annualized_vol_quotes.ndim != 1:
        raise ValueError(
            "maturities and annualized_vol_quotes must be one-dimensional arrays."
        )

    if maturities.size == 0:
        raise ValueError("At least one variance-swap observation is required.")

    if maturities.size != annualized_vol_quotes.size:
        raise ValueError(
            "maturities and annualized_vol_quotes must have the same length."
        )

    if not np.all(np.isfinite(maturities)):
        raise ValueError("maturities must be finite.")

    if np.any(maturities <= 0.0):
        raise ValueError("maturities must be strictly positive.")

    if np.any(np.diff(maturities) <= 0.0):
        raise ValueError("maturities must be strictly increasing.")

    if not np.all(np.isfinite(annualized_vol_quotes)):
        raise ValueError("annualized_vol_quotes must be finite.")

    if np.any(annualized_vol_quotes < 0.0):
        raise ValueError("annualized_vol_quotes must be non-negative.")

    total_variance = maturities * annualized_vol_quotes**2
    time_boundaries = np.concatenate(([0.0], maturities))
    variance_boundaries = np.concatenate(([0.0], total_variance))
    forward_variance = np.diff(variance_boundaries) / np.diff(time_boundaries)

    if np.any(forward_variance < 0.0):
        raise ValueError(
            "The market quotes imply negative forward variance."
        )

    G0_values = np.sqrt(forward_variance)
    return time_boundaries, G0_values


def evaluate_piecewise_constant_g0_on_grid(
    T,
    N,
    time_boundaries,
    G0_values,
):
    r"""
    Evaluate a piecewise-constant G_0 curve on the left grid points
    t_i = i T / N for i = 0, ..., N - 1.

    Parameters
    ----------
    T : float
        Model maturity in years.

    N : int
        Positive number of model time steps. The grid points used for the
        evaluation are t_i = i T / N for i = 0, ..., N - 1.

    time_boundaries : array_like, shape (n_intervals + 1,)
        Increasing interval boundaries in years, beginning at zero. The
        curve is defined on interval [time_boundaries[k],
        time_boundaries[k + 1]).

    G0_values : array_like, shape (n_intervals,)
        Finite, non-negative value associated with each interval in
        time_boundaries.

    Returns
    -------
    G0_grid : ndarray, shape (N, 1, 1)
        Piecewise-constant values at the left endpoints of the model grid.
        At a boundary, the interval to its right is selected. After the
        last boundary, the last value in G0_values is used as a flat
        extrapolation.

    Raises
    ------
    ValueError
        If T is non-finite or negative, if N is not a positive
        integer, or if the boundary and value arrays do not describe a
        finite, increasing, non-empty piecewise-constant curve beginning at
        zero.

    Notes
    -----
    Interval selection uses np.searchsorted with side set to right
    - 1 followed by clipping to the available interval indices. Thus
    time_boundaries[k] belongs to interval k, except for the final boundary,
    which uses the flat extrapolation value.
    """
    try:
        T = float(T)
    except (TypeError, ValueError) as exc:
        raise ValueError("T must be a finite, non-negative scalar.") from exc

    if not np.isfinite(T) or T < 0.0:
        raise ValueError("T must be finite and non-negative.")

    if isinstance(N, bool) or not isinstance(N, numbers.Integral) or N <= 0:
        raise ValueError("N must be a positive integer.")

    try:
        time_boundaries = np.asarray(time_boundaries, dtype=float)
        G0_values = np.asarray(G0_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "time_boundaries and G0_values must contain numeric values."
        ) from exc

    if time_boundaries.ndim != 1 or G0_values.ndim != 1:
        raise ValueError(
            "time_boundaries and G0_values must be one-dimensional arrays."
        )

    if G0_values.size == 0:
        raise ValueError("G0_values must contain at least one interval value.")

    if time_boundaries.size != G0_values.size + 1:
        raise ValueError(
            "time_boundaries must have exactly one more entry than G0_values."
        )

    if not np.all(np.isfinite(time_boundaries)):
        raise ValueError("time_boundaries must be finite.")

    if time_boundaries[0] != 0.0:
        raise ValueError("time_boundaries must start at zero.")

    if np.any(np.diff(time_boundaries) <= 0.0):
        raise ValueError("time_boundaries must be strictly increasing.")

    if not np.all(np.isfinite(G0_values)):
        raise ValueError("G0_values must be finite.")

    if np.any(G0_values < 0.0):
        raise ValueError("G0_values must be non-negative.")

    t = np.arange(int(N), dtype=float) * T / int(N)
    interval_indices = (
        np.searchsorted(time_boundaries, t, side="right") - 1
    )
    interval_indices = np.clip(
        interval_indices,
        0,
        G0_values.size - 1,
    )

    return G0_values[interval_indices].reshape(int(N), 1, 1)


def integrated_piecewise_constant_g0_squared(
    T,
    time_boundaries,
    G0_values,
):
    r"""
    Integrate the squared piecewise-constant G_0 curve exactly.

    Parameters
    ----------
    T : float
        Upper integration limit in years.

    time_boundaries : array_like, shape (n_intervals + 1,)
        Increasing interval boundaries in years, beginning at zero. The
        curve is defined on interval [time_boundaries[k],
        time_boundaries[k + 1]).

    G0_values : array_like, shape (n_intervals,)
        Finite, non-negative value associated with each interval in
        time_boundaries.

    Returns
    -------
    expected_integrated_variance : float
        Exact value of integral_0^T G_0(t)^2 dt as a float. The
        overlap of each calibration interval with [0, T] is multiplied by
        its squared curve value. Beyond the last boundary, the last value in
        G0_values is extrapolated flat.

    Raises
    ------
    ValueError
        If T is non-finite or negative, or if the boundary and value
        arrays do not describe a finite, increasing, non-empty
        piecewise-constant curve beginning at zero.

    Notes
    -----
    The integration is performed from interval intersections, not from a
    model time grid, so it is exact for the supplied piecewise-constant
    curve. No numerical tolerance or clipping is applied.
    """
    try:
        T = float(T)
    except (TypeError, ValueError) as exc:
        raise ValueError("T must be a finite, non-negative scalar.") from exc

    if not np.isfinite(T) or T < 0.0:
        raise ValueError("T must be finite and non-negative.")

    try:
        time_boundaries = np.asarray(time_boundaries, dtype=float)
        G0_values = np.asarray(G0_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "time_boundaries and G0_values must contain numeric values."
        ) from exc

    if time_boundaries.ndim != 1 or G0_values.ndim != 1:
        raise ValueError(
            "time_boundaries and G0_values must be one-dimensional arrays."
        )

    if G0_values.size == 0:
        raise ValueError("G0_values must contain at least one interval value.")

    if time_boundaries.size != G0_values.size + 1:
        raise ValueError(
            "time_boundaries must have exactly one more entry than G0_values."
        )

    if not np.all(np.isfinite(time_boundaries)):
        raise ValueError("time_boundaries must be finite.")

    if time_boundaries[0] != 0.0:
        raise ValueError("time_boundaries must start at zero.")

    if np.any(np.diff(time_boundaries) <= 0.0):
        raise ValueError("time_boundaries must be strictly increasing.")

    if not np.all(np.isfinite(G0_values)):
        raise ValueError("G0_values must be finite.")

    if np.any(G0_values < 0.0):
        raise ValueError("G0_values must be non-negative.")

    if T == 0.0:
        return 0.0

    interval_starts = time_boundaries[:-1]
    interval_ends = time_boundaries[1:]
    overlap_lengths = np.maximum(
        np.minimum(interval_ends, T) - interval_starts,
        0.0,
    )
    integrated_variance = np.sum(overlap_lengths * G0_values**2)

    # Flat extrapolation after the last available variance-swap maturity.
    last_maturity = time_boundaries[-1]
    if T > last_maturity:
        integrated_variance += (T - last_maturity) * G0_values[-1] ** 2

    return float(integrated_variance)
