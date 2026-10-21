"""Simulation of compensated Gaussian compound-Poisson log jumps."""

import numbers

import numpy as np

from vss.jumps.parameters import (
    gaussian_jump_normalization,
    gaussian_jump_normalizations,
    validate_common_idiosyncratic_jump_parameters,
)


def _validate_simulation_inputs(N, T, mc_start, mc_stop, use_antithetic):
    """Validate the common time-grid and batch arguments."""
    if isinstance(N, bool) or not isinstance(N, numbers.Integral) or N <= 0:
        raise ValueError("N must be a positive integer.")
    try:
        T = float(T)
    except (TypeError, ValueError) as exc:
        raise ValueError("T must be a finite, non-negative scalar.") from exc
    if not np.isfinite(T) or T < 0.0:
        raise ValueError("T must be finite and non-negative.")
    if mc_stop <= mc_start:
        raise ValueError("mc_stop must be greater than mc_start.")
    batch_mc = int(mc_stop - mc_start)
    if use_antithetic and batch_mc % 2:
        raise ValueError("The batch size must be even with antithetic sampling.")
    return int(N), T, batch_mc


def _validate_curve(time_boundaries, g0_values):
    """Validate one piecewise-constant total loading curve."""
    try:
        time_boundaries = np.asarray(time_boundaries, dtype=float)
        g0_values = np.asarray(g0_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "time_boundaries and g0_values must contain numeric values."
        ) from exc
    if time_boundaries.ndim != 1 or g0_values.ndim != 1:
        raise ValueError("time_boundaries and g0_values must be one-dimensional.")
    if g0_values.size == 0 or time_boundaries.size != g0_values.size + 1:
        raise ValueError(
            "time_boundaries must have exactly one more entry than non-empty "
            "g0_values."
        )
    if (
        not np.all(np.isfinite(time_boundaries))
        or time_boundaries[0] != 0.0
        or np.any(np.diff(time_boundaries) <= 0.0)
    ):
        raise ValueError(
            "time_boundaries must be finite, start at zero, and be strictly "
            "increasing."
        )
    if not np.all(np.isfinite(g0_values)) or np.any(g0_values < 0.0):
        raise ValueError("g0_values must be finite and non-negative.")
    return time_boundaries, g0_values


def _grid_aligned_segments(N, T, time_boundaries, g0_values):
    """Split the simulation grid at every curve boundary."""
    curves = [
        _validate_curve(boundaries, values)
        for boundaries, values in zip(time_boundaries, g0_values)
    ]
    if T == 0.0:
        return []

    grid = np.linspace(0.0, T, N + 1)
    internal = [
        boundaries[(boundaries > 0.0) & (boundaries < T)]
        for boundaries, _ in curves
    ]
    boundaries = np.unique(np.concatenate((grid, *internal)))
    segments = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        model_index = min(int(np.searchsorted(grid, left, side="right") - 1), N - 1)
        values = []
        for curve_boundaries, curve_values in curves:
            curve_index = np.searchsorted(
                curve_boundaries,
                left,
                side="right",
            ) - 1
            curve_index = int(np.clip(curve_index, 0, curve_values.size - 1))
            values.append(curve_values[curve_index])
        segments.append((model_index, right - left, np.asarray(values)))
    return segments


def _normal_samples(rng, size, use_antithetic):
    """Generate standard normals with optional centered antithetic pairing."""
    if not use_antithetic:
        return rng.standard_normal(size=size)
    half = size[-1] // 2
    first = rng.standard_normal(size=(*size[:-1], half))
    return np.concatenate((first, -first), axis=-1)


def _poisson_samples(rng, rate, batch_mc, use_antithetic):
    """Generate counts shared inside each antithetic pair."""
    if not use_antithetic:
        return rng.poisson(rate, size=batch_mc)
    half = batch_mc // 2
    first = rng.poisson(rate, size=half)
    return np.concatenate((first, first))


def simulate_marginal_gaussian_jump_log_returns(
    N,
    T,
    time_boundaries,
    g0_values,
    jump_intensity,
    jump_mean,
    mc_start,
    mc_stop,
    seed=None,
    *,
    use_antithetic=False,
):
    r"""Simulate one compensated Gaussian jump log-return process.

    Parameters
    ----------
    N : int
        Number of uniform output time steps on ``[0, T]``.

    T : float
        Maturity in years.

    time_boundaries : array_like, shape (n_intervals + 1,)
        Boundaries of the piecewise-constant total curve ``g_0``.

    g0_values : array_like, shape (n_intervals,)
        Values of ``g_0``. The last value is extrapolated flat.

    jump_intensity : float
        Poisson intensity in inverse years.

    jump_mean : float
        Mean of the Gaussian mark, whose variance is one.

    mc_start, mc_stop : int
        Indices defining the simulated batch.

    seed : int or None, optional
        Local random seed.

    use_antithetic : bool, optional
        If true, each pair shares its Poisson counts and uses opposite
        centered Gaussian marks.

    Returns
    -------
    log_returns : ndarray, shape (N, batch_mc)
        Cumulative compensated jump log returns at the output grid points.

    Notes
    -----
    Every output interval is split at boundaries of ``g_0``. Counts and
    Gaussian sums are therefore simulated with the exact loading on each
    subinterval.
    """
    N, T, batch_mc = _validate_simulation_inputs(
        N,
        T,
        mc_start,
        mc_stop,
        use_antithetic,
    )
    alpha = gaussian_jump_normalization(jump_intensity, jump_mean)
    jump_intensity = float(jump_intensity)
    jump_mean = float(jump_mean)
    segments = _grid_aligned_segments(
        N,
        T,
        [time_boundaries],
        [g0_values],
    )
    rng = np.random.default_rng(seed)
    increments = np.zeros((N, batch_mc))

    for model_index, duration, values in segments:
        h = alpha * values[0]
        counts = _poisson_samples(
            rng,
            jump_intensity * duration,
            batch_mc,
            use_antithetic,
        )
        normals = _normal_samples(rng, (batch_mc,), use_antithetic)
        jump_sum = counts * jump_mean + np.sqrt(counts) * normals
        compensator = duration * jump_intensity * np.expm1(
            h * jump_mean + 0.5 * h**2
        )
        increments[model_index] += h * jump_sum - compensator

    return np.cumsum(increments, axis=0)


def simulate_common_idiosyncratic_gaussian_jump_log_returns(
    N,
    T,
    time_boundaries,
    g0_values,
    jump_intensities,
    jump_means,
    common_jump_intensity,
    jump_size_correlation,
    mc_start,
    mc_stop,
    seed=None,
    *,
    use_antithetic=False,
):
    r"""Simulate two compensated common/idiosyncratic jump processes.

    Parameters
    ----------
    N, T : int, float
        Number of uniform output steps and maturity in years.

    time_boundaries : sequence of two array_like
        Boundaries of the two piecewise-constant total curves.

    g0_values : sequence of two array_like
        Values of the two total curves ``g_0^i``.

    jump_intensities, jump_means : array_like, shape (2,)
        Marginal Poisson intensities and Gaussian mark means.

    common_jump_intensity : float
        Intensity of simultaneous jumps.

    jump_size_correlation : float
        Correlation of the two common Gaussian marks.

    mc_start, mc_stop : int
        Indices defining the simulated batch.

    seed : int or None, optional
        Local random seed.

    use_antithetic : bool, optional
        If true, each pair shares all counts and uses opposite centered
        Gaussian marks.

    Returns
    -------
    log_returns : ndarray, shape (2, N, batch_mc)
        Cumulative compensated jump log returns at the output grid points.
    """
    N, T, batch_mc = _validate_simulation_inputs(
        N,
        T,
        mc_start,
        mc_stop,
        use_antithetic,
    )
    if len(time_boundaries) != 2 or len(g0_values) != 2:
        raise ValueError("time_boundaries and g0_values must each contain two curves.")
    (
        jump_intensities,
        jump_means,
        common_jump_intensity,
        jump_size_correlation,
    ) = validate_common_idiosyncratic_jump_parameters(
        jump_intensities,
        jump_means,
        common_jump_intensity,
        jump_size_correlation,
    )
    alpha = gaussian_jump_normalizations(jump_intensities, jump_means)
    idiosyncratic_intensities = jump_intensities - common_jump_intensity
    segments = _grid_aligned_segments(
        N,
        T,
        time_boundaries,
        g0_values,
    )
    rng = np.random.default_rng(seed)
    increments = np.zeros((2, N, batch_mc))
    rho_complement = np.sqrt(max(0.0, 1.0 - jump_size_correlation**2))

    for model_index, duration, values in segments:
        h = alpha * values
        common_counts = _poisson_samples(
            rng,
            common_jump_intensity * duration,
            batch_mc,
            use_antithetic,
        )
        common_normals = _normal_samples(
            rng,
            (2, batch_mc),
            use_antithetic,
        )
        correlated_common_normals = np.vstack((
            common_normals[0],
            jump_size_correlation * common_normals[0]
            + rho_complement * common_normals[1],
        ))
        common_sums = (
            jump_means[:, None] * common_counts
            + np.sqrt(common_counts)[None, :] * correlated_common_normals
        )

        idiosyncratic_sums = np.empty((2, batch_mc))
        for asset in range(2):
            counts = _poisson_samples(
                rng,
                idiosyncratic_intensities[asset] * duration,
                batch_mc,
                use_antithetic,
            )
            normals = _normal_samples(
                rng,
                (batch_mc,),
                use_antithetic,
            )
            idiosyncratic_sums[asset] = (
                counts * jump_means[asset] + np.sqrt(counts) * normals
            )

        compensator = duration * jump_intensities * np.expm1(
            h * jump_means + 0.5 * h**2
        )
        increments[:, model_index] += (
            h[:, None] * (common_sums + idiosyncratic_sums)
            - compensator[:, None]
        )

    return np.cumsum(increments, axis=1)
