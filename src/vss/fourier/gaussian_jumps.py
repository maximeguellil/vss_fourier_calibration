"""Fourier--Laplace transforms of compensated Gaussian jumps."""

import numpy as np

from vss.jumps.parameters import (
    gaussian_jump_normalization,
    gaussian_jump_normalizations,
    validate_common_idiosyncratic_jump_parameters,
)


def _validate_piecewise_constant_curve(time_boundaries, g0_values):
    """Validate and return one piecewise-constant non-negative curve."""
    try:
        time_boundaries = np.asarray(time_boundaries, dtype=float)
        g0_values = np.asarray(g0_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "time_boundaries and g0_values must contain numeric values."
        ) from exc

    if time_boundaries.ndim != 1 or g0_values.ndim != 1:
        raise ValueError("time_boundaries and g0_values must be one-dimensional.")
    if g0_values.size == 0:
        raise ValueError("g0_values must contain at least one value.")
    if time_boundaries.size != g0_values.size + 1:
        raise ValueError(
            "time_boundaries must have exactly one more entry than g0_values."
        )
    if not np.all(np.isfinite(time_boundaries)):
        raise ValueError("time_boundaries must be finite.")
    if time_boundaries[0] != 0.0:
        raise ValueError("time_boundaries must start at zero.")
    if np.any(np.diff(time_boundaries) <= 0.0):
        raise ValueError("time_boundaries must be strictly increasing.")
    if not np.all(np.isfinite(g0_values)) or np.any(g0_values < 0.0):
        raise ValueError("g0_values must be finite and non-negative.")
    return time_boundaries, g0_values


def _piecewise_constant_segments_up_to_maturity(
    T,
    time_boundaries,
    g0_values,
):
    """Return exact segment lengths and curve values on ``[0, T]``."""
    try:
        T = float(T)
    except (TypeError, ValueError) as exc:
        raise ValueError("T must be a finite, non-negative scalar.") from exc
    if not np.isfinite(T) or T < 0.0:
        raise ValueError("T must be finite and non-negative.")

    time_boundaries, g0_values = _validate_piecewise_constant_curve(
        time_boundaries,
        g0_values,
    )
    if T == 0.0:
        return np.empty(0), np.empty(0)

    internal = time_boundaries[(time_boundaries > 0.0) & (time_boundaries < T)]
    segment_boundaries = np.concatenate(([0.0], internal, [T]))
    left_endpoints = segment_boundaries[:-1]
    indices = np.searchsorted(
        time_boundaries,
        left_endpoints,
        side="right",
    ) - 1
    indices = np.clip(indices, 0, g0_values.size - 1)
    return np.diff(segment_boundaries), g0_values[indices]


def _joint_piecewise_constant_segments_up_to_maturity(
    T,
    time_boundaries,
    g0_values,
):
    """Align two piecewise-constant curves on their union of boundaries."""
    if len(time_boundaries) != 2 or len(g0_values) != 2:
        raise ValueError("time_boundaries and g0_values must each contain two curves.")

    validated = [
        _validate_piecewise_constant_curve(boundaries, values)
        for boundaries, values in zip(time_boundaries, g0_values)
    ]
    try:
        T = float(T)
    except (TypeError, ValueError) as exc:
        raise ValueError("T must be a finite, non-negative scalar.") from exc
    if not np.isfinite(T) or T < 0.0:
        raise ValueError("T must be finite and non-negative.")
    if T == 0.0:
        return np.empty(0), np.empty((0, 2))

    internal_boundaries = [
        boundaries[(boundaries > 0.0) & (boundaries < T)]
        for boundaries, _ in validated
    ]
    segment_boundaries = np.unique(
        np.concatenate(([0.0], *internal_boundaries, [T]))
    )
    left_endpoints = segment_boundaries[:-1]
    aligned_values = np.empty((left_endpoints.size, 2))
    for asset, (boundaries, values) in enumerate(validated):
        indices = np.searchsorted(boundaries, left_endpoints, side="right") - 1
        indices = np.clip(indices, 0, values.size - 1)
        aligned_values[:, asset] = values[indices]
    return np.diff(segment_boundaries), aligned_values


def marginal_gaussian_jump_transform(
    u,
    T,
    time_boundaries,
    g0_values,
    jump_intensity,
    jump_mean,
):
    r"""Evaluate the transform of one compensated Gaussian jump component.

    Parameters
    ----------
    u : array_like of complex
        Fourier--Laplace arguments. The returned array has the same shape.

    T : float
        Maturity in years.

    time_boundaries : array_like, shape (n_intervals + 1,)
        Boundaries of the piecewise-constant total variance curve, beginning
        at zero.

    g0_values : array_like, shape (n_intervals,)
        Values of the total loading ``g_0``. The last value is extrapolated
        flat if ``T`` exceeds the last boundary.

    jump_intensity : float
        Marginal Poisson intensity ``lambda``.

    jump_mean : float
        Mean ``m_J`` of a Gaussian mark with unit variance.

    Returns
    -------
    transform : ndarray or complex scalar
        Values of ``E[exp(u * L_T^J)]`` for the compensated jump log-return.

    Notes
    -----
    The loading on each interval is ``h = alpha * g_0``, with ``alpha``
    computed by :func:`vss.jumps.parameters.gaussian_jump_normalization`.
    The time integral is evaluated exactly over the supplied intervals.
    """
    try:
        u = np.asarray(u, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError("u must contain numeric values.") from exc
    if not np.all(np.isfinite(u)):
        raise ValueError("u must be finite.")

    alpha = gaussian_jump_normalization(jump_intensity, jump_mean)
    jump_intensity = float(jump_intensity)
    jump_mean = float(jump_mean)
    durations, values = _piecewise_constant_segments_up_to_maturity(
        T,
        time_boundaries,
        g0_values,
    )

    exponent = np.zeros_like(u, dtype=complex)
    for duration, g0 in zip(durations, values):
        h = alpha * g0
        mark_transform = np.exp(u * h * jump_mean + 0.5 * (u * h) ** 2)
        martingale_transform = np.exp(h * jump_mean + 0.5 * h**2)
        exponent += duration * jump_intensity * (
            mark_transform - 1.0 - u * (martingale_transform - 1.0)
        )
    return np.exp(exponent)


def joint_gaussian_jump_transform(
    u,
    T,
    time_boundaries,
    g0_values,
    jump_intensities,
    jump_means,
    common_jump_intensity,
    jump_size_correlation,
):
    r"""Evaluate the joint transform of two compensated jump components.

    Parameters
    ----------
    u : array_like of complex, shape (..., 2)
        Joint Fourier--Laplace arguments.

    T : float
        Maturity in years.

    time_boundaries : sequence of two array_like
        Piecewise-constant curve boundaries for the two assets.

    g0_values : sequence of two array_like
        Total ``g_0`` curve values for the two assets.

    jump_intensities, jump_means : array_like, shape (2,)
        Marginal Poisson intensities and Gaussian mark means.

    common_jump_intensity : float
        Intensity ``lambda_c`` of simultaneous jumps.

    jump_size_correlation : float
        Correlation ``rho_J`` between the two common Gaussian marks.

    Returns
    -------
    transform : ndarray or complex scalar, shape u.shape[:-1]
        Values of the joint compensated jump transform.

    Notes
    -----
    The two curves are aligned on the union of their boundaries, and the
    time integral is evaluated exactly on the resulting intervals.
    """
    try:
        u = np.asarray(u, dtype=complex)
    except (TypeError, ValueError) as exc:
        raise ValueError("u must contain numeric values.") from exc
    if u.ndim == 0 or u.shape[-1] != 2:
        raise ValueError("u must have shape (..., 2).")
    if not np.all(np.isfinite(u)):
        raise ValueError("u must be finite.")

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
    durations, values = _joint_piecewise_constant_segments_up_to_maturity(
        T,
        time_boundaries,
        g0_values,
    )

    exponent = np.zeros(u.shape[:-1], dtype=complex)
    idiosyncratic_intensities = jump_intensities - common_jump_intensity
    for duration, g0 in zip(durations, values):
        h = alpha * g0
        uh = u * h
        marginal_mark_transforms = np.exp(
            uh * jump_means + 0.5 * uh**2
        )
        common_mark_transform = np.exp(
            np.sum(uh * jump_means, axis=-1)
            + 0.5
            * (
                uh[..., 0] ** 2
                + uh[..., 1] ** 2
                + 2.0
                * jump_size_correlation
                * uh[..., 0]
                * uh[..., 1]
            )
        )
        martingale_transforms = np.exp(
            h * jump_means + 0.5 * h**2
        )
        exponent += duration * (
            common_jump_intensity * (common_mark_transform - 1.0)
            + np.sum(
                idiosyncratic_intensities
                * (marginal_mark_transforms - 1.0),
                axis=-1,
            )
            - np.sum(
                u
                * jump_intensities
                * (martingale_transforms - 1.0),
                axis=-1,
            )
        )
    return np.exp(exponent)
