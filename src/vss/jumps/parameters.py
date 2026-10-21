"""Compute and validate parameters of the Gaussian jump specification."""

import numpy as np


def gaussian_jump_normalization(jump_intensity, jump_mean):
    r"""Return the variance-preserving normalization for one asset.

    Parameters
    ----------
    jump_intensity : float
        Non-negative Poisson intensity in inverse years.

    jump_mean : float
        Mean of the Gaussian log-jump mark. Its variance is one.

    Returns
    -------
    alpha : float
        Normalization
        ``1 / sqrt(1 + jump_intensity * (1 + jump_mean**2))``.

    Raises
    ------
    ValueError
        If either argument is non-finite or if the intensity is negative.
    """
    try:
        jump_intensity = float(jump_intensity)
        jump_mean = float(jump_mean)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "jump_intensity and jump_mean must be scalar numeric values."
        ) from exc

    if not np.isfinite(jump_intensity) or jump_intensity < 0.0:
        raise ValueError("jump_intensity must be finite and non-negative.")
    if not np.isfinite(jump_mean):
        raise ValueError("jump_mean must be finite.")

    return float(
        1.0 / np.sqrt(1.0 + jump_intensity * (1.0 + jump_mean**2))
    )


def gaussian_jump_normalizations(jump_intensities, jump_means):
    r"""Return the variance-preserving normalizations of several assets.

    Parameters
    ----------
    jump_intensities : array_like, shape (d,)
        Non-negative marginal Poisson intensities in inverse years.

    jump_means : array_like, shape (d,)
        Means of the marginal Gaussian log-jump marks, whose variances are
        one.

    Returns
    -------
    alpha : ndarray, shape (d,)
        Elementwise values
        ``1 / sqrt(1 + jump_intensities * (1 + jump_means**2))``.

    Raises
    ------
    ValueError
        If the inputs are not finite one-dimensional arrays of the same
        non-zero length, or if an intensity is negative.
    """
    try:
        jump_intensities = np.asarray(jump_intensities, dtype=float)
        jump_means = np.asarray(jump_means, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "jump_intensities and jump_means must contain numeric values."
        ) from exc

    if jump_intensities.ndim != 1 or jump_means.ndim != 1:
        raise ValueError("jump_intensities and jump_means must be one-dimensional.")
    if jump_intensities.size == 0:
        raise ValueError("At least one asset is required.")
    if jump_intensities.shape != jump_means.shape:
        raise ValueError("jump_intensities and jump_means must have the same shape.")
    if not np.all(np.isfinite(jump_intensities)):
        raise ValueError("jump_intensities must be finite.")
    if np.any(jump_intensities < 0.0):
        raise ValueError("jump_intensities must be non-negative.")
    if not np.all(np.isfinite(jump_means)):
        raise ValueError("jump_means must be finite.")

    return 1.0 / np.sqrt(
        1.0 + jump_intensities * (1.0 + jump_means**2)
    )


def validate_common_idiosyncratic_jump_parameters(
    jump_intensities,
    jump_means,
    common_jump_intensity,
    jump_size_correlation,
):
    """Validate the two-asset common/idiosyncratic jump parameters.

    Parameters
    ----------
    jump_intensities : array_like, shape (2,)
        Marginal Poisson intensities ``(lambda_1, lambda_2)``.

    jump_means : array_like, shape (2,)
        Gaussian mark means ``(m_J^1, m_J^2)``.

    common_jump_intensity : float
        Common intensity ``lambda_c``, between zero and the smaller marginal
        intensity.

    jump_size_correlation : float
        Correlation ``rho_J`` of the two marks at a common jump time.

    Returns
    -------
    jump_intensities : ndarray, shape (2,)
    jump_means : ndarray, shape (2,)
    common_jump_intensity : float
    jump_size_correlation : float
        Validated numeric parameters.

    Raises
    ------
    ValueError
        If the marginal arrays are invalid, if the common intensity is
        outside its admissible interval, or if the correlation is outside
        ``[-1, 1]``.
    """
    _ = gaussian_jump_normalizations(jump_intensities, jump_means)
    jump_intensities = np.asarray(jump_intensities, dtype=float)
    jump_means = np.asarray(jump_means, dtype=float)
    if jump_intensities.shape != (2,):
        raise ValueError("jump_intensities and jump_means must have shape (2,).")

    try:
        common_jump_intensity = float(common_jump_intensity)
        jump_size_correlation = float(jump_size_correlation)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "common_jump_intensity and jump_size_correlation must be scalar."
        ) from exc

    if not np.isfinite(common_jump_intensity):
        raise ValueError("common_jump_intensity must be finite.")
    if (
        common_jump_intensity < 0.0
        or common_jump_intensity > np.min(jump_intensities)
    ):
        raise ValueError(
            "common_jump_intensity must lie between zero and the smaller "
            "marginal intensity."
        )
    if (
        not np.isfinite(jump_size_correlation)
        or abs(jump_size_correlation) > 1.0
    ):
        raise ValueError("jump_size_correlation must lie in [-1, 1].")

    return (
        jump_intensities,
        jump_means,
        common_jump_intensity,
        jump_size_correlation,
    )


def gaussian_jump_quadratic_variation_matrix(
    jump_intensities,
    jump_means,
    common_jump_intensity,
    jump_size_correlation,
):
    r"""Return the instantaneous second-moment matrix of two jump processes.

    The diagonal entries are ``lambda_i * (1 + m_i**2)``. The off-diagonal
    entry is ``lambda_c * (m_1*m_2 + rho_J)`` and therefore contains only
    simultaneous jumps.

    Parameters
    ----------
    jump_intensities, jump_means : array_like, shape (2,)
        Marginal Poisson intensities and Gaussian mark means.

    common_jump_intensity : float
        Intensity of simultaneous jumps.

    jump_size_correlation : float
        Correlation of the Gaussian marks at a simultaneous jump.

    Returns
    -------
    Q_J : ndarray, shape (2, 2)
        Matrix satisfying ``E[d[J]_t] = Q_J dt``.
    """
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

    diagonal = jump_intensities * (1.0 + jump_means**2)
    cross = common_jump_intensity * (
        jump_means[0] * jump_means[1] + jump_size_correlation
    )
    return np.array([[diagonal[0], cross], [cross, diagonal[1]]])
