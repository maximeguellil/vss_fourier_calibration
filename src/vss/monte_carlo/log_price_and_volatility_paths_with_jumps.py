"""Combine VSS diffusion paths with independent Gaussian log jumps."""

import numpy as np

from .log_price_and_volatility_paths import (
    simulate_log_prices_and_sigmas_batch,
)
from .simulate_gaussian_jumps import (
    simulate_common_idiosyncratic_gaussian_jump_log_returns,
    simulate_marginal_gaussian_jump_log_returns,
)


def _derived_jump_seed(seed):
    """Derive a reproducible jump seed without changing the diffusion seed."""
    if seed is None:
        return None
    sequence = np.random.SeedSequence(seed)
    return int(sequence.spawn(1)[0].generate_state(1, dtype=np.uint64)[0])


def simulate_log_prices_and_sigmas_with_jumps_batch(
    d,
    N,
    T,
    Y0,
    rho,
    MM,
    chol_V,
    mc_start,
    mc_stop,
    jump_time_boundaries,
    jump_g0_values,
    jump_intensities,
    jump_means,
    seed=None,
    *,
    common_jump_intensity=None,
    jump_size_correlation=None,
    use_antithetic=False,
    return_log_returns=True,
    return_sigmas=True,
    return_delta_B=False,
):
    """Simulate VSS diffusion paths and add independent Gaussian log jumps.

    Parameters
    ----------
    d, N, T, Y0, rho, MM, chol_V, mc_start, mc_stop
        Diffusion inputs passed unchanged to
        :func:`simulate_log_prices_and_sigmas_batch`.

    jump_time_boundaries, jump_g0_values
        For ``d=1``, one boundary array and one value array. For ``d=2``,
        sequences containing the two marginal piecewise-constant curves.

    jump_intensities, jump_means
        Scalars for ``d=1`` and arrays of shape ``(2,)`` for ``d=2``.

    seed : int or None, optional
        Seed used by the diffusion. A separate reproducible child seed is
        used for the independent jump component.

    common_jump_intensity, jump_size_correlation : float, optional
        Required for ``d=2``. They define the frequency and mark correlation
        of simultaneous jumps.

    use_antithetic, return_log_returns, return_sigmas, return_delta_B : bool
        Simulation and output options matching the diffusion simulator.

    Returns
    -------
    log_returns : ndarray or None
        Combined cumulative diffusion and jump log returns.

    sigmas : ndarray or None
        Diffusive instantaneous volatility matrices.

    delta_B : ndarray or None
        Brownian increments of the diffusive component.

    Raises
    ------
    ValueError
        If ``d`` is not one or two, or if the two-asset common-jump
        parameters are omitted.
    """
    if d not in {1, 2}:
        raise ValueError("The jump extension currently supports d equal to 1 or 2.")

    log_returns, sigmas, delta_B = simulate_log_prices_and_sigmas_batch(
        d=d,
        N=N,
        T=T,
        Y0=Y0,
        rho=rho,
        MM=MM,
        chol_V=chol_V,
        mc_start=mc_start,
        mc_stop=mc_stop,
        seed=seed,
        use_antithetic=use_antithetic,
        return_log_returns=return_log_returns,
        return_sigmas=return_sigmas,
        return_delta_B=return_delta_B,
    )
    if not return_log_returns:
        return log_returns, sigmas, delta_B

    jump_seed = _derived_jump_seed(seed)
    if d == 1:
        jump_log_returns = simulate_marginal_gaussian_jump_log_returns(
            N=N,
            T=T,
            time_boundaries=jump_time_boundaries,
            g0_values=jump_g0_values,
            jump_intensity=jump_intensities,
            jump_mean=jump_means,
            mc_start=mc_start,
            mc_stop=mc_stop,
            seed=jump_seed,
            use_antithetic=use_antithetic,
        )[None, :, :]
    else:
        if common_jump_intensity is None or jump_size_correlation is None:
            raise ValueError(
                "common_jump_intensity and jump_size_correlation are required "
                "when d=2."
            )
        jump_log_returns = (
            simulate_common_idiosyncratic_gaussian_jump_log_returns(
                N=N,
                T=T,
                time_boundaries=jump_time_boundaries,
                g0_values=jump_g0_values,
                jump_intensities=jump_intensities,
                jump_means=jump_means,
                common_jump_intensity=common_jump_intensity,
                jump_size_correlation=jump_size_correlation,
                mc_start=mc_start,
                mc_stop=mc_stop,
                seed=jump_seed,
                use_antithetic=use_antithetic,
            )
        )
    return log_returns + jump_log_returns, sigmas, delta_B
