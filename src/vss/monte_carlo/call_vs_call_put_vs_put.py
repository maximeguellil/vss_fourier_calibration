import time

import numpy as np
from scipy.stats import norm

from .log_price_and_volatility_paths import (
    simulate_log_prices_and_sigmas_batch,
)
from .log_price_and_volatility_paths_with_jumps import (
    simulate_log_prices_and_sigmas_with_jumps_batch,
)


def mc_call_vs_call_put_vs_put_price(
    S_0,
    K_array,
    d: int,
    N: int,
    T: float,
    Y0: np.ndarray,
    rho: np.ndarray,
    MM: np.ndarray,
    chol_V: np.ndarray,
    MC: int,
    *,
    batch_mc: int = 1024,
    seed: int | None = None,
    use_antithetic: bool = False,
    option_type: str,
    return_std_error: bool = True,
    relative_half_width_target: float | None = None,
    scale_floor: float = 1e-8,
    minimum_batches: int = 3,
    stability_batches: int = 3,
    stability_tolerance: float = 0.02,
    confidence_level: float = 0.95,
    max_seconds: float | None = None,
    return_diagnostics: bool = False,
    jump_time_boundaries=None,
    jump_g0_values=None,
    jump_intensities=None,
    jump_means=None,
    common_jump_intensity=None,
    jump_size_correlation=None,
):
    """
    Monte Carlo estimator of call-vs-call and/or put-vs-put prices under the Quadratic Gaussian
    volatility model, for a vector of strikes (zero interest rates, no dividends).

    The diffusion uses chol_V, a factor of the covariance matrix V (see
    compute_V_matrix_for_mc), and a left-point Euler scheme for log-prices.
    When jump parameters are supplied, independent compensated Gaussian jump
    log-returns are added batch by batch. With r = 0 the prices are the plain
    expected payoffs
        call_vs_call(K)
            = E[ ((S_T^1 - K)^+ + (S_T^2 - K)^+) / 2 - (B_T - K)^+ ],
        put_vs_put(K)
            = E[ ((K - S_T^1)^+ + (K - S_T^2)^+) / 2 - (K - B_T)^+ ],
        B_T = (S_T^1 + S_T^2) / 2,
        S_T^i = S_0^i * exp(log(S_T^i/S_0^i)).

    The two payoffs are equal pathwise when the individual options and the basket option have the
    same strike and maturity. They are computed directly from the same simulated trajectories.

    Parameters
    ----------
    S_0 : 1D array_like of float, shape (2,)
        Initial asset prices [S_0^1, S_0^2].

    K_array : 1D array_like of float
        Common strike prices for the individual and basket options.

    d, N, T : int, int, float
        Model / discretization parameters (d = 2 here).

    Y0 : array_like, shape (n,)
        Initial volatility factor Y_0.

    rho : ndarray, shape (d, d)
        Correlation matrix of B, with d<B>_t = rho dt.

    MM : ndarray, shape (N*d^2, N*n)
        Block-diagonal matrix M.

    chol_V : ndarray, shape (N*(d*d + d), r)
        Cholesky factor of V (from compute_V_matrix_for_mc + spectral_cholesky).

    MC : int
        Total number of Monte Carlo trajectories.

    batch_mc : int, optional
        Number of trajectories per batch. If use_antithetic=True, batch_mc must be even.

    seed : int or None, optional
        Base seed. Each batch uses a derived seed for reproducibility.

    use_antithetic : bool, optional
        If True, each batch uses antithetic normals (Z, -Z). Standard errors are then computed
        from the independent pair averages.

    option_type : {"call_vs_call", "put_vs_put", "both"}
        Which prices to return.

    return_std_error : bool, optional
        If True, also return the Monte Carlo standard-error path(s).

    relative_half_width_target : float or None, optional
        If provided, stop between batches once every requested price has a
        confidence half-width below this fraction of its estimated scale and
        the latest cumulative estimates are stable.

    max_seconds : float or None, optional
        Wall-clock budget checked between completed batches.

    return_diagnostics : bool, optional
        If True, append a dictionary containing effective sample counts,
        elapsed times, relative confidence widths, and the stopping status to
        the usual return tuple.  The default False preserves the historical
        return signature.

    jump_time_boundaries, jump_g0_values : sequence, optional
        Boundaries and values of the two marginal piecewise-constant total
        curves. Leave both unset for the original diffusion-only simulation.

    jump_intensities, jump_means : array_like, shape (2,), optional
        Marginal Poisson intensities and Gaussian mark means.

    common_jump_intensity, jump_size_correlation : float, optional
        Intensity and mark correlation of simultaneous jumps. All six jump
        arguments must be supplied together.

    Returns
    -------
    MC_path : ndarray, shape (nbatches,)
        Cumulative number of simulated trajectories after each batch.

    price_path : ndarray, shape (nbatches, n_strikes)
        Cumulative price estimate after each batch (put-vs-put or call-vs-call).

    std_path : ndarray, shape (nbatches, n_strikes), optional
        Cumulative standard error after each batch (if return_std_error).

    For option_type == "both", the return is
        (MC_path, put_vs_put_path, call_vs_call_path[, put_vs_put_std, call_vs_call_std]).
    """
    if option_type not in {"call_vs_call", "put_vs_put", "both"}:
        raise ValueError(
            "option_type must be one of "
            "{'call_vs_call', 'put_vs_put', 'both'}"
        )

    S_0 = np.asarray(S_0, dtype=float)
    K_array = np.asarray(K_array, dtype=float).reshape(-1)
    nK = K_array.size

    if d != 2:
        raise ValueError("d must be equal to 2")
    if S_0.shape != (2,):
        raise ValueError(f"S_0 must have shape (2,), got {S_0.shape}")
    if MC <= 0:
        raise ValueError("MC must be positive")
    if batch_mc <= 0:
        raise ValueError("batch_mc must be positive")
    if use_antithetic and (batch_mc % 2 != 0):
        raise ValueError("batch_mc must be even when use_antithetic=True")
    if relative_half_width_target is not None:
        if relative_half_width_target <= 0.0:
            raise ValueError("relative_half_width_target must be positive")
        if not return_std_error:
            raise ValueError(
                "return_std_error must be True when a precision target is used"
            )
    if scale_floor <= 0.0:
        raise ValueError("scale_floor must be positive")
    if minimum_batches <= 0 or stability_batches < 2:
        raise ValueError(
            "minimum_batches must be positive and stability_batches must be at least 2"
        )
    if stability_tolerance <= 0.0:
        raise ValueError("stability_tolerance must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if max_seconds is not None and max_seconds <= 0.0:
        raise ValueError("max_seconds must be positive when provided")

    jump_arguments = (
        jump_time_boundaries,
        jump_g0_values,
        jump_intensities,
        jump_means,
        common_jump_intensity,
        jump_size_correlation,
    )
    use_jumps = any(argument is not None for argument in jump_arguments)
    if use_jumps and any(argument is None for argument in jump_arguments):
        raise ValueError("All common/idiosyncratic jump arguments are required.")

    compute_call_vs_call = option_type in {"call_vs_call", "both"}
    compute_put_vs_put = option_type in {"put_vs_put", "both"}

    # Online aggregation (per strike) of call-vs-call and put-vs-put payoffs
    if compute_call_vs_call:
        sum_call_vs_call = np.zeros(nK)
        sum_call_vs_call2 = np.zeros(nK)
        call_vs_call_path, call_vs_call_std = [], []
    if compute_put_vs_put:
        sum_put_vs_put = np.zeros(nK)
        sum_put_vs_put2 = np.zeros(nK)
        put_vs_put_path, put_vs_put_std = [], []

    MC_effective = 0
    independent_sample_count = 0
    MC_path = []
    effective_sample_count_path = []
    elapsed_seconds_path = []
    relative_half_width_path = []
    start_time = time.perf_counter()
    z_value = float(norm.ppf(0.5 + 0.5 * confidence_level))
    status = "path_limit_reached"
    message = "The requested path limit was reached before the precision target."

    nbatches = (MC + batch_mc - 1) // batch_mc
    for b in range(nbatches):
        if b and max_seconds is not None:
            if time.perf_counter() - start_time >= max_seconds:
                status = "runtime_limit_reached"
                message = "The runtime limit was reached between Monte Carlo batches."
                break

        mc_start = b * batch_mc
        mc_stop = min((b + 1) * batch_mc, MC)
        current_batch = mc_stop - mc_start

        if use_antithetic and (current_batch % 2 != 0):
            mc_stop -= 1
            current_batch -= 1
            if current_batch == 0:
                break

        batch_seed = None if seed is None else (seed + 10_000 * b)

        simulation_arguments = dict(
            d=d,
            N=N,
            T=T,
            Y0=Y0,
            rho=rho,
            MM=MM,
            chol_V=chol_V,
            mc_start=mc_start,
            mc_stop=mc_stop,
            seed=batch_seed,
            use_antithetic=use_antithetic,
            return_log_returns=True,
            return_sigmas=False,
            return_delta_B=False,
        )
        if use_jumps:
            log_returns, _, _ = (
                simulate_log_prices_and_sigmas_with_jumps_batch(
                    **simulation_arguments,
                    jump_time_boundaries=jump_time_boundaries,
                    jump_g0_values=jump_g0_values,
                    jump_intensities=jump_intensities,
                    jump_means=jump_means,
                    common_jump_intensity=common_jump_intensity,
                    jump_size_correlation=jump_size_correlation,
                )
            )
        else:
            log_returns, _, _ = simulate_log_prices_and_sigmas_batch(
                **simulation_arguments
            )

        # S_T: last time index (i = N)
        log_ST = log_returns[:, -1, :]                  # (2, current_batch)
        S_T = S_0[:, None] * np.exp(log_ST)             # (2, current_batch)
        basket_T = 0.5 * (S_T[0] + S_T[1])             # (current_batch,)

        # Payoffs per strike: shape (current_batch, n_strikes)
        diff_1 = S_T[0, :, None] - K_array[None, :]
        diff_2 = S_T[1, :, None] - K_array[None, :]
        diff_basket = basket_T[:, None] - K_array[None, :]

        if compute_call_vs_call:
            call_vs_call_payoff = (
                0.5 * (
                    np.maximum(diff_1, 0.0)
                    + np.maximum(diff_2, 0.0)
                )
                - np.maximum(diff_basket, 0.0)
            )
        if compute_put_vs_put:
            put_vs_put_payoff = (
                0.5 * (
                    np.maximum(-diff_1, 0.0)
                    + np.maximum(-diff_2, 0.0)
                )
                - np.maximum(-diff_basket, 0.0)
            )

        if use_antithetic:
            half_batch = current_batch // 2
            if compute_call_vs_call:
                call_vs_call_samples = 0.5 * (
                    call_vs_call_payoff[:half_batch]
                    + call_vs_call_payoff[half_batch:]
                )
            if compute_put_vs_put:
                put_vs_put_samples = 0.5 * (
                    put_vs_put_payoff[:half_batch]
                    + put_vs_put_payoff[half_batch:]
                )
            current_sample_count = half_batch
        else:
            if compute_call_vs_call:
                call_vs_call_samples = call_vs_call_payoff
            if compute_put_vs_put:
                put_vs_put_samples = put_vs_put_payoff
            current_sample_count = current_batch

        if compute_call_vs_call:
            sum_call_vs_call += call_vs_call_samples.sum(axis=0)
            sum_call_vs_call2 += (call_vs_call_samples ** 2).sum(axis=0)
        if compute_put_vs_put:
            sum_put_vs_put += put_vs_put_samples.sum(axis=0)
            sum_put_vs_put2 += (put_vs_put_samples ** 2).sum(axis=0)

        MC_effective += current_batch
        independent_sample_count += current_sample_count
        MC_path.append(MC_effective)
        effective_sample_count_path.append(independent_sample_count)

        if compute_call_vs_call:
            call_vs_call_hat = sum_call_vs_call / independent_sample_count
            call_vs_call_path.append(call_vs_call_hat.copy())
        if compute_put_vs_put:
            put_vs_put_hat = sum_put_vs_put / independent_sample_count
            put_vs_put_path.append(put_vs_put_hat.copy())

        if return_std_error:
            if independent_sample_count < 2:
                if compute_call_vs_call:
                    call_vs_call_std.append(np.full(nK, np.nan))
                if compute_put_vs_put:
                    put_vs_put_std.append(np.full(nK, np.nan))
            else:
                if compute_call_vs_call:
                    var_call_vs_call = np.maximum(
                        (
                            sum_call_vs_call2
                            - sum_call_vs_call ** 2 / independent_sample_count
                        )
                        / (independent_sample_count - 1),
                        0.0,
                    )
                    call_vs_call_std.append(
                        np.sqrt(var_call_vs_call / independent_sample_count)
                    )
                if compute_put_vs_put:
                    var_put_vs_put = np.maximum(
                        (
                            sum_put_vs_put2
                            - sum_put_vs_put ** 2 / independent_sample_count
                        )
                        / (independent_sample_count - 1),
                        0.0,
                    )
                    put_vs_put_std.append(
                        np.sqrt(var_put_vs_put / independent_sample_count)
                    )

        if return_std_error:
            if compute_call_vs_call:
                active_price = call_vs_call_path[-1]
                active_standard_error = call_vs_call_std[-1]
            else:
                active_price = put_vs_put_path[-1]
                active_standard_error = put_vs_put_std[-1]
            relative_half_width = np.max(
                z_value * active_standard_error
                / np.maximum(np.abs(active_price), scale_floor)
            )
        else:
            relative_half_width = np.nan
        relative_half_width_path.append(relative_half_width)
        elapsed_seconds_path.append(time.perf_counter() - start_time)

        if compute_call_vs_call:
            active_path = np.asarray(call_vs_call_path)
        else:
            active_path = np.asarray(put_vs_put_path)
        enough_batches = len(active_path) >= max(
            minimum_batches, stability_batches
        )
        if enough_batches:
            recent = active_path[-stability_batches:]
            relative_moves = np.abs(np.diff(recent, axis=0)) / np.maximum(
                np.abs(recent[-1]), scale_floor
            )
            stable = np.max(relative_moves) <= stability_tolerance
        else:
            stable = False
        precise = (
            relative_half_width_target is not None
            and np.isfinite(relative_half_width)
            and relative_half_width <= relative_half_width_target
        )
        if precise and stable:
            status = "precision_target_reached"
            message = (
                "The relative confidence half-width and stability targets "
                "were reached."
            )
            break
        if max_seconds is not None:
            if time.perf_counter() - start_time >= max_seconds:
                status = "runtime_limit_reached"
                message = "The runtime limit was reached after a completed batch."
                break

        del log_returns, log_ST, S_T, basket_T
        del diff_1, diff_2, diff_basket

    MC_path = np.asarray(MC_path, dtype=int)
    if compute_call_vs_call:
        call_vs_call_path = np.asarray(call_vs_call_path)
    if compute_put_vs_put:
        put_vs_put_path = np.asarray(put_vs_put_path)

    diagnostics = {
        "effective_sample_count_path": np.asarray(
            effective_sample_count_path, dtype=int
        ),
        "elapsed_seconds_path": np.asarray(elapsed_seconds_path, dtype=float),
        "relative_half_width_path": np.asarray(
            relative_half_width_path, dtype=float
        ),
        "confidence_level": float(confidence_level),
        "status": status,
        "message": message,
    }

    def with_diagnostics(values):
        if not isinstance(values, tuple):
            values = (values,)
        return (*values, diagnostics) if return_diagnostics else values

    if option_type == "call_vs_call":
        if return_std_error:
            return with_diagnostics(
                (MC_path, call_vs_call_path, np.asarray(call_vs_call_std))
            )
        return with_diagnostics((MC_path, call_vs_call_path))
    if option_type == "put_vs_put":
        if return_std_error:
            return with_diagnostics(
                (MC_path, put_vs_put_path, np.asarray(put_vs_put_std))
            )
        return with_diagnostics((MC_path, put_vs_put_path))
    if return_std_error:
        return with_diagnostics((
            MC_path,
            put_vs_put_path,
            call_vs_call_path,
            np.asarray(put_vs_put_std),
            np.asarray(call_vs_call_std),
        ))
    return with_diagnostics((MC_path, put_vs_put_path, call_vs_call_path))
