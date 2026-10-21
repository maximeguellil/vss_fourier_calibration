"""Monte Carlo estimation of the signed two-point implied-volatility skew."""

from dataclasses import dataclass
import time

import numpy as np
from scipy.stats import norm

from vss.fourier.skew import implied_volatility_from_call_price

from .log_price_and_volatility_paths import (
    simulate_log_prices_and_sigmas_batch,
)
from .log_price_and_volatility_paths_with_jumps import (
    simulate_log_prices_and_sigmas_with_jumps_batch,
)


@dataclass(frozen=True)
class MonteCarloSkewResult:
    """Cumulative Monte Carlo diagnostics for a two-point skew estimate."""

    path_count: np.ndarray
    effective_sample_count: np.ndarray
    call_price_path: np.ndarray
    covariance_call_means_path: np.ndarray
    implied_volatility_path: np.ndarray
    vega_path: np.ndarray
    skew_path: np.ndarray
    standard_error_path: np.ndarray
    confidence_lower_path: np.ndarray
    confidence_upper_path: np.ndarray
    relative_half_width_path: np.ndarray
    batch_skew_path: np.ndarray
    batch_standard_error_path: np.ndarray
    elapsed_seconds_path: np.ndarray
    confidence_level: float
    status: str
    message: str


def black_scholes_vega(S0, K, T, volatility):
    """Return the zero-rate Black--Scholes Vega."""

    S0 = float(S0)
    K = float(K)
    T = float(T)
    volatility = float(volatility)
    if not np.isfinite(S0) or S0 <= 0.0:
        raise ValueError("S0 must be finite and strictly positive.")
    if not np.isfinite(K) or K <= 0.0:
        raise ValueError("K must be finite and strictly positive.")
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be finite and strictly positive.")
    if not np.isfinite(volatility) or volatility <= 0.0:
        return 0.0

    sigma_sqrt_T = volatility * np.sqrt(T)
    d1 = (
        np.log(S0 / K) + 0.5 * volatility**2 * T
    ) / sigma_sqrt_T
    return float(S0 * norm.pdf(d1) * np.sqrt(T))


def absolute_value_confidence_interval(lower, upper):
    """Map a signed confidence interval to the range of its absolute value."""

    lower_array, upper_array = np.broadcast_arrays(
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )
    if np.any(lower_array > upper_array):
        raise ValueError("lower must not exceed upper.")

    absolute_lower = np.where(
        lower_array >= 0.0,
        lower_array,
        np.where(upper_array <= 0.0, np.abs(upper_array), 0.0),
    )
    absolute_upper = np.where(
        lower_array >= 0.0,
        upper_array,
        np.where(
            upper_array <= 0.0,
            np.abs(lower_array),
            np.maximum(np.abs(lower_array), np.abs(upper_array)),
        ),
    )
    if absolute_lower.ndim == 0:
        return float(absolute_lower), float(absolute_upper)
    return absolute_lower, absolute_upper


def _skew_statistics(
    call_prices,
    covariance_call_means,
    *,
    S0,
    strikes,
    T,
    log_moneyness,
    minimum_vega,
):
    """Transform joint call-price statistics into skew statistics."""

    implied_volatilities = np.array([
        implied_volatility_from_call_price(price, S0, strike, T)
        for price, strike in zip(call_prices, strikes)
    ])
    vegas = np.array([
        black_scholes_vega(S0, strike, T, volatility)
        for strike, volatility in zip(strikes, implied_volatilities)
    ])
    if np.any(vegas <= minimum_vega):
        raise ValueError(
            "The Black--Scholes Vega is too small for a stable delta-method "
            f"interval (minimum observed Vega={vegas.min():.3e})."
        )

    delta_k = float(log_moneyness[1] - log_moneyness[0])
    skew = float(
        (implied_volatilities[1] - implied_volatilities[0]) / delta_k
    )
    gradient = np.array([
        -1.0 / (delta_k * vegas[0]),
        1.0 / (delta_k * vegas[1]),
    ])
    variance = float(gradient @ covariance_call_means @ gradient)
    standard_error = float(np.sqrt(max(variance, 0.0)))
    return implied_volatilities, vegas, skew, standard_error


def mc_skew(
    S_0,
    d: int,
    N: int,
    T: float,
    Y0: np.ndarray,
    rho: np.ndarray,
    MM: np.ndarray,
    chol_V: np.ndarray,
    MC: int,
    *,
    log_moneyness=np.log(np.array([0.99, 1.01])),
    batch_mc: int = 1024,
    seed: int | None = None,
    use_antithetic: bool = True,
    confidence_level: float = 0.95,
    minimum_vega: float = 1e-10,
    relative_half_width_target: float | None = None,
    scale_floor: float = 1e-8,
    minimum_batches: int = 3,
    stability_batches: int = 3,
    stability_tolerance: float = 0.02,
    max_seconds: float | None = None,
    jump_time_boundaries=None,
    jump_g0_values=None,
    jump_intensity=None,
    jump_mean=None,
):
    """Estimate a signed ATM-forward skew and its pointwise confidence interval.

    Both call strikes are evaluated on the same terminal values.  Their full
    covariance is propagated through the two Black--Scholes inversions with
    the delta method.  When antithetic sampling is used, pair averages are the
    independent observations used for every covariance and standard-error
    calculation. Gaussian compound-Poisson jumps are included when all four
    ``jump_*`` arguments are provided; otherwise the original diffusion-only
    simulation is used.
    """

    S_0 = float(S_0)
    T = float(T)
    log_moneyness = np.asarray(log_moneyness, dtype=float)
    if d != 1:
        raise ValueError("d must be equal to 1 for a marginal skew.")
    if not np.isfinite(S_0) or S_0 <= 0.0:
        raise ValueError("S_0 must be finite and strictly positive.")
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be finite and strictly positive.")
    if log_moneyness.shape != (2,) or not np.isfinite(log_moneyness).all():
        raise ValueError("log_moneyness must contain two finite values.")
    if log_moneyness[0] == log_moneyness[1]:
        raise ValueError("The two log-moneyness values must be distinct.")
    if MC <= 0 or batch_mc <= 0:
        raise ValueError("MC and batch_mc must be positive.")
    if use_antithetic and batch_mc % 2:
        raise ValueError("batch_mc must be even with antithetic sampling.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")
    if minimum_vega <= 0.0 or scale_floor <= 0.0:
        raise ValueError("minimum_vega and scale_floor must be positive.")
    if relative_half_width_target is not None:
        if relative_half_width_target <= 0.0:
            raise ValueError("relative_half_width_target must be positive.")
    if minimum_batches <= 0 or stability_batches < 2:
        raise ValueError(
            "minimum_batches must be positive and stability_batches must be at least 2."
        )
    if stability_tolerance <= 0.0:
        raise ValueError("stability_tolerance must be positive.")
    if max_seconds is not None and max_seconds <= 0.0:
        raise ValueError("max_seconds must be positive when provided.")

    jump_arguments = (
        jump_time_boundaries,
        jump_g0_values,
        jump_intensity,
        jump_mean,
    )
    use_jumps = any(argument is not None for argument in jump_arguments)
    if use_jumps and any(argument is None for argument in jump_arguments):
        raise ValueError(
            "jump_time_boundaries, jump_g0_values, jump_intensity, and "
            "jump_mean must be provided together."
        )

    strikes = S_0 * np.exp(log_moneyness)
    z_value = float(norm.ppf(0.5 + 0.5 * confidence_level))
    start_time = time.perf_counter()

    sum_payoffs = np.zeros(2)
    sum_outer_products = np.zeros((2, 2))
    independent_sample_count = 0
    simulated_path_count = 0

    path_counts = []
    effective_counts = []
    call_price_path = []
    covariance_mean_path = []
    implied_volatility_path = []
    vega_path = []
    skew_path = []
    standard_error_path = []
    confidence_lower_path = []
    confidence_upper_path = []
    relative_half_width_path = []
    batch_skew_path = []
    batch_standard_error_path = []
    elapsed_seconds_path = []
    valid_batch_skews = []
    status = "path_limit_reached"
    message = "The requested path limit was reached before the precision target."

    number_of_batches = (MC + batch_mc - 1) // batch_mc
    for batch_index in range(number_of_batches):
        if batch_index and max_seconds is not None:
            if time.perf_counter() - start_time >= max_seconds:
                status = "runtime_limit_reached"
                message = "The runtime limit was reached between Monte Carlo batches."
                break

        current_batch = min(batch_mc, MC - simulated_path_count)
        if use_antithetic and current_batch % 2:
            current_batch -= 1
        if current_batch <= 0:
            break
        batch_seed = (
            None if seed is None else int(seed) + 10_000 * batch_index
        )
        simulation_arguments = dict(
            d=d,
            N=N,
            T=T,
            Y0=Y0,
            rho=rho,
            MM=MM,
            chol_V=chol_V,
            mc_start=simulated_path_count,
            mc_stop=simulated_path_count + current_batch,
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
                    jump_intensities=jump_intensity,
                    jump_means=jump_mean,
                )
            )
        else:
            log_returns, _, _ = simulate_log_prices_and_sigmas_batch(
                **simulation_arguments
            )

        terminal_prices = S_0 * np.exp(log_returns[0, -1])
        payoffs = np.maximum(
            terminal_prices[:, None] - strikes[None, :],
            0.0,
        )
        if use_antithetic:
            half_batch = current_batch // 2
            samples = 0.5 * (
                payoffs[:half_batch] + payoffs[half_batch:]
            )
        else:
            samples = payoffs

        batch_call_prices = samples.mean(axis=0)
        batch_skew = np.nan
        try:
            batch_volatilities = np.array([
                implied_volatility_from_call_price(price, S_0, strike, T)
                for price, strike in zip(batch_call_prices, strikes)
            ])
            batch_skew = float(
                (batch_volatilities[1] - batch_volatilities[0])
                / (log_moneyness[1] - log_moneyness[0])
            )
        except ValueError:
            pass
        batch_skew_path.append(batch_skew)
        if np.isfinite(batch_skew):
            valid_batch_skews.append(batch_skew)
        batch_standard_error_path.append(
            np.std(valid_batch_skews, ddof=1) / np.sqrt(len(valid_batch_skews))
            if len(valid_batch_skews) >= 2 else np.nan
        )

        sum_payoffs += samples.sum(axis=0)
        sum_outer_products += samples.T @ samples
        independent_sample_count += samples.shape[0]
        simulated_path_count += current_batch

        call_prices = sum_payoffs / independent_sample_count
        if independent_sample_count >= 2:
            sample_covariance = (
                sum_outer_products
                - np.outer(sum_payoffs, sum_payoffs)
                / independent_sample_count
            ) / (independent_sample_count - 1)
            covariance_call_means = (
                sample_covariance / independent_sample_count
            )
        else:
            covariance_call_means = np.full((2, 2), np.nan)

        try:
            (
                implied_volatilities,
                vegas,
                skew,
                standard_error,
            ) = _skew_statistics(
                call_prices,
                covariance_call_means,
                S0=S_0,
                strikes=strikes,
                T=T,
                log_moneyness=log_moneyness,
                minimum_vega=minimum_vega,
            )
            half_width = z_value * standard_error
            lower = skew - half_width
            upper = skew + half_width
            relative_half_width = half_width / max(abs(skew), scale_floor)
        except ValueError:
            implied_volatilities = np.full(2, np.nan)
            vegas = np.full(2, np.nan)
            skew = standard_error = lower = upper = np.nan
            relative_half_width = np.nan

        path_counts.append(simulated_path_count)
        effective_counts.append(independent_sample_count)
        call_price_path.append(call_prices.copy())
        covariance_mean_path.append(covariance_call_means.copy())
        implied_volatility_path.append(implied_volatilities)
        vega_path.append(vegas)
        skew_path.append(skew)
        standard_error_path.append(standard_error)
        confidence_lower_path.append(lower)
        confidence_upper_path.append(upper)
        relative_half_width_path.append(relative_half_width)
        elapsed_seconds_path.append(time.perf_counter() - start_time)

        enough_batches = len(path_counts) >= max(
            minimum_batches, stability_batches
        )
        recent_skews = np.asarray(skew_path[-stability_batches:])
        stable = (
            enough_batches
            and np.isfinite(recent_skews).all()
            and np.max(np.abs(np.diff(recent_skews)))
            / max(abs(recent_skews[-1]), scale_floor)
            <= stability_tolerance
        )
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

    if not path_counts:
        raise RuntimeError("No Monte Carlo batch was completed.")
    if not np.isfinite(skew_path[-1]):
        status = "failed"
        message = (
            "The final call prices could not be converted to a stable skew; "
            "inspect the implied-volatility and Vega diagnostics."
        )

    return MonteCarloSkewResult(
        path_count=np.asarray(path_counts, dtype=int),
        effective_sample_count=np.asarray(effective_counts, dtype=int),
        call_price_path=np.asarray(call_price_path),
        covariance_call_means_path=np.asarray(covariance_mean_path),
        implied_volatility_path=np.asarray(implied_volatility_path),
        vega_path=np.asarray(vega_path),
        skew_path=np.asarray(skew_path),
        standard_error_path=np.asarray(standard_error_path),
        confidence_lower_path=np.asarray(confidence_lower_path),
        confidence_upper_path=np.asarray(confidence_upper_path),
        relative_half_width_path=np.asarray(relative_half_width_path),
        batch_skew_path=np.asarray(batch_skew_path),
        batch_standard_error_path=np.asarray(batch_standard_error_path),
        elapsed_seconds_path=np.asarray(elapsed_seconds_path),
        confidence_level=confidence_level,
        status=status,
        message=message,
    )
