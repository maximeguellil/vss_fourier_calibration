import numpy as np

from .log_price_and_volatility_paths import (
    simulate_log_prices_and_sigmas_batch,
)


def mc_vol_swap_price(
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
    return_std_error: bool = True,
):
    """
    Monte Carlo estimator of the non-annualized volatility-swap price E[sqrt(V_bar_T)] under the
    Quadratic Gaussian volatility model, with V_bar_T = int_0^T sigma_t^2 dt the integrated
    variance (d = 1).

    The simulation uses the chol_V a Cholesky factor of the covariance matrix V
    sampled batch by batch via simulate_log_prices_and_sigmas_batch. The integrated variance is
    approximated by the left-point Riemann sum V_bar_T ≈ sum_{j=1}^N sigma_j^2 * (T/N), consistent
    with the left-point Euler scheme used elsewhere.

    Parameters
    ----------
    d, N, T : int, int, float
        Model / discretization parameters (d = 1 here).

    Y0 : array_like, shape (n,)
        Initial volatility factor Y_0.

    rho : ndarray, shape (d, d)
        Correlation matrix of B, with d<B>_t = rho dt. This estimator
        is specialized to d = 1, where rho = [[1.0]].

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

    return_std_error : bool, optional
        If True, also return the Monte Carlo standard-error path.

    Returns
    -------
    MC_path : ndarray, shape (nbatches,)
        Cumulative number of simulated trajectories after each batch.

    price_path : ndarray, shape (nbatches,)
        Cumulative non-annualized volatility-swap estimate E[sqrt(V_bar_T)] after each batch.
        The corresponding annualized quote is price_path / sqrt(T).

    std_path : ndarray, shape (nbatches,), optional
        Cumulative standard error after each batch (if return_std_error).
    """
    if MC <= 0:
        raise ValueError("MC must be positive")
    if batch_mc <= 0:
        raise ValueError("batch_mc must be positive")
    if use_antithetic and (batch_mc % 2 != 0):
        raise ValueError("batch_mc must be even when use_antithetic=True")

    dt = T / N

    sum_H = 0.0
    sum_H2 = 0.0
    MC_effective = 0
    independent_sample_count = 0

    MC_path, price_path, std_path = [], [], []

    nbatches = (MC + batch_mc - 1) // batch_mc
    for b in range(nbatches):
        mc_start = b * batch_mc
        mc_stop = min((b + 1) * batch_mc, MC)
        current_batch = mc_stop - mc_start

        if use_antithetic and (current_batch % 2 != 0):
            mc_stop -= 1
            current_batch -= 1
            if current_batch == 0:
                break

        batch_seed = None if seed is None else (seed + 10_000 * b)

        _, sigmas, _ = simulate_log_prices_and_sigmas_batch(
            d=d, N=N, T=T, Y0=Y0, rho=rho, MM=MM, chol_V=chol_V,
            mc_start=mc_start, mc_stop=mc_stop, seed=batch_seed,
            use_antithetic=use_antithetic,
            return_log_returns=False, return_sigmas=True, return_delta_B=False,
        )

        # Integrated variance per path (d = 1): sum_j sigma_j^2 * dt
        sigma2 = sigmas[0, 0, :, :] ** 2                 # (N, current_batch)
        V_bar = dt * sigma2.sum(axis=0)                  # (current_batch,)
        H = np.sqrt(V_bar)                               # payoff = realised volatility

        if use_antithetic:
            half_batch = current_batch // 2
            H_samples = 0.5 * (H[:half_batch] + H[half_batch:])
        else:
            H_samples = H

        sum_H += H_samples.sum()
        sum_H2 += (H_samples ** 2).sum()
        MC_effective += current_batch
        independent_sample_count += H_samples.size

        vs_hat = sum_H / independent_sample_count
        MC_path.append(MC_effective)
        price_path.append(vs_hat)

        if return_std_error:
            if independent_sample_count < 2:
                std_path.append(np.nan)
            else:
                var_H = max(
                    (
                        sum_H2
                        - sum_H ** 2 / independent_sample_count
                    )
                    / (independent_sample_count - 1),
                    0.0,
                )
                std_path.append(
                    np.sqrt(var_H / independent_sample_count)
                )

        del sigmas, sigma2, V_bar, H, H_samples

    MC_path = np.asarray(MC_path, dtype=int)
    price_path = np.asarray(price_path)
    if return_std_error:
        return MC_path, price_path, np.asarray(std_path)
    return MC_path, price_path
