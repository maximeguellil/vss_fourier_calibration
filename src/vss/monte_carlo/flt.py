import numpy as np

from ..fourier.admissibility import _check_admissibility_Ud
from .log_price_and_volatility_paths import (
    simulate_log_prices_and_sigmas_batch,
)


def _vss_fourier_laplace_eval_unique_mc(
    u,
    w,
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
    Monte Carlo estimator of xi_mc_0(u,w) = E[ exp( u^T log(S_T/S_0) + int_0^T Tr(w sigma rho sigma^T) ds ) ].

    The simulation uses the Gaussian construction:
        X := (vec(sigma_1),...,vec(sigma_N), ΔB_1,...,ΔB_N) ~ N(mu_N, V),
    with mu_N = (MM @ g_N0, 0_{N*d}), g_N0 = (Y0,...,Y0), and chol_V
    a factor of V. It uses left-point Euler both for:
      - log-prices (already inside simulate_log_prices_and_sigmas_batch),
      - the integral of Tr(w sigma rho sigma^T).

    Parameters
    ----------
    u : array_like, shape (d,)
        Complex Laplace/Fourier variable.

    w : array_like, shape (d,d)
        Complex symmetric matrix variable.

    d, N, T : int, int, float
        Model / discretization parameters.

    Y0 : array_like, shape (n,)
        Initial volatility factor Y_0.

    rho : ndarray, shape (d, d)
        Correlation matrix of B, with d<B>_t = rho dt.

    MM : ndarray, shape (N*d^2, N*n)
        Block-diagonal matrix M.

    chol_V : ndarray, shape (N*(d*d + d), N*(d*d + d))
        Cholesky factor of V.

    MC : int
        Total number of Monte Carlo trajectories.

    batch_mc : int, optional
        Number of trajectories per batch.
        If use_antithetic=True, batch_mc must be even.

    seed : int or None, optional
        Base seed. Each batch uses a derived seed for reproducibility.

    use_antithetic : bool, optional
        If True, each batch uses antithetic normals (Z, -Z). Standard errors are then computed
        from the independent pair averages.

    return_std_error : bool, optional
        If True, also return the Monte Carlo standard error (complex).

    Returns
    -------
    MC_path : ndarray, shape (nbatches,), optional
        Cumulative number of effective Monte Carlo trajectories
        after each batch.

    xi_path : ndarray, shape (nbatches,), optional
        Cumulative Monte Carlo estimator after each batch.

    std_path : ndarray, shape (nbatches,), optional
        Cumulative Monte Carlo standard error after each batch.
    """
    u = np.asarray(u, dtype=complex).reshape(d)
    w = np.asarray(w, dtype=complex).reshape(d, d)
    _check_admissibility_Ud(u, w, d)

    if MC <= 0:
        raise ValueError("MC must be positive")
    if batch_mc <= 0:
        raise ValueError("batch_mc must be positive")
    if use_antithetic and (batch_mc % 2 != 0):
        raise ValueError("batch_mc must be even when use_antithetic=True")

    dt = T / N

    # Online aggregation to avoid storing all payoffs
    sum_H = 0.0 + 0.0j
    sum_H2 = 0.0 + 0.0j  # for variance proxy (complex); we will use |H|^2 for error magnitude if desired
    MC_effective = 0
    independent_sample_count = 0

    # Store convergence path
    xi_path = []
    std_path = []
    MC_path = []

    # Batch loop
    nbatches = (MC + batch_mc - 1) // batch_mc
    for b in range(nbatches):
        mc_start = b * batch_mc
        mc_stop = min((b + 1) * batch_mc, MC)
        current_batch = mc_stop - mc_start

        if use_antithetic and (current_batch % 2 != 0):
            # To keep the contract "batch must be even", shrink last batch by 1 if needed
            # making MC_effective = MC - 1 if the last current_batch is odd and use_antithetic=True, else MC
            mc_stop -= 1
            current_batch -= 1
            if current_batch == 0:
                break

        batch_seed = None if seed is None else (seed + 10_000 * b)

        log_returns, sigmas, _ = simulate_log_prices_and_sigmas_batch(
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
            return_sigmas=True,
            return_delta_B=False,
        )

        # log(S_T/S_0): last time index (i=N) -> index -1
        log_ST = log_returns[:, -1, :]  # shape (d, current_batch)

        # u^T log(S_T/S_0) per path
        term_log = np.einsum("a,am->m", u, log_ST)  # shape (current_batch,)

        # Left-point Euler for integral of Tr(w sigma rho sigma^T)
        # A_j = sigma_j rho sigma_j^T, computed for all j,m
        A = np.einsum(
            "abjm,bc,ecjm->aejm",
            sigmas, rho, sigmas,
        )  # (d,d,N,current_batch)
        tr_wA = np.einsum("ab,bajm->jm", w, A)            # (N,current_batch)
        term_int = dt * np.sum(tr_wA, axis=0)             # (current_batch,)

        H = np.exp(term_log + term_int)                   # (current_batch,)

        if use_antithetic:
            half_batch = current_batch // 2
            H_samples = 0.5 * (H[:half_batch] + H[half_batch:])
        else:
            H_samples = H

        sum_H += np.sum(H_samples)
        sum_H2 += np.sum(H_samples * np.conjugate(H_samples))
        MC_effective += current_batch
        independent_sample_count += H_samples.size

        if MC_effective == 0:
            raise RuntimeError("No trajectories were simulated (check MC, batch_mc, use_antithetic).")

        # Cumulative estimates
        xi_mc = sum_H / independent_sample_count
        xi_path.append(xi_mc)
        MC_path.append(MC_effective)

        if return_std_error:
            if independent_sample_count < 2:
                std_path.append(np.nan)
            else:
                sample_variance = max(
                    (
                        sum_H2
                        - np.abs(sum_H) ** 2 / independent_sample_count
                    ).real
                    / (independent_sample_count - 1),
                    0.0,
                )
                std_mc = np.sqrt(
                    sample_variance / independent_sample_count
                ).real
                std_path.append(std_mc)

        del log_returns, sigmas, log_ST, term_log, A, tr_wA, term_int, H
        del H_samples

    if return_std_error:
        return (
            np.array(MC_path),
            np.array(xi_path),
            np.array(std_path),
        )

    return np.array(MC_path), np.array(xi_path)
