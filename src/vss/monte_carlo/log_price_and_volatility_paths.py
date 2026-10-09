import numpy as np


def simulate_log_prices_and_sigmas_batch(
    d: int,
    N: int,
    T: float,
    Y0: np.ndarray,
    rho: np.ndarray,
    MM: np.ndarray,
    chol_V: np.ndarray,
    mc_start: int,
    mc_stop: int,
    seed: int | None = None,
    *,
    use_antithetic: bool = False,
    return_log_returns: bool = True,
    return_sigmas: bool = True,
    return_delta_B: bool = False,
):
    """
    Simulate, on the uniform grid t_i = i*T/N, a Monte Carlo batch of:
      - instantaneous volatility matrices sigma_j = sigma(t_{j-1}, Y_{t_{j-1}}), j=1..N,
      - log-returns log(S_{t_i}/S_0), i=1..N,

    under the Gaussian construction:
        X := (vec(sigma_1),...,vec(sigma_N), ΔB_1,...,ΔB_N) in R^{N(d^2+d)}
        X ~ N(mu_N, V),
    where mu_N = (MM @ g_N0, 0_{N*d}) and g_N0 = (Y0,...,Y0),
    and chol_V is a factor of V.

    The discretization for log-prices uses left-point Euler:
        log(S_{t_i}/S_0)
        ≈ sum_{j=1}^i [ sigma_j ΔB_j - 1/2 Δt diag(sigma_j rho sigma_j^T) ].

    Parameters
    ----------
    d : int
        Number of assets.

    N : int
        Number of time steps on the uniform grid.

    T : float
        Maturity.

    Y0 : array_like, shape (n,)
        Initial volatility factor Y_0.

    rho : ndarray, shape (d, d)
        Correlation matrix of B, with d<B>_t = rho dt.

    MM : ndarray, shape (N*d^2, N*n)
        Block-diagonal matrix M.

    chol_V : ndarray, shape (N*(d*d + d), r)
        Covariance factor such that V = chol_V @ chol_V.T obtained
        via the function compute_V_matrix_for_mc.
        r is the numerical rank of chol_V, i.e the number of eigvals > tol.

    mc_start, mc_stop : int
        Indices defining the Monte Carlo batch.
        batch_mc = mc_stop - mc_start trajectories are simulated.

    seed : int or None, optional
        Random seed for reproducibility (local to this function call).

    use_antithetic : bool, optional
    If True, use antithetic variables to reduce Monte Carlo variance.
    Standard normal samples Z are generated in pairs (Z, -Z), leading to
    correlated trajectories (X, -X) with the same Gaussian law.
    In this case, the batch size batch_mc = mc_stop - mc_start must be even.
    Default is False.

    return_log_returns : bool, optional
        If True, return the simulated log-returns log(S_{t_i}/S_0).
        If False, log-returns are not stored nor returned.
        Default is True.

    return_sigmas : bool, optional
        If True, return the instantaneous volatility matrices sigma_j.
        If False, sigma_j are not stored nor returned.
        Default is True.

    return_delta_B : bool, optional
        If True, return the Brownian increments ΔB_j.
        If False, ΔB_j are not stored nor returned.
        Default is False.

    Returns
    -------
    log_returns : ndarray, shape (d, N, batch_mc)
        log(S_{t_i}/S_0) for i=1..N (does not include i=0).

    sigmas : ndarray, shape (d, d, N, batch_mc)
        sigma_j = sigma(t_{j-1}, Y_{t_{j-1}}) for j=1..N.

    delta_B : ndarray, shape (d, N, batch_mc)
        Brownian increments ΔB_j for j=1..N
    """
    if mc_stop <= mc_start:
        raise ValueError("mc_stop must be > mc_start")
    batch_mc = mc_stop - mc_start

    if use_antithetic:
        if batch_mc % 2 != 0:
            raise ValueError("batch_mc must be even when using antithetic variables")
        half = batch_mc // 2
    else:
        half = batch_mc

    dim = N * (d**2 + d)
    chol_V = np.asarray(chol_V)

    if chol_V.shape[0] != dim:
        raise ValueError(
        f"chol_V must have shape (dim, r), got {chol_V.shape}"
        )

    r = chol_V.shape[1]

    dt = T / N

    rng = np.random.default_rng(seed)

    # Draw standard normals and correlate: X = chol_V @ Z
    Z_half = rng.standard_normal(size=(r, half))

    if use_antithetic:
        Z = np.concatenate([Z_half, -Z_half], axis=1)
    else:
        Z = Z_half

    # Compute initiale value of volatility and brownian increments
    Y0 = np.asarray(Y0).reshape(-1)  # shape (n,) where n = len(Y0)
    g_N0 = np.tile(Y0, N)          # (N*n,)
    Mg = MM @ g_N0                 # (N*d^2,)
    zeros = np.zeros(N * d)        # (N*d,)

    X0 = np.concatenate([Mg, zeros])

    # Compute volatility path
    X = X0[:, None] + chol_V @ Z  # shape (dim, batch_mc)

    # Split X into sigma-part and deltaB-part
    Nd2 = N * d**2
    X_sigma = X[:Nd2, :]          # (N*d^2, batch_mc)
    X_dB = X[Nd2:, :]             # (N*d, batch_mc)
    del X, Z

    # Reconstruct each matrix sigma_j
    # Keep each time block together: column-wise vec gives axes
    # (time, column, row, path), then transpose to (row, column, time, path).
    sigmas = (
        X_sigma
        .reshape(N, d, d, batch_mc)
        .transpose(2, 1, 0, 3)
    )  # shape (d, d, N, batch_mc)
    del X_sigma

    # Reshape delta_B: (N*d, batch_mc) -> (d, N, batch_mc)
    delta_B = (
        X_dB
        .reshape(N, d, batch_mc)
        .transpose(1, 0, 2)
    )
    del X_dB

    log_returns = None
    if return_log_returns:
        # Compute log-returns by left-point Euler (vectorized)
        # term_j = sigma_j ΔB_j - 1/2 diag(sigma_j sigma_j^T) dt
        martingale = np.einsum(
            "abjm,bjm->ajm",
            sigmas,
            delta_B,
        )  # sigma_j @ ΔB_j

        drift = 0.5 * dt * np.einsum(
            "abjm,bc,acjm->ajm",
            sigmas, rho, sigmas,
        )
        log_returns = np.cumsum(martingale - drift, axis=1)
        del martingale, drift

    # Drop unused outputs to save memory
    if not return_sigmas:
        del sigmas
        sigmas = None

    if not return_delta_B:
        del delta_B
        delta_B = None

    return log_returns, sigmas, delta_B
