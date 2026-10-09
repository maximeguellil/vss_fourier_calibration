import numpy as np

from .log_price_and_volatility_paths import (
    simulate_log_prices_and_sigmas_batch,
)


def mc_basket_call_put_price(
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
):
    """
    Monte Carlo estimator of European call and/or put prices on the two-dimensional arithmetic
    basket (S_T^1 + S_T^2) / 2 under the Quadratic Gaussian volatility model, for a vector of
    strikes (zero interest rates, no dividends).

    The simulation uses the chol_V a Cholesky factor of the covariance matrix V
    (see compute_V_matrix_for_mc), sampled batch by batch via simulate_log_prices_and_sigmas_batch
    with a left-point Euler scheme for the log-prices. With r = 0 the prices are the plain expected
    payoffs
        call(K) = E[ (B_T - K)^+ ],   put(K) = E[ (K - B_T)^+ ],
        B_T = (S_T^1 + S_T^2) / 2,
        S_T^i = S_0^i * exp(log(S_T^i/S_0^i)).

    Parameters
    ----------
    S_0 : 1D array_like of float, shape (2,)
        Initial asset prices [S_0^1, S_0^2].

    K_array : 1D array_like of float
        Basket strike prices.

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

    option_type : {"call", "put", "both"}
        Which basket-option prices to return.

    return_std_error : bool, optional
        If True, also return the Monte Carlo standard-error path(s).

    Returns
    -------
    MC_path : ndarray, shape (nbatches,)
        Cumulative number of simulated trajectories after each batch.

    price_path : ndarray, shape (nbatches, n_strikes)
        Cumulative basket-price estimate after each batch (put or call).

    std_path : ndarray, shape (nbatches, n_strikes), optional
        Cumulative standard error after each batch (if return_std_error).

    For option_type == "both", the return is
        (MC_path, put_path, call_path[, put_std, call_std]).
    """
    if option_type not in {"call", "put", "both"}:
        raise ValueError("option_type must be one of {'call', 'put', 'both'}")

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

    compute_call = option_type in {"call", "both"}
    compute_put = option_type in {"put", "both"}

    # Online aggregation (per strike) of basket call and put payoffs
    if compute_call:
        sum_call = np.zeros(nK)
        sum_call2 = np.zeros(nK)
        call_path, call_std = [], []
    if compute_put:
        sum_put = np.zeros(nK)
        sum_put2 = np.zeros(nK)
        put_path, put_std = [], []

    MC_effective = 0
    independent_sample_count = 0
    MC_path = []

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

        log_returns, _, _ = simulate_log_prices_and_sigmas_batch(
            d=d, N=N, T=T, Y0=Y0, rho=rho, MM=MM, chol_V=chol_V,
            mc_start=mc_start, mc_stop=mc_stop, seed=batch_seed,
            use_antithetic=use_antithetic,
            return_log_returns=True, return_sigmas=False, return_delta_B=False,
        )

        # S_T: last time index (i = N)
        log_ST = log_returns[:, -1, :]                  # (2, current_batch)
        S_T = S_0[:, None] * np.exp(log_ST)             # (2, current_batch)
        basket_T = 0.5 * (S_T[0] + S_T[1])             # (current_batch,)

        # Payoffs per strike: shape (current_batch, n_strikes)
        diff = basket_T[:, None] - K_array[None, :]
        if compute_call:
            call_payoff = np.maximum(diff, 0.0)
        if compute_put:
            put_payoff = np.maximum(-diff, 0.0)

        if use_antithetic:
            half_batch = current_batch // 2
            if compute_call:
                call_samples = 0.5 * (
                    call_payoff[:half_batch] + call_payoff[half_batch:]
                )
            if compute_put:
                put_samples = 0.5 * (
                    put_payoff[:half_batch] + put_payoff[half_batch:]
                )
            current_sample_count = half_batch
        else:
            if compute_call:
                call_samples = call_payoff
            if compute_put:
                put_samples = put_payoff
            current_sample_count = current_batch

        if compute_call:
            sum_call += call_samples.sum(axis=0)
            sum_call2 += (call_samples ** 2).sum(axis=0)
        if compute_put:
            sum_put += put_samples.sum(axis=0)
            sum_put2 += (put_samples ** 2).sum(axis=0)

        MC_effective += current_batch
        independent_sample_count += current_sample_count
        MC_path.append(MC_effective)

        if compute_call:
            call_hat = sum_call / independent_sample_count
            call_path.append(call_hat.copy())
        if compute_put:
            put_hat = sum_put / independent_sample_count
            put_path.append(put_hat.copy())

        if return_std_error:
            if independent_sample_count < 2:
                if compute_call:
                    call_std.append(np.full(nK, np.nan))
                if compute_put:
                    put_std.append(np.full(nK, np.nan))
            else:
                if compute_call:
                    var_call = np.maximum(
                        (
                            sum_call2
                            - sum_call ** 2 / independent_sample_count
                        )
                        / (independent_sample_count - 1),
                        0.0,
                    )
                    call_std.append(
                        np.sqrt(var_call / independent_sample_count)
                    )
                if compute_put:
                    var_put = np.maximum(
                        (
                            sum_put2
                            - sum_put ** 2 / independent_sample_count
                        )
                        / (independent_sample_count - 1),
                        0.0,
                    )
                    put_std.append(
                        np.sqrt(var_put / independent_sample_count)
                    )

        del log_returns, log_ST, S_T, basket_T, diff

    MC_path = np.asarray(MC_path, dtype=int)
    if compute_call:
        call_path = np.asarray(call_path)
    if compute_put:
        put_path = np.asarray(put_path)

    if option_type == "call":
        return (MC_path, call_path, np.asarray(call_std)) if return_std_error else (MC_path, call_path)
    if option_type == "put":
        return (MC_path, put_path, np.asarray(put_std)) if return_std_error else (MC_path, put_path)
    if return_std_error:
        return MC_path, put_path, call_path, np.asarray(put_std), np.asarray(call_std)
    return MC_path, put_path, call_path
