import numpy as np
from scipy.special import gamma as gamma_func, gammainc, gammaincc, hyperu

from ..one_asset.shifted_fractional import shifted_fractional_KK_SIG_d1_n1
from ..one_asset.sum_exponential import sum_exponential_KK_SIG_d1_n1


def shifted_fractional_sum_exponential_KK_SIG_d2_n2(
    N: int,
    T: float,
    H: float,
    delta: float,
    kappa_array: np.ndarray,
    c_array: np.ndarray,
    order: str = "frac-exp",
):
    """
    Build the matrix K_{N,0} and the covariance integral core SIG_integrals for
    d = n = 2, using one shifted fractional and one sum-of-exponentials
    Volterra kernel:
        K_F(t,s) = 1_{s<t} (t - s + delta)^{H - 1/2} / Gamma(H + 1/2),
        K_E(t,s) = 1_{s<t} sum_{m=1}^p c_m exp(-kappa_m (t - s)).
    The order argument assigns the order of these kernels to the two volatility factors.

    The matrices eta and R_W are not arguments: SIG_integrals depends only on
    (T, N, H, delta, kappa_array, c_array, order). For diagonal
    eta = diag(nu_1, nu_2), the full covariance is obtained from
    SIG_integrals by compute_SIG_matrix_d2_n2.

    The diagonal blocks of K_{N,0} and SIG_integrals use the one-asset
    shifted fractional and sum-of-exponentials formulas. For the cross
    integral, let s = t_{i-1} be the fractional time, t = t_{j-1} the
    exponential time, tmin = min(s, t), alpha = H + 1/2,
    L = s - tmin + delta, and U = s + delta. Each exponential term has
        I_m = exp(-kappa_m (t - s - delta)) * kappa_m^{-alpha}
              * (P(alpha, kappa_m U) - P(alpha, kappa_m L)),
    where P is the regularized lower incomplete gamma function. For
    kappa_m = 0, use I_m = (U^alpha - L^alpha) / Gamma(alpha + 1).
    The cross integral is sum_m c_m I_m; the reverse block is its transpose.

    Notes
    -----
    The fractional diagonal block is computed by
    shifted_fractional_KK_SIG_d1_n1, so  delta > 0 and H = 0 is not handled. Validity
    ranges (not implemented): 0 < H < 1/2 when delta = 0, and
    -1/2 < H < 1/2 when delta > 0. The decay rates kappa_m are nonnegative.

    Parameters
    ----------
    N : int
        Number of time steps.

    T : float
        Maturity.

    H : float
        Hurst index of the fractional volatility factor.

    delta : float
        Fractional kernel shift delta >= 0.

    kappa_array : ndarray, shape (1, p)
        Exponential decay rates kappa_m >= 0 (kappa_m = 0 gives a constant term).
        NB: kernel decay rates, unrelated to the model mean-reversion kappa (= 0 here).

    c_array : ndarray, shape (1, p)
        Weights c_m of the exponential terms, distinct from the entries
        of the price-volatility correlation matrix C.

    order : {"frac-exp", "exp-frac"}, optional
        Kernel assigned to factors 1 and 2, respectively. Default is
        "frac-exp".

    Returns
    -------
    KK : ndarray, shape (2*N, 2*N)
        Discretized diagonal kernel matrix K_N, ordered by time then factor.

    SIG_integrals : ndarray, shape (N, N, 2, 2)
        Kernel integral core; SIG_integrals[i, j, a, b] is the integral of
        K_a(t_i, r) K_b(t_j, r) over r from 0 to min(t_i, t_j), with
        zero-based time indices i, j and factor indices a, b.
    """
    if order not in {"frac-exp", "exp-frac"}:
        raise ValueError("order must be 'frac-exp' or 'exp-frac'")

    H = float(H)
    delta = float(delta)
    kappa_array = np.asarray(kappa_array, dtype=float)
    c_array = np.asarray(c_array, dtype=float)

    if kappa_array.ndim != 2 or kappa_array.shape[0] != 1:
        raise ValueError(f"kappa_array must have shape (1, p), got {kappa_array.shape}")
    if c_array.shape != kappa_array.shape:
        raise ValueError(f"c_array must have the same shape as kappa_array {kappa_array.shape}, got {c_array.shape}")

    p = kappa_array.shape[1]
    frac_index = 0 if order == "frac-exp" else 1
    exp_index = 1 - frac_index

    # Diagonal kernel and integral blocks from the one-asset builders
    KK = np.zeros((2 * N, 2 * N))
    SIG_integrals = np.zeros((N, N, 2, 2))

    K_frac, SIG_frac = shifted_fractional_KK_SIG_d1_n1(
        N=N, T=T, H=H, delta=delta
    )
    K_exp, SIG_exp = sum_exponential_KK_SIG_d1_n1(
        N=N, T=T, kappa_array=kappa_array, c_array=c_array
    )
    KK[frac_index::2, frac_index::2] = K_frac
    KK[exp_index::2, exp_index::2] = K_exp
    SIG_integrals[:, :, frac_index, frac_index] = SIG_frac
    SIG_integrals[:, :, exp_index, exp_index] = SIG_exp

    # Cross integral core: weighted sum over the exponential terms
    t = np.linspace(0.0, T, N + 1)
    ti = t[:-1]
    TI = ti[:, None]  # fractional time
    TJ = ti[None, :]  # exponential time
    tmin = np.minimum(TI, TJ)
    L = TI - tmin + delta
    U = TI + delta
    alpha = H + 0.5

    SIG_cross = np.zeros((N, N))

    for m in range(p):
        kap = kappa_array[0, m]
        c = c_array[0, m]
        if kap == 0.0:
            term = (U ** alpha - L ** alpha) / gamma_func(alpha + 1.0)
        else:
            lower = kap * L
            upper = np.broadcast_to(kap * U, L.shape)
            term = np.empty_like(L)
            moderate = lower <= 100.0
            gamma_diff = (
                gammainc(alpha, upper[moderate])
                - gammainc(alpha, lower[moderate])
            )
            use_upper = lower[moderate] > 1.0
            gamma_diff[use_upper] = (
                gammaincc(alpha, lower[moderate][use_upper])
                - gammaincc(alpha, upper[moderate][use_upper])
            )
            term[moderate] = (
                np.exp(-kap * (TJ - TI - delta)[moderate])
                * kap ** (-alpha)
                * gamma_diff
            )

            # Scale the incomplete gamma terms when their exponential prefactor is large.
            large = ~moderate
            if np.any(large):
                scaled_lower = lower[large] ** alpha * hyperu(1.0, 1.0 + alpha, lower[large])
                scaled_upper = upper[large] ** alpha * hyperu(1.0, 1.0 + alpha, upper[large])
                tmin_full = np.broadcast_to(tmin, L.shape)
                TJ_full = np.broadcast_to(TJ, L.shape)
                term[large] = (
                    np.exp(-kap * (TJ_full[large] - tmin_full[large]))
                    * kap ** (-alpha) / gamma_func(alpha)
                    * (scaled_lower - np.exp(-kap * tmin_full[large]) * scaled_upper)
                )
        SIG_cross += c * term

    SIG_integrals[:, :, frac_index, exp_index] = SIG_cross
    SIG_integrals[:, :, exp_index, frac_index] = SIG_cross.T

    return KK, SIG_integrals
