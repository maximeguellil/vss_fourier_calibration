import numpy as np

from ..one_asset.sum_exponential import sum_exponential_KK_SIG_d1_n1


def sum_exponential_KK_SIG_d2_n2(
    N: int,
    T: float,
    kappa_array: np.ndarray,
    c_array: np.ndarray,
):
    """
    Build the matrix K_{N,0} and the covariance integral core SIG_integrals for
    d = n = 2, using diagonal sums of exponential Volterra kernels:
        K_l(t,s) = 1_{s<t} sum_{m=1}^p c_{l,m} exp(-kappa_{l,m} (t - s)),
    with per-term weight c_{l,m} and decay rate kappa_{l,m} >= 0. A term with
    kappa_{l,m} = 0 is a constant contribution c_{l,m} (Brownian).

    The matrices eta and R_W are not arguments: SIG_integrals depends only on
    (T, N, kappa_array, c_array). For diagonal eta = diag(nu_1, nu_2), the full
    covariance is recovered by
        Sigma_{N,0} = np.einsum(
            "ijab,ab,a,b->iajb", SIG_integrals, R_W, nu, nu
        ).reshape(2*N, 2*N),
    where nu = (nu_1, nu_2) and R_W is the covariance matrix of W.

    Closed forms (t_k = k T / N). The diagonal blocks of K_{N,0} and
    SIG_integrals use the one-asset sum-of-exponentials formulas. For
    tmin = min(t_{i-1}, t_{j-1}), the cross integral is
        (SIG_integrals)_{ij12} = sum_{m,m'} c_{1,m} c_{2,m'}
            exp(-kappa_{1,m}(t_{i-1}-tmin) - kappa_{2,m'}(t_{j-1}-tmin))
            * (1 - exp(-(kappa_{1,m} + kappa_{2,m'}) tmin))
            / (kappa_{1,m} + kappa_{2,m'}).
    The quotient equals tmin when both decay rates are zero, and
    (SIG_integrals)_{ij21} = (SIG_integrals)_{ji12}.

    Parameters
    ----------
    N : int
        Number of time steps.

    T : float
        Maturity.

    kappa_array : ndarray, shape (2, p)
        Exponential decay rates kappa_{l,m} >= 0 for each factor l.
        A factor with fewer than p terms can be padded with zero weights.
        NB: kernel decay rates, unrelated to the model mean-reversion kappa (= 0 here).

    c_array : ndarray, shape (2, p)
        Weights c_{l,m} of the exponential terms, distinct from the entries
        of the price-volatility correlation matrix C.

    Returns
    -------
    KK : ndarray, shape (2*N, 2*N)
        Discretized diagonal kernel matrix K_N, ordered by time then factor.

    SIG_integrals : ndarray, shape (N, N, 2, 2)
        Kernel integral core; SIG_integrals[i, j, a, b] is the integral of
        K_a(t_i, r) K_b(t_j, r) over r from 0 to min(t_i, t_j), with
        zero-based time indices i, j and factor indices a, b.
    """
    kappa_array = np.asarray(kappa_array, dtype=float)
    c_array = np.asarray(c_array, dtype=float)

    if kappa_array.ndim != 2 or kappa_array.shape[0] != 2:
        raise ValueError(f"kappa_array must have shape (2, p), got {kappa_array.shape}")
    if c_array.shape != kappa_array.shape:
        raise ValueError(f"c_array must have the same shape as kappa_array {kappa_array.shape}, got {c_array.shape}")

    p = kappa_array.shape[1]

    # Diagonal kernel and integral blocks from the one-asset builder
    KK = np.zeros((2 * N, 2 * N))
    SIG_integrals = np.zeros((N, N, 2, 2))

    for ell in range(2):
        K_block, SIG_block = sum_exponential_KK_SIG_d1_n1(
            N=N,
            T=T,
            kappa_array=kappa_array[ell:ell + 1],
            c_array=c_array[ell:ell + 1],
        )
        KK[ell::2, ell::2] = K_block
        SIG_integrals[:, :, ell, ell] = SIG_block

    # Cross integral core: weighted double sum over terms of both factors
    t = np.linspace(0.0, T, N + 1)
    ti = t[:-1]
    TI = ti[:, None]
    TJ = ti[None, :]
    tmin = np.minimum(TI, TJ)

    for m in range(p):
        a = kappa_array[0, m]
        for mp in range(p):
            b = kappa_array[1, mp]
            weight = c_array[0, m] * c_array[1, mp]
            if a + b == 0.0:
                term = tmin.copy()                        # int_0^{tmin} 1 du = min(t_{i-1}, t_{j-1})
            else:
                term = (
                    np.exp(-a * (TI - tmin) - b * (TJ - tmin))
                    * -np.expm1(-(a + b) * tmin) / (a + b)
                )
            SIG_integrals[:, :, 0, 1] += weight * term

    SIG_integrals[:, :, 1, 0] = SIG_integrals[:, :, 0, 1].T

    return KK, SIG_integrals
