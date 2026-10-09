import numpy as np


def sum_exponential_KK_SIG_d1_n1(
    N: int,
    T: float,
    kappa_array: np.ndarray,
    c_array: np.ndarray,
):
    """
    Build the matrix K_{N,0} and the eta-free covariance core SIG_eta_free for
    d = n = 1, using a sum of exponential Volterra kernels:
        K(t,s) = 1_{s<t} sum_{m=1}^p c_m exp(-kappa_m (t - s)),
    with per-term weight c_m and decay rate kappa_m >= 0. A term with
    kappa_m = 0 is a constant contribution c_m (Brownian).
    Taking p = 1 gives a single exponential kernel.

    As with the other builders, eta is not an argument: SIG_eta_free depends only on
    (T, N, kappa_array, c_array), and the full covariance is
        Sigma_{N,0} = np.kron(SIG_eta_free, eta @ eta.T),
    where eta has shape (1, 1).

    Closed forms (t_k = k T / N). Integrated kernel (weighted single sum), for j <= i-1,
        (K_N)_{ij} = sum_m c_m ( exp(-kappa_m(t_{i-1}-t_j)) - exp(-kappa_m(t_{i-1}-t_{j-1})) ) / kappa_m
                    -> sum_m c_m * (T/N)   for the terms with kappa_m = 0.
    Eta-free covariance (weighted double sum), with tmin = min(t_{i-1}, t_{j-1}),
        (SIG_eta_free)_{ij} = sum_{m,m'} c_m c_{m'}
            ( exp(-kappa_m(t_{i-1}-tmin) - kappa_{m'}(t_{j-1}-tmin)) - exp(-kappa_m t_{i-1} - kappa_{m'} t_{j-1}) )
            / (kappa_m + kappa_{m'}),
    each term being int_0^{tmin} c_m e^{-kappa_m(t_{i-1}-u)} c_{m'} e^{-kappa_{m'}(t_{j-1}-u)} du.
    A pair with kappa_m = kappa_{m'} = 0 has limit c_m c_{m'} tmin (a single vanishing rate is
    handled by the general formula, denominator > 0).

    Parameters
    ----------
    N : int
        Number of time steps.

    T : float
        Maturity.

    kappa_array : ndarray, shape (1, p)
        Exponential decay rates kappa_m >= 0 (kappa_m = 0 gives a constant term).
        NB: kernel decay rates, unrelated to the model mean-reversion kappa (= 0 here).

    c_array : ndarray, shape (1, p)
        Weights c_m of the exponential terms.

    Returns
    -------
    KK : ndarray, shape (N, N)
        Discretized kernel matrix K_N.

    SIG_eta_free : ndarray, shape (N, N)
        Eta-free covariance core; Sigma_{N,0} = np.kron(SIG_eta_free, eta @ eta.T).
    """
    kappa_array = np.asarray(kappa_array, dtype=float)
    c_array = np.asarray(c_array, dtype=float)

    if kappa_array.ndim != 2 or kappa_array.shape[0] != 1:
        raise ValueError(f"kappa_array must have shape (1, p), got {kappa_array.shape}")
    if c_array.shape != kappa_array.shape:
        raise ValueError(f"c_array must have the same shape as kappa_array {kappa_array.shape}, got {c_array.shape}")

    p = kappa_array.shape[1]

    t = np.linspace(0.0, T, N + 1)

    # Time grids (same construction as in the other builders)
    tj_1 = np.tile(t[:-1], N).reshape(N, N)   # t_{j-1}
    ti_1 = tj_1.T                              # t_{i-1}
    tj   = np.tile(t[1:],  N).reshape(N, N)   # t_j

    mask = (tj <= ti_1)
    diff1 = ti_1[mask] - tj_1[mask]   # t_{i-1} - t_{j-1}
    diff2 = ti_1[mask] - tj[mask]     # t_{i-1} - t_j

    # K_N: weighted sum over the exponential terms
    KK = np.zeros((N, N))

    for m in range(p):
        kap = kappa_array[0, m]
        c = c_array[0, m]
        if kap == 0.0:
            KK[mask] += c * (diff1 - diff2)          # c * (t_j - t_{j-1}) = c * T/N
        else:
            KK[mask] += c * (np.exp(-kap * diff2) - np.exp(-kap * diff1)) / kap

    # Eta-free covariance core: weighted double sum over the exponential terms
    ti = t[:-1]
    TI = ti[:, None]
    TJ = ti[None, :]
    tmin = np.minimum(TI, TJ)

    SIG_eta_free = np.zeros((N, N))

    for m in range(p):
        a = kappa_array[0, m]
        for mp in range(p):
            b = kappa_array[0, mp]
            weight = c_array[0, m] * c_array[0, mp]
            if a + b == 0.0:
                term = tmin.copy()                        # int_0^{tmin} 1 du = min(t_{i-1}, t_{j-1})
            else:
                term = (np.exp(-a * (TI - tmin) - b * (TJ - tmin)) - np.exp(-a * TI - b * TJ)) / (a + b)
            SIG_eta_free += weight * term

    return KK, SIG_eta_free
