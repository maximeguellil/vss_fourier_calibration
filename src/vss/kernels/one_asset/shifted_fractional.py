import numpy as np
from scipy.special import gamma as gamma_func, hyp2f1


def shifted_fractional_KK_SIG_d1_n1(
    N: int,
    T: float,
    H: float,
    delta: float,
):
    """
    Build the matrix K_{N,0} and the eta-free covariance matrix SIG_eta_free for
    d = n = 1, using a shifted fractional Volterra kernel.

    The volatility-of-volatility matrix eta is deliberately not an argument: Since the covariance
    matrix Sigma_{N,0} is factored by eta, we can compute the eta-free part SIG_eta_free, depending
    only on (T, N, H, delta), once and reuse it across many eta.

    The full covariance is recovered by
        Sigma_{N,0} = np.kron(SIG_eta_free, eta @ eta.T),
    where eta has shape (1, 1).

    The kernel is
        K(t,s) = 1_{s<t} (t - s + delta)^{H - 1/2} / Gamma(H + 1/2),
    with shift delta >= 0. Setting delta = 0 recovers the pure fractional kernel.

    Notes
    -----
    H != 0 is assumed (the diagonal carries a 1/(2H) factor). The case H = 0 is NOT handled
    here. To include it one would replace the diagonal by its H -> 0 limit
    SIG_eta_free^{ii} = log((t_{i-1} + delta) / delta) / Gamma(alpha)^2,
    where alpha = H + 1/2 (requires delta > 0), the pure fractional kernel
    (delta = 0) not being square-integrable at H = 0.
    Validity ranges (not implemented): 0 < H < 1/2 when delta = 0, and
    -1/2 < H < 1/2 when delta > 0.

    Parameters
    ----------
    N : int
        Number of time steps.

    T : float
        Maturity.

    H : float
        Hurst index of the volatility factor.

    delta : float
        Kernel shift delta >= 0 (delta = 0 gives the pure fractional kernel).

    Returns
    -------
    KK : ndarray, shape (N, N)
        Discretized kernel matrix K_N.

    SIG_eta_free : ndarray, shape (N, N)
        Eta-free covariance core; the full covariance is
        Sigma_{N,0} = np.kron(SIG_eta_free, eta @ eta.T).
    """
    H = float(H)
    delta = float(delta)

    t = np.linspace(0.0, T, N + 1)

    # Time grids
    tj_1 = np.tile(t[:-1], N).reshape(N, N)   # t_{j-1}
    ti_1 = tj_1.T                              # t_{i-1}
    tj   = np.tile(t[1:],  N).reshape(N, N)   # t_j

    mask = (tj <= ti_1)

    # K_N

    KK = np.zeros((N, N))
    alpha = H + 0.5
    eps = delta
    diff1 = ti_1[mask] - tj_1[mask]
    diff2 = ti_1[mask] - tj[mask]

    KK[mask] = ((diff1 + eps)**alpha - (diff2 + eps)**alpha) / gamma_func(1.0 + alpha)

    # Eta-free covariance core

    ti = t[:-1]
    tj = t[:-1]

    TI = ti[:, None]
    TJ = tj[None, :]

    tmin = np.minimum(TI, TJ)
    tmax = np.maximum(TI, TJ)
    time_gap = tmax - tmin

    # Diagonal (i = j) and off-diagonal (i != j) parts, both restricted to t_{i-1} > 0
    positive = tmin > 0.0
    diagonal = positive & (time_gap <= 1e-14 * np.maximum(1.0, tmax))
    off_diagonal = positive & ~diagonal

    SIG_eta_free = np.zeros((N, N))

    # Diagonal part (i = j): closed form
    SIG_eta_free[diagonal] = (
        (tmin[diagonal] + eps) ** (2.0 * H) - eps ** (2.0 * H)
    ) / (2.0 * H * gamma_func(alpha) ** 2)

    # Off-diagonal part (i != j): form 3 via the primitive P
    delta_od = time_gap[off_diagonal]

    def primitive(x):
        return (
            delta_od ** (alpha - 1.0)
            * x ** alpha / alpha
            * hyp2f1(1.0 - alpha, alpha, alpha + 1.0, -x / delta_od)
        )

    SIG_eta_free[off_diagonal] = (
        primitive(tmin[off_diagonal] + eps) - primitive(eps)
    ) / gamma_func(alpha) ** 2

    return KK, SIG_eta_free
