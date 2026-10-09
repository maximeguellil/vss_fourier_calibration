import numpy as np
from scipy.special import gamma as gamma_func, hyp2f1

from ..one_asset.shifted_fractional import shifted_fractional_KK_SIG_d1_n1


def shifted_fractional_KK_SIG_d2_n2(
    N: int,
    T: float,
    H_array: np.ndarray,
    delta_array: np.ndarray,
):
    """
    Build the matrix K_{N,0} and the covariance integral core SIG_integrals for
    d = n = 2, using diagonal shifted fractional Volterra kernels:
        K_l(t,s) = 1_{s<t} (t - s + delta_l)^{H_l - 1/2} / Gamma(H_l + 1/2),
    with shifts delta_l >= 0. Setting delta_l = 0 gives a pure fractional kernel.

    The matrices eta and R_W are not arguments: SIG_integrals depends only on
    (T, N, H_array, delta_array). For diagonal eta = diag(nu_1, nu_2), the full
    covariance is obtained from SIG_integrals by compute_SIG_matrix_d2_n2.

    The diagonal blocks of K_{N,0} and SIG_integrals use the one-asset shifted
    fractional formulas. For tmin = min(t_{i-1}, t_{j-1}), set
        A = t_{i-1} - tmin + delta_1,  B = t_{j-1} - tmin + delta_2,
        alpha_l = H_l + 1/2.
    The cross integral is
        (SIG_integrals)_{ij12} = 1 / (Gamma(alpha_1) Gamma(alpha_2))
            * int_0^tmin (x + A)^{alpha_1 - 1} (x + B)^{alpha_2 - 1} dx.
    For A < B, with D = B - A, its primitive is
        P(y) = D^{alpha_2 - 1} y^{alpha_1} / alpha_1
               * 2F1(1 - alpha_2, alpha_1; alpha_1 + 1; -y / D),
    evaluated at y = A + tmin and y = A. For B < A, exchange the factor
    indices. When A = B, use the power or logarithmic limit instead.
    Finally, (SIG_integrals)_{ij21} = (SIG_integrals)_{ji12}.

    Notes
    -----
    Each diagonal block is computed by shifted_fractional_KK_SIG_d1_n1, so delta_l > 0
    and H_l = 0 is not handled. Validity ranges (not implemented): 0 < H_l < 1/2
    when delta_l = 0, and -1/2 < H_l < 1/2 when delta_l > 0.

    Parameters
    ----------
    N : int
        Number of time steps.

    T : float
        Maturity.

    H_array : ndarray, shape (2,)
        Hurst indices H_l of the two volatility factors.

    delta_array : ndarray, shape (2,)
        Kernel shifts delta_l >= 0 for the two factors.

    Returns
    -------
    KK : ndarray, shape (2*N, 2*N)
        Discretized diagonal kernel matrix K_N, ordered by time then factor.

    SIG_integrals : ndarray, shape (N, N, 2, 2)
        Kernel integral core; SIG_integrals[i, j, a, b] is the integral of
        K_a(t_i, r) K_b(t_j, r) over r from 0 to min(t_i, t_j), with
        zero-based time indices i, j and factor indices a, b.
    """
    H_array = np.asarray(H_array, dtype=float)
    delta_array = np.asarray(delta_array, dtype=float)

    if H_array.shape != (2,):
        raise ValueError(f"H_array must have shape (2,), got {H_array.shape}")
    if delta_array.shape != (2,):
        raise ValueError(f"delta_array must have shape (2,), got {delta_array.shape}")

    # Diagonal kernel and integral blocks from the one-asset builder
    KK = np.zeros((2 * N, 2 * N))
    SIG_integrals = np.zeros((N, N, 2, 2))

    for ell in range(2):
        K_block, SIG_block = shifted_fractional_KK_SIG_d1_n1(
            N=N,
            T=T,
            H=H_array[ell],
            delta=delta_array[ell],
        )
        KK[ell::2, ell::2] = K_block
        SIG_integrals[:, :, ell, ell] = SIG_block

    # Cross integral core
    t = np.linspace(0.0, T, N + 1)
    ti = t[:-1]
    TI = ti[:, None]
    TJ = ti[None, :]
    tmin = np.minimum(TI, TJ)

    A = TI - tmin + delta_array[0]
    B = TJ - tmin + delta_array[1]
    alpha_1 = H_array[0] + 0.5
    alpha_2 = H_array[1] + 0.5

    positive = tmin > 0.0
    equal = positive & (
        np.abs(A - B) <= 1e-14 * np.maximum(1.0, np.maximum(A, B))
    )
    A_smaller = positive & ~equal & (A < B)
    B_smaller = positive & ~equal & (B < A)

    SIG_cross = np.zeros((N, N))

    def primitive(y, D, alpha_small, alpha_large):
        return (
            D ** (alpha_large - 1.0)
            * y ** alpha_small / alpha_small
            * hyp2f1(1.0 - alpha_large, alpha_small, alpha_small + 1.0, -y / D)
        )

    D = (B - A)[A_smaller]
    y = A[A_smaller]
    SIG_cross[A_smaller] = primitive(y + tmin[A_smaller], D, alpha_1, alpha_2) - primitive(y, D, alpha_1, alpha_2)

    D = (A - B)[B_smaller]
    y = B[B_smaller]
    SIG_cross[B_smaller] = primitive(y + tmin[B_smaller], D, alpha_2, alpha_1) - primitive(y, D, alpha_2, alpha_1)

    q = H_array[0] + H_array[1]
    y = 0.5 * (A[equal] + B[equal])
    if q == 0.0:
        SIG_cross[equal] = np.log1p(tmin[equal] / y)
    else:
        SIG_cross[equal] = ((y + tmin[equal]) ** q - y ** q) / q

    SIG_cross /= gamma_func(alpha_1) * gamma_func(alpha_2)
    SIG_integrals[:, :, 0, 1] = SIG_cross
    SIG_integrals[:, :, 1, 0] = SIG_cross.T

    return KK, SIG_integrals
