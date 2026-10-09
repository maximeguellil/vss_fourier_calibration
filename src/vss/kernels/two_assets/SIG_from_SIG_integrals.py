import numpy as np


def compute_SIG_matrix_d2_n2(SIG_integrals, nu, R_W):
    """
    Construct the covariance matrix Sigma_{N,0} of the two volatility factors
    from the kernel integrals, diagonal volatility-of-volatility matrix nu,
    and Brownian covariance matrix R_W.

    Parameters
    ----------
    SIG_integrals : ndarray, shape (N, N, 2, 2)
        Kernel integrals, with SIG_integrals[i, j, a, b] equal to the integral
        of K_a(t_i, r) K_b(t_j, r) over r from 0 to min(t_i, t_j).

    nu : array_like, shape (2,) or (2, 2)
        Diagonal entries of nu, or the diagonal matrix itself.

    R_W : ndarray, shape (2, 2)
        Instantaneous covariance matrix of the two Brownian factors W.

    Returns
    -------
    SIG : ndarray, shape (2*N, 2*N)
        Covariance matrix Sigma_{N,0}, ordered by time then factor, with
        SIG[2*i+a, 2*j+b] = nu_a nu_b (R_W)_{ab} SIG_integrals[i,j,a,b].
    """
    SIG_integrals = np.asarray(SIG_integrals, dtype=float)
    nu = np.asarray(nu, dtype=float)
    R_W = np.asarray(R_W, dtype=float)

    if SIG_integrals.ndim != 4 or SIG_integrals.shape[2:] != (2, 2) or SIG_integrals.shape[0] != SIG_integrals.shape[1]:
        raise ValueError(f"SIG_integrals must have shape (N, N, 2, 2), got {SIG_integrals.shape}")
    if nu.shape == (2, 2):
        if not np.allclose(nu, np.diag(np.diag(nu)), atol=1e-12, rtol=0):
            raise ValueError("nu must be diagonal")
        nu = np.diag(nu)
    elif nu.shape != (2,):
        raise ValueError(f"nu must have shape (2,) or (2, 2), got {nu.shape}")
    if R_W.shape != (2, 2):
        raise ValueError(f"R_W must have shape (2, 2), got {R_W.shape}")

    N = SIG_integrals.shape[0]
    SIG = np.einsum(
        "ijab,ab,a,b->iajb", SIG_integrals, R_W, nu, nu
    ).reshape(2 * N, 2 * N)

    return SIG
