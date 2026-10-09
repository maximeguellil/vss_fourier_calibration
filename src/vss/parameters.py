import numpy as np


def compute_tilde_Gamma(n, d, Gamma):
    """
    Compute the matrix tilde{Gamma} in R^{d x (n d)} from the
    vectorized coefficients Gamma.

    Parameters
    ----------
    n : int
        Number of volatility factors.

    d : int
        Number of assets.

    Gamma : ndarray, shape (d*d, n)
        Column-major vectorized coefficients gamma_{ij}: row i + d*j
        contains gamma_{ij}, with zero-based indices i and j.

    Returns
    -------
    tilde_Gamma : ndarray, shape (d, n*d)
        Matrix collecting the coefficients tilde{gamma}_l used in the
        computation of E[Z_t rho Z_t^T].
    """
    # Column-major convention for d=2: vec(Z) = (Z_11, Z_21, Z_12, Z_22)^T.
    # Hence Gamma_tensor[i, j, l] = Gamma[i + d*j, l].
    Gamma = Gamma.reshape(d, d, n, order="F")
    tilde_Gamma = np.concatenate(
        [Gamma[:, :, l] for l in range(n)],
        axis=1
    )  # shape (d, n*d)
    return tilde_Gamma


def compute_EZZ_inv_sqrt(d, tilde_Gamma, rho, Y0, SIG_ii):
    """
    Compute the inverse square root of the matrix
        E[Z_{t_{i-1}} rho Z_{t_{i-1}}^T] in R^{d x d}.

    Parameters
    ----------
    d : int
        Number of assets.

    tilde_Gamma : ndarray, shape (d, n*d)
        Computed via Gamma and the function compute_tilde_Gamma.

    rho : ndarray, shape (d, d)
        Correlation matrix of B, with d<B>_t = rho dt.

    Y0 : array_like of shape (n,)
        Initial volatility factor.

    SIG_ii : ndarray, shape (n, n)
        Covariance matrix of the volatility factor Y at time t_{i-1}.

    Returns
    -------
    EZZ_inv_sqrt : ndarray, shape (d, d)
        Inverse square root of E[Z_{t_{i-1}} rho Z_{t_{i-1}}^T].
    """
    # Equivalent to block-weighted sum
    n = SIG_ii.shape[0]

    tilde_gamma = tilde_Gamma.reshape(d, n, d).transpose(1, 0, 2)
    M2 = np.outer(Y0, Y0) + SIG_ii

    EZZ = np.einsum(
        "lm,lij,jk,mqk->iq",
        M2, tilde_gamma, rho, tilde_gamma,
    )
    EZZ = 0.5 * (EZZ + EZZ.T)

    eigvals, eigvecs = np.linalg.eigh(EZZ)
    return eigvecs @ np.diag(eigvals**-0.5) @ eigvecs.T


def compute_M_matrix(
    n,
    d,
    N,
    SIG,
    Gamma,
    tilde_Gamma,
    rho,
    G0,
    Y0,
):
    """
    Construct the block-diagonal matrix M = diag(M_1, ..., M_N), where
        M_i = (I_d ⊗ G0(t_{i-1}) E[Z_{t_{i-1}} rho Z_{t_{i-1}}^T]^{-1/2}) Gamma

    Parameters
    ----------
    n : int
        Number of volatility factors.

    d : int
        Number of assets.

    N : int
        Number of time steps.

    SIG : ndarray, shape (N*n, N*n)
        Covariance matrix Sigma_{N,0}; the block (i,i) is Cov(Y_{t_{i-1}}).

    Gamma : ndarray, shape (d*d, n)
        Vectorized coefficients gamma_{ij}.

    tilde_Gamma : ndarray, shape (d, n*d)
        Computed via Gamma and the function compute_tilde_Gamma

    rho : ndarray, shape (d, d)
        Correlation matrix of B, with d<B>_t = rho dt.

    G0 : ndarray, shape (N, d, d)
        Deterministic calibration matrices G_0(t_{i-1}).

    Y0 : array_like of shape (n,)
        Initial volatility factor.

    Returns
    -------
    MM : ndarray, shape (N*d*d, N*n)
        Block-diagonal matrix M.
    """
    MM = np.zeros((N * d * d, N * n))

    for i in range(N):
        # Compute E[Z_{t_{i-1}} rho Z_{t_{i-1}}^T]^{-1/2}
        SIG_ii = SIG[i*n:(i+1)*n, i*n:(i+1)*n]  # (n, n)
        EZZ_inv_sqrt = compute_EZZ_inv_sqrt(d, tilde_Gamma, rho, Y0, SIG_ii)

        # Build M_i
        A = G0[i] @ EZZ_inv_sqrt              # (d, d)
        Mi = np.kron(np.eye(d), A) @ Gamma   # (d^2, n)

        # Insert block
        MM[
            i*d*d:(i+1)*d*d,
            i*n:(i+1)*n
        ] = Mi

    return MM


def compute_M_matrix_d2_n2(
    N,
    SIG,
    Gamma,
    marginal_G0,
    Y0,
):
    """
    Construct the matrix M for the d = n = 2 specialization of
    Example 1.1.

    In this specialization, Z_t = diag(Y_t^1, Y_t^2) and

        G_0(t) = D(t) E[Z_t rho Z_t^T]^{1/2},

    where

        D_ii(t) = g_0^i(t) / sqrt(E[(Y_t^i)^2]).

    Consequently, the matrix entering the Fourier and Monte-Carlo
    discretizations simplifies to

        M_i = (I_2 ⊗ D(t_{i-1})) Gamma.

    Parameters
    ----------
    N : int
        Number of time steps.

    SIG : ndarray, shape (2*N, 2*N)
        Covariance matrix Sigma_{N,0}; the block (i, i) is the
        covariance matrix of Y at time t_{i-1}.

    Gamma : ndarray, shape (4, 2)
        Column-major vectorized coefficients defining
        Z_t = diag(Y_t^1, Y_t^2).

    marginal_G0 : ndarray, shape (N, 2)
        Values of the two marginal curves g_0^1 and g_0^2 on the
        left-point time grid used by the numerical discretization.

    Y0 : array_like, shape (2,)
        Initial volatility factors.

    Returns
    -------
    MM : ndarray, shape (4*N, 2*N)
        Block-diagonal matrix M.
    """
    N = int(N)
    SIG = np.asarray(SIG, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)
    marginal_G0 = np.asarray(marginal_G0, dtype=float)
    Y0 = np.asarray(Y0, dtype=float)

    if N <= 0:
        raise ValueError("N must be a positive integer.")
    if SIG.shape != (2 * N, 2 * N):
        raise ValueError("SIG must have shape (2*N, 2*N).")
    if Gamma.shape != (4, 2):
        raise ValueError("Gamma must have shape (4, 2).")
    if marginal_G0.shape != (N, 2):
        raise ValueError("marginal_G0 must have shape (N, 2).")
    if Y0.shape != (2,):
        raise ValueError("Y0 must have shape (2,).")
    if not np.all(np.isfinite(marginal_G0)):
        raise ValueError("marginal_G0 must contain finite values.")
    if np.any(marginal_G0 < 0.0):
        raise ValueError("marginal_G0 must contain non-negative values.")

    MM = np.zeros((4 * N, 2 * N))

    for i in range(N):
        SIG_ii = SIG[2*i:2*(i + 1), 2*i:2*(i + 1)]
        second_moments = Y0**2 + np.diag(SIG_ii)
        if np.any(second_moments <= 0.0):
            raise ValueError("The second moments of Y must be positive.")

        D_i = np.diag(marginal_G0[i] / np.sqrt(second_moments))
        M_i = np.kron(np.eye(2), D_i) @ Gamma
        MM[4*i:4*(i + 1), 2*i:2*(i + 1)] = M_i

    return MM
