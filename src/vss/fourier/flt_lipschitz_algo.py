import numpy as np
from joblib import Parallel, delayed

from .admissibility import _check_admissibility_Ud


def _vss_fourier_laplace_discontinuous_eval_unique_det(
    u,
    w,
    n,
    d,
    N,
    T,
    C,
    rho,
    R_W,
    V,
    g_tilde,
):
    """
    Evaluate the discrete-time approximation of the Fourier–Laplace transform
    of log-prices and integrated covariance for the Quadratic Gaussian volatility
    model, for a single pair (u, w)

    This function computes
        exp( <Psi(u,w) g_tilde, g_tilde> ) / sqrt(det(Phi(u,w))),
    i.e. the approximation xi_N(u,w) without the discontinuous phase factor
        exp(i * pi * k_N(u,w)).
    The integrated covariance is int_0^T sigma_s rho sigma_s^T ds, paired
    with w through Tr(w sigma_s rho sigma_s^T).

    V and g_tilde are assumed to be precomputed offline.

    Parameters
    ----------
    u : array of shape (d,)
        Complex Laplace/Fourier variable applied to the log-price.

    w : array of shape (d, d)
        Complex symmetric matrix applied to the integrated covariance.

    n : int
        Dimension of the volatility factor.

    d : int
        Number of assets.

    N : int
        Number of time steps on the uniform grid.

    T : float
        Maturity.

    C : ndarray, shape (d, n)
        Matrix whose columns satisfy C_k^T C_k <= 1.

    rho : ndarray, shape (d, d)
        Correlation matrix of B, with d<B>_t = rho dt.

    R_W : ndarray, shape (n, n)
        Covariance matrix of W, with d<W>_t = R_W dt and
        R_W = C.T @ rho @ C + I_n - diag(C.T @ C).

    V : ndarray of shape (N*(d**2 + n), N*(d**2 + n))
        Precomputed covariance matrix V_{N,0} (independent of (u,w)) computed
        by the function compute_V_matrix_for_fourier.

    g_tilde : ndarray of shape (N*(d**2 + n),).
        Precomputed vector tilde{g}_{N,0} (independent of (u,w)).

    Returns
    -------
    fourier_laplace_value : complex
        Value of exp( <Psi g_tilde, g_tilde> ) / sqrt(det(Phi)) as a complex scalar.
    """
    # Admissibility checks for (u, w) in U_d
    u = np.asarray(u)
    w = np.asarray(w)
    _check_admissibility_Ud(u, w, d)

    # Basic shape checks for offline objects
    dim = N * (d**2 + n)
    V = np.asarray(V)
    if V.shape != (dim, dim):
        raise ValueError(f"V must have shape ({dim},{dim}), got {V.shape}")

    g_tilde = np.asarray(g_tilde)
    if g_tilde.ndim == 1:
        if g_tilde.shape != (dim,):
            raise ValueError(f"g_tilde must have shape ({dim},) or ({dim},1), got {g_tilde.shape}")
        g_tilde = g_tilde.reshape(dim, 1)
    elif g_tilde.ndim == 2:
        if g_tilde.shape != (dim, 1):
            raise ValueError(f"g_tilde must have shape ({dim},1), got {g_tilde.shape}")
    else:
        raise ValueError("g_tilde must be a 1D or 2D array")

    # Construction of alpha(u,w) and beta(u).
    u_col = u.reshape(d, 1)
    uuT = u_col @ u_col.T
    diag_u = np.diag(u)

    R_W_pinv = np.linalg.pinv(R_W, hermitian=True)

    alpha = (
        np.kron(rho, w + 0.5 * (uuT - diag_u))
        - 0.5 * np.kron(
            rho @ C @ R_W_pinv @ C.T @ rho,
            uuT,
        )
    )

    beta = np.kron(
        R_W_pinv @ C.T @ rho,
        u_col.T,
    )

    if alpha.shape != (d**2, d**2):
        raise ValueError(
            f"alpha must have shape {(d**2, d**2)}, "
            f"got {alpha.shape}."
        )

    if beta.shape != (n, d**2):
        raise ValueError(
            f"beta must have shape {(n, d**2)}, "
            f"got {beta.shape}."
        )

    # Build w_{N,0}(u,w)
    dt = T/N
    d2 = d * d
    Nd2 = N * d2
    Nn = N*n

    # omega11 = dt * kron(I_N, alpha)  -> block diagonal fill
    # NOTE: We do not use explicit Kronecker products kron(I_N, ·) to avoid
    # allocating large zero blocks; instead we fill the block-diagonal structure directly for efficiency.
    omega11 = np.zeros((Nd2, Nd2), dtype=complex)
    for i in range(N):
        a0 = i * d2
        a1 = (i + 1) * d2
        omega11[a0:a1, a0:a1] = dt * alpha

    # omega12 = 0.5 * (kron(I_N, beta))^T
    # tilde_beta has shape (N * beta_rows, N * beta_cols)
    br, bc = beta.shape
    tilde_beta = np.zeros((N * br, N * bc), dtype=complex)
    for i in range(N):
        r0 = i * br
        r1 = (i + 1) * br
        c0 = i * bc
        c1 = (i + 1) * bc
        tilde_beta[r0:r1, c0:c1] = beta
    omega12 = 0.5 * tilde_beta.T
    omega21 = 0.5 * tilde_beta

    omega22 = np.zeros((Nn, Nn), dtype=complex)

    omega = np.block([[omega11, omega12],
                      [omega21, omega22]]).astype(complex, copy=False)

    # Phi, det(Phi), Psi, quadratic form
    Phi = np.eye(dim, dtype=complex) - 2.0 * (omega @ V)
    det_Phi = np.linalg.det(Phi)

    Psi = np.linalg.solve(Phi, omega)
    quad = (g_tilde.T @ (Psi @ g_tilde)).item()
    fourier_laplace_value = np.exp(quad) / np.sqrt(det_Phi)

    return fourier_laplace_value, det_Phi


def vss_fourier_laplace_discontinuous_det(
    u_array,
    w_array,
    n,
    d,
    N,
    T,
    C,
    rho,
    R_W,
    V,
    g_tilde,
):
    """
    Evaluate the discrete-time approximation of the Fourier–Laplace transform
    of log-prices and integrated covariance for the Quadratic Gaussian volatility
    model, for every pair (u, w) in (u_array, w_array). The code is parallelized over
    the input arrays (u_array, w_array).

    This function computes
        exp( <Psi(u,w) g_tilde, g_tilde> ) / sqrt(det(Phi(u,w))),
    i.e. the approximation xi_N(u,w) without the discontinuous phase factor
        exp(i * pi * k_N(u,w)).
    The integrated covariance is int_0^T sigma_s rho sigma_s^T ds, paired
    with w through Tr(w sigma_s rho sigma_s^T).

    V and g_tilde are assumed to be precomputed offline.

    Parameters
    ----------
    u_array : ndarray, shape (c, d)
        Complex Laplace/Fourier variables applied to the log-prices.

    w_array : ndarray, shape (c, d, d)
        Complex symmetric matrices applied to the integrated covariance.

    n : int
        Dimension of the volatility factor.

    d : int
        Number of assets.

    N : int
        Number of time steps on the uniform grid.

    T : float
        Maturity.

    C : ndarray, shape (d, n)
        Matrix whose columns satisfy C_k^T C_k <= 1.

    rho : ndarray, shape (d, d)
        Correlation matrix of B, with d<B>_t = rho dt.

    R_W : ndarray, shape (n, n)
        Covariance matrix of W, with d<W>_t = R_W dt and
        R_W = C.T @ rho @ C + I_n - diag(C.T @ C).

    V : ndarray of shape (N*(d**2 + n), N*(d**2 + n))
        Precomputed covariance matrix V_{N,0} (independent of (u,w)) computed
        by the function compute_V_matrix_for_fourier.

    g_tilde : ndarray of shape (N*(d**2 + n),).
        Precomputed vector tilde{g}_{N,0} (independent of (u,w)).

    Notes
    -----
    c denotes the number of pairs (u_array[e], w_array[e]) for which
    the Fourier–Laplace transform is evaluated. For every
    e = 0, ..., c - 1, u_array[e] has shape (d,) and w_array[e]
    has shape (d, d).

    Returns
    -------
    fourier_laplace_values : 1D numpy array of complex numbers
        Values of the Fourier-Laplace transform
        exp( <Psi g_tilde, g_tilde> ) / sqrt(det(Phi)) for each pair (u, w).
    """
    u_array = np.asarray(u_array)
    w_array = np.asarray(w_array)

    if u_array.shape[0] != w_array.shape[0]:
        raise ValueError("u_array and w_array must have the same first dimension (c).")

    c = u_array.shape[0]

    results = Parallel(n_jobs=-1)(
        delayed(_vss_fourier_laplace_discontinuous_eval_unique_det)(
            u_array[e],
            w_array[e],
            n, d, N, T, C, rho, R_W, V, g_tilde
        )
        for e in range(c)
    )

    values, det_values = zip(*results)
    return np.asarray(values), np.asarray(det_values)


def vss_fourier_laplace_det_lip_based(discontinuous_fourier_laplace_values: np.ndarray, det_values: np.ndarray):
    """
    Uses the Lipschitz-based algorithm to obtain the Fourier-Laplace
    values from the discontinuous Fourier-Laplace values.

    Parameters
    ----------
    discontinuous_fourier_laplace_values : array of complex numbers
        Obtained via the function vss_fourier_laplace_discontinuous_det

    det_values : array of complex numbers
        Obtained via the function vss_fourier_laplace_discontinuous_det

    Returns
    -------
    fourier_laplace_values : 1D numpy array of complex numbers
        Values of the Fourier-Laplace transform
        e^{i pi k} exp( <Psi(u,w) g_tilde, g_tilde> ) / sqrt(det(Phi(u,w))) for each pair (u, w).
    """
    # Argument of determinant
    arg = np.angle(det_values)

    # Detect jumps close to pi
    jumps = np.abs(arg[1:] - arg[:-1]) > np.pi
    bad_ind = np.where(jumps)[0]

    # Apply sign flips
    corrected_fourier_laplace_values = discontinuous_fourier_laplace_values.copy()
    for i in bad_ind.tolist():
        corrected_fourier_laplace_values[i+1:] = - corrected_fourier_laplace_values[i+1:]

    return corrected_fourier_laplace_values
