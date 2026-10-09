import numpy as np


def compute_V_matrix_for_mc(
    d,
    N,
    T,
    eta,
    C,
    rho,
    KK,
    SIG,
    MM,
):
    """
    Construct the covariance matrix V_{N,0} of the vectorized volatility
    matrices and the increments of B, where d<B>_t = rho dt and
    d<W,B>_t = C.T @ rho dt.

    Parameters
    ----------
    d : int
        Number of assets.

    N : int
        Number of time steps.

    T : float
        Maturity.

    eta : ndarray, shape (n, d)
        Volatility-of-volatility matrix.

    C : ndarray, shape (d, n)
        Matrix whose columns satisfy 0 <= C_k^T C_k <= 1 for all
        k in {1,...,n}, C_k being the k-th column of C.

    rho : ndarray, shape (d, d)
        Correlation matrix of B: rho = rho.T is positive semidefinite,
        has unit diagonal, and d<B>_t = rho dt.

    KK : ndarray, shape (N*d, N*n)
        Discretized kernel matrix K_{N,0}.

    SIG : ndarray, shape (N*n, N*n)
        Covariance matrix Sigma_{N,0} of the volatility factors Y.

    MM : ndarray, shape (N*d*d, N*n)
        Block-diagonal matrix M.

    Returns
    -------
    V : ndarray, shape (N*(d^2 + d), N*(d^2 + d))
        Covariance matrix V_{N,0}.
    """
    Nd = N * d
    dt = T / N
    I_N = np.eye(N)

    # Top-left block: M Σ M^T
    V11 = MM @ SIG @ MM.T                        # (Nd^2, Nd^2)

   # Kronecker blocks
    I_eta = np.kron(I_N, eta)                    # (Nn, Nd)
    I_Ct_rho = np.kron(I_N, C.T @ rho)           # (Nn, Nd)

   # Top-right block
    V12 = MM @ I_eta @ KK @ I_Ct_rho             # (Nd^2, Nd)

    # Bottom-left block (symmetry)
    V21 = V12.T                                  # (Nd, Nd^2)

    # Bottom-right block
    V22 = dt * np.kron(I_N, rho)

    # Assemble full matrix
    V = np.block([
        [V11, V12],
        [V21, V22],
    ])

    return V


def spectral_cholesky(V, tol=1e-12):
    """
    Spectral Cholesky factorization of a symmetric positive semi-definite matrix.

    - Enforces symmetry (even if V PSD in theory, it can be not symmetric due to numerical noise)
    - Discards small / negative eigenvalues due to numerical noise

    Parameters
    ----------
    V : ndarray, shape (m, m)
        Covariance matrix (should be symmetric PSD).

    tol : float
        Eigenvalue threshold.

    Returns
    -------
    L : ndarray, shape (m, r)
        Factor such that V ≈ L @ L.T.
        r is the numerical rank of V, i.e the number of eigvals > tol.
    """
    # Enforce symmetry
    V = 0.5 * (V + V.T)

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(V)

    # Numerical cleanup
    eigvals = np.maximum(eigvals, 0.0)

    mask = eigvals > tol
    if not np.any(mask):
        raise ValueError("Covariance matrix is numerically zero.")

    L = eigvecs[:, mask] @ np.diag(np.sqrt(eigvals[mask]))
    return L
