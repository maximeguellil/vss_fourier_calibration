import numpy as np


def compute_tilde_g_N0(Y0, n, N, MM):
    """
    Construct the vector tilde{g}_{N,0} for t = 0.

    At t = 0, the conditional expectation g_0(t_{i-1}) equals Y0
    for all i = 1,...,N. Hence
        g_{N,0} = (Y0, ..., Y0) in R^{Nn},
    and
        tilde{g}_{N,0} = ( MM g_{N,0}, 0_{Nn} ).

    Parameters
    ----------
    Y0 : array_like, shape (n,)
        Initial volatility factor Y0.

    n : int
        Dimension of the volatility factor.

    N : int
        Number of time steps.

    MM : ndarray, shape (N*d*d, N*n)
        Block-diagonal matrix M_{N,0}.

    Returns
    -------
    tilde_g : ndarray, shape (N*(d^2 + n),)
        Vector tilde{g}_{N,0}.
    """
    Y0 = np.asarray(Y0).reshape(n)

    g_N0 = np.tile(Y0, N)          # (N*n,)
    Mg = MM @ g_N0                 # (N*d^2,)
    zeros = np.zeros(N * n)        # (N*n,)

    tilde_g = np.concatenate([Mg, zeros])
    return tilde_g


def compute_V_matrix_for_fourier(
    n,
    N,
    T,
    eta,
    R_W,
    KK,
    SIG,
    MM,
):
    """
    Construct the covariance matrix V_{N,0} of the vectorized volatility
    matrices and the increments of W, where d<W>_t = R_W dt.

    Parameters
    ----------
    n : int
        Dimension of the volatility factor.

    N : int
        Number of time steps.

    T : float
        Maturity.

    eta : ndarray, shape (n, d)
        Volatility-of-volatility matrix.

    R_W : ndarray, shape (n, n)
        Instantaneous covariance matrix of W, with
        R_W = C.T @ rho @ C + I_n - diag(diag(C.T @ C)).

    KK : ndarray, shape (N*d, N*n)
        Discretized kernel matrix K_{N,0}.

    SIG : ndarray, shape (N*n, N*n)
        Covariance matrix Sigma_{N,0} of the volatility factors Y.

    MM : ndarray, shape (N*d*d, N*n)
        Block-diagonal matrix M.

    Returns
    -------
    V : ndarray, shape (N*(d^2 + n), N*(d^2 + n))
        Covariance matrix V_{N,0}.
    """
    Nn = N * n
    dt = T / N
    I_N = np.eye(N)

    # Top-left block: M Σ M^T
    V11 = MM @ SIG @ MM.T                        # (Nd^2, Nd^2)

    # Kronecker blocks
    I_eta = np.kron(I_N, eta)                    # (Nn, Nd)

    # Top-right block
    V12 = MM @ I_eta @ KK @ np.kron(I_N, R_W)    # (Nd^2, Nn)

    # Bottom-left block (symmetry)
    V21 = V12.T                                  # (Nd, Nd^2)

    # Bottom-right block
    V22 = dt * np.kron(I_N, R_W)

    # Assemble full matrix
    V = np.block([
        [V11, V12],
        [V21, V22],
    ])

    return V
