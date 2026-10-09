import numpy as np


def _check_admissibility_Ud(u: np.ndarray, w: np.ndarray, d: int, atol: float = 1e-12) -> None:
    """
     Parameters
    ----------
    Check (u,w) in U_d:
      - Re(u) >= 0 componentwise
      - sum_i Re(u_i) <= 1
      - w is complex-symmetric
      - Re(w) is negative semidefinite

    Raises a ValueError if the condition is violated.
    """
    u = np.asarray(u)
    w = np.asarray(w)

    if u.shape != (d,):
        raise ValueError(f"u must have shape ({d},), got {u.shape}")
    if w.shape != (d, d):
        raise ValueError(f"w must have shape ({d},{d}), got {w.shape}")

    Re_u = np.real(u)
    if np.any(Re_u < -atol):
        raise ValueError("Re(u) must be componentwise nonnegative")
    if np.sum(Re_u) > 1.0 + atol:
        raise ValueError("Sum of Re(u) must be <= 1")

    Re_w = np.real(w)
    if not np.allclose(w, w.T, atol=atol):
        raise ValueError("w must be symmetric (complex-symmetric)")
    if np.any(np.linalg.eigvalsh(Re_w) > atol):
        raise ValueError("Re(w) must be negative semidefinite")


def _check_admissibility_C(C, tol=1e-12):
    """
    Check that each column C_k of C satisfies 0 <= C_k^T C_k <= 1.
    Raises a ValueError if the condition is violated.

    Parameters
    ----------
    C : ndarray, shape (d, n)
        Correlation matrix whose columns have norm at most one.

    tol : float
        Numerical tolerance.
    """
    C = np.asarray(C)
    if C.ndim != 2:
        raise ValueError("C must be a 2D array")

    column_norm_sq = np.sum(C * C, axis=0)  # ||C_k||^2

    for k, val in enumerate(column_norm_sq):
        if val < -tol:
            raise ValueError(
                f"Invalid correlation matrix: column {k} has negative norm squared "
                f"(C_k^T C_k = {val})."
            )
        if val > 1.0 + tol:
            raise ValueError(
                f"Invalid correlation matrix: column {k} has norm squared greater than 1 "
                f"(C_k^T C_k = {val})."
            )


def _check_admissibility_rho(rho, tol=1e-12):
    """Check that rho is a finite, symmetric positive semidefinite correlation matrix.

    Parameters
    ----------
    rho : ndarray, shape (d, d)
        Correlation matrix.

    tol : float
        Numerical tolerance.
    """
    rho = np.asarray(rho)
    if np.iscomplexobj(rho) or not np.issubdtype(rho.dtype, np.number):
        raise ValueError("rho must be a real-valued matrix")
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho must be a square matrix")
    if not np.all(np.isfinite(rho)):
        raise ValueError("rho must contain only finite values")
    if not np.allclose(rho, rho.T, atol=tol, rtol=0):
        raise ValueError("rho must be symmetric")
    if not np.allclose(np.diag(rho), 1.0, atol=tol, rtol=0):
        raise ValueError("rho must have a unit diagonal")
    if np.any(np.linalg.eigvalsh(0.5 * (rho + rho.T)) < -tol):
        raise ValueError("rho must be positive semidefinite")
