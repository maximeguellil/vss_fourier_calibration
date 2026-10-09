import numpy as np
from scipy.special import loggamma

from .call_put import call_put_price_fourier


def basket_call_put_price_fourier(
    joint_fourier_laplace_values,
    marginal_fourier_laplace_values,
    S_0,
    K_array,
    alpha,
    joint_sample_points,
    joint_weights,
    marginal_sample_points,
    marginal_weights,
    option_type,
):
    """
    Computes European call and/or put prices on the two-dimensional arithmetic
    basket (S_T^1 + S_T^2) / 2, using Fourier inversion and Gauss-Laguerre
    quadrature.

    The basket-put pricing identity (zero rates and no dividends) is
        P_b(K) = I_b(K) + (P_1(2*K) + P_2(2*K)) / 2 - K,

    where
        I_b(K) = 2 / (2*pi)^2 * int_0^inf int_0^inf
                 Re(F_K(x_1, x_2) + F_K(x_1, -x_2)) dx_1 dx_2,

        F_K(x_1, x_2)
            = exp(i*<z,s>) * G_hat_K(z) * xi(-i*z),
        z = (x_1 + i*alpha_1, x_2 + i*alpha_2),
        s = (-log(S_0^1), -log(S_0^2)),

    and
        G_hat_K(z)
            = K * (2*K)^(i*(z_1 + z_2))
              * Gamma(i*z_1) * Gamma(i*z_2)
              / Gamma(2 + i*(z_1 + z_2))

    is the meromorphic continuation of the Fourier transform of the basket-put
    payoff. The two missing quadrants are recovered by complex conjugation.

    Here
        xi(u) = E[exp(u^T log(S_T / S_0))]

    is the joint Fourier-Laplace transform of the log-returns. The marginal
    puts P_1(2*K) and P_2(2*K) are computed with call_put_price_fourier
    from call_put.py.

    Basket-call prices are obtained from call-put parity:
        C_b(K) - P_b(K) = (S_0^1 + S_0^2) / 2 - K.

    Assumptions:
    - zero interest rates,
    - no dividends,
    - an equally weighted two-dimensional arithmetic basket.

    Parameters
    ----------
    joint_fourier_laplace_values : np.ndarray
        Joint Fourier-Laplace-transform values with shape (2, m, m). For
        positive joint_sample_points x_p and x_q,

            joint_fourier_laplace_values[0, p, q]
                = xi(alpha_1 - i*x_p, alpha_2 - i*x_q),

            joint_fourier_laplace_values[1, p, q]
                = xi(alpha_1 - i*x_p, alpha_2 + i*x_q).

    marginal_fourier_laplace_values : np.ndarray
        Marginal Fourier-Laplace-transform values with shape (2, m_marginal):

            marginal_fourier_laplace_values[0, r]
                = xi(1/2 + i*u_r, 0),

            marginal_fourier_laplace_values[1, r]
                = xi(0, 1/2 + i*u_r),

        where u_r = marginal_sample_points[r].

    S_0 : 1D numpy array of float
        Initial asset prices [S_0^1, S_0^2].

    K_array : 1D numpy array of float
        Array of basket strike prices.

    alpha : 1D numpy array of float
        Positive contour shift [alpha_1, alpha_2], satisfying
        alpha_1 + alpha_2 < 1.

    joint_sample_points : 1D numpy array of float
        Positive Gauss-Laguerre points used in the joint integral.

    joint_weights : 1D numpy array of float
        Gauss-Laguerre weights corresponding to joint_sample_points.

    marginal_sample_points : 1D numpy array of float
        Positive Gauss-Laguerre points used in the marginal Lewis formulas.

    marginal_weights : 1D numpy array of float
        Gauss-Laguerre weights corresponding to marginal_sample_points.

    option_type : {"call", "put", "both"}
        Specifies which basket-option prices to return:
        - "call": return only basket-call prices,
        - "put": return only basket-put prices,
        - "both": return both basket-put and basket-call prices.

    Returns
    -------
    call_prices : np.ndarray
        Returned if option_type == "call".

    put_prices : np.ndarray
        Returned if option_type == "put".

    put_prices, call_prices : tuple of np.ndarray
        Returned if option_type == "both".
    """
    if option_type not in {"call", "put", "both"}:
        raise ValueError("option_type must be one of {'call', 'put', 'both'}")

    S_0 = np.asarray(S_0, dtype=float)
    K_array = np.atleast_1d(np.asarray(K_array, dtype=float))
    alpha = np.asarray(alpha, dtype=float)
    joint_sample_points = np.asarray(joint_sample_points, dtype=float)
    joint_weights = np.asarray(joint_weights, dtype=float)
    marginal_sample_points = np.asarray(marginal_sample_points, dtype=float)
    marginal_weights = np.asarray(marginal_weights, dtype=float)
    joint_fourier_laplace_values = np.asarray(
        joint_fourier_laplace_values,
        dtype=complex,
    )
    marginal_fourier_laplace_values = np.asarray(
        marginal_fourier_laplace_values,
        dtype=complex,
    )

    # Basic tests on shape/dimension of arrays.
    if S_0.shape != (2,):
        raise ValueError(f"S_0 must have shape (2,), got {S_0.shape}")
    if K_array.ndim != 1:
        raise ValueError("K_array must be a 1D array")
    if alpha.shape != (2,):
        raise ValueError(f"alpha must have shape (2,), got {alpha.shape}")
    if np.any(~np.isfinite(S_0)) or np.any(S_0 <= 0.0):
        raise ValueError("The two initial prices must be finite and positive")
    if np.any(~np.isfinite(K_array)) or np.any(K_array <= 0.0):
        raise ValueError("All strike prices must be finite and positive")
    if np.any(~np.isfinite(alpha)) or np.any(alpha <= 0.0):
        raise ValueError("alpha_1 and alpha_2 must be finite and positive")
    if np.sum(alpha) >= 1.0:
        raise ValueError("alpha_1 + alpha_2 must be strictly smaller than 1")
    if joint_sample_points.ndim != 1 or joint_weights.ndim != 1:
        raise ValueError("joint_sample_points and joint_weights must be 1D arrays")
    if len(joint_sample_points) != len(joint_weights):
        raise ValueError("joint_sample_points and joint_weights must have the same length")
    if np.any(~np.isfinite(joint_sample_points)) or np.any(joint_sample_points <= 0.0):
        raise ValueError("joint_sample_points must be finite and strictly positive")
    if np.any(~np.isfinite(joint_weights)):
        raise ValueError("joint_weights must be finite")
    if marginal_sample_points.ndim != 1 or marginal_weights.ndim != 1:
        raise ValueError("marginal_sample_points and marginal_weights must be 1D arrays")
    if len(marginal_sample_points) != len(marginal_weights):
        raise ValueError("marginal_sample_points and marginal_weights must have the same length")
    if np.any(~np.isfinite(marginal_sample_points)) or np.any(marginal_sample_points <= 0.0):
        raise ValueError("marginal_sample_points must be finite and strictly positive")
    if np.any(~np.isfinite(marginal_weights)):
        raise ValueError("marginal_weights must be finite")

    number_of_joint_points = len(joint_sample_points)
    expected_joint_shape = (
        2,
        number_of_joint_points,
        number_of_joint_points,
    )
    if joint_fourier_laplace_values.shape != expected_joint_shape:
        raise ValueError(
            "joint_fourier_laplace_values must have shape "
            f"{expected_joint_shape}, got {joint_fourier_laplace_values.shape}"
        )

    number_of_marginal_points = len(marginal_sample_points)
    expected_marginal_shape = (2, number_of_marginal_points)
    if marginal_fourier_laplace_values.shape != expected_marginal_shape:
        raise ValueError(
            "marginal_fourier_laplace_values must have shape "
            f"{expected_marginal_shape}, got {marginal_fourier_laplace_values.shape}"
        )

    doubled_strikes = 2.0 * K_array

    put_prices_1 = call_put_price_fourier(
        fourier_laplace_values=marginal_fourier_laplace_values[0],
        S_0=S_0[0],
        K_array=doubled_strikes,
        sample_points=marginal_sample_points,
        weights=marginal_weights,
        option_type="put",
    )

    put_prices_2 = call_put_price_fourier(
        fourier_laplace_values=marginal_fourier_laplace_values[1],
        S_0=S_0[1],
        K_array=doubled_strikes,
        sample_points=marginal_sample_points,
        weights=marginal_weights,
        option_type="put",
    )

    x_1, x_2 = np.meshgrid(
        joint_sample_points,
        joint_sample_points,
        indexing="ij",
    )

    corrected_weights = joint_weights * np.exp(joint_sample_points)
    weights_matrix = corrected_weights[:, np.newaxis] * corrected_weights[np.newaxis, :]
    s = -np.log(S_0)

    basket_integrals = np.zeros(len(K_array), dtype=complex)

    for quadrant, sign_2 in enumerate((1.0, -1.0)):
        z_1 = x_1 + 1j * alpha[0]
        z_2 = sign_2 * x_2 + 1j * alpha[1]
        z_sum = z_1 + z_2

        log_gamma_ratio = (
            loggamma(1j * z_1)
            + loggamma(1j * z_2)
            - loggamma(2.0 + 1j * z_sum)
        )

        G_hat = K_array[np.newaxis, np.newaxis, :] * np.exp(
            1j
            * z_sum[:, :, np.newaxis]
            * np.log(2.0 * K_array[np.newaxis, np.newaxis, :])
            + log_gamma_ratio[:, :, np.newaxis]
        )

        phase = np.exp(1j * (z_1 * s[0] + z_2 * s[1]))
        integrand = (
            phase[:, :, np.newaxis]
            * G_hat
            * joint_fourier_laplace_values[quadrant, :, :, np.newaxis]
        )

        basket_integrals += np.sum(
            weights_matrix[:, :, np.newaxis] * integrand,
            axis=(0, 1),
        )

    basket_integrals = 2.0 * basket_integrals.real / (2.0 * np.pi) ** 2

    put_prices = basket_integrals + 0.5 * (put_prices_1 + put_prices_2) - K_array

    if option_type == "put":
        return put_prices

    basket_spot = 0.5 * (S_0[0] + S_0[1])
    call_prices = put_prices + basket_spot - K_array

    if option_type == "call":
        return call_prices
    else:
        return put_prices, call_prices
