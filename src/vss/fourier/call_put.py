import numpy as np


def call_put_price_fourier(
    fourier_laplace_values,
    S_0,
    K_array,
    sample_points,
    weights,
    option_type="both",
):
    """
    Computes the price of European call and/or put options for a vector of strike prices,
    using a Fourier inversion technique of Lewis type and Gauss–Laguerre quadrature.

    The pricing identity (Lewis formula, zero rates and no dividends) is
        C(K) = S_0
             - (sqrt(S_0*K) / pi) * int_0^inf
               Re( exp(i*u*log(S_0/K)) * xi(1/2 + i*u) / (u^2 + 1/4) ) du,

    where
        xi(z) = E[ exp( z * log(S_T / S_0) ) ]
    is the Fourier–Laplace transform of the log-return.

    Put prices are obtained from the same integral representation:
        P(K) = K
             - (sqrt(S_0*K) / pi) * int_0^inf
               Re( exp(i*u*log(S_0/K)) * xi(1/2 + i*u) / (u^2 + 1/4) ) du.

    Assumptions:
    - zero interest rates,
    - no dividends.

    Parameters
    ----------
    fourier_laplace_values : np.ndarray
        Values of xi(1/2 + i*u) evaluated at u = sample_points.

    S_0 : float
        Initial asset price.

    K_array : 1D numpy array of float
        Array of strike prices.

    sample_points : 1D numpy array of float
        Gauss–Laguerre quadrature points (positive).

    weights : 1D numpy array of float
        Gauss–Laguerre weights corresponding to each sample point.

    option_type : {"call", "put", "both"}, optional
        Specifies which option prices to return:
        - "call": return only call prices,
        - "put": return only put prices,
        - "both": return both put and call prices (default).

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

    # Compute log-moneyness values for each strike price
    k_array = np.log(S_0 / K_array)

    # Reshape arrays to enable broadcasting
    sample_points_2d = sample_points[:, np.newaxis]  # (n_points, 1)
    weights_2d = weights[:, np.newaxis]               # (n_points, 1)
    k_array_2d = k_array[np.newaxis, :]               # (1, n_strikes)

    # Gauss–Laguerre integrand
    integrand = np.exp(sample_points_2d) * (
        np.exp(1j * sample_points_2d * k_array_2d)
        * fourier_laplace_values[:, np.newaxis]
        / (sample_points_2d**2 + 0.25)
    ).real

    # Quadrature
    integrals = np.sum(weights_2d * integrand, axis=0)

    if option_type == "call":
        call_prices = S_0 - np.sqrt(S_0 * K_array) / np.pi * integrals
        return call_prices
    elif option_type == "put":
        put_prices = K_array - np.sqrt(S_0 * K_array) / np.pi * integrals
        return put_prices
    else:
        put_prices = K_array - np.sqrt(S_0 * K_array) / np.pi * integrals
        call_prices = S_0 - np.sqrt(S_0 * K_array) / np.pi * integrals
        return put_prices, call_prices
