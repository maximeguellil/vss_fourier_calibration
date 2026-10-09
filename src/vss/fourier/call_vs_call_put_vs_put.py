import numpy as np

from .basket_call_put import basket_call_put_price_fourier
from .call_put import call_put_price_fourier


def call_vs_call_put_vs_put_price_fourier(
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
    Computes call-vs-call and/or put-vs-put prices for a vector of strike
    prices, using marginal Fourier prices and arithmetic-basket Fourier prices.

    The pricing identities (zero rates and no dividends) are
        CVC(K) = (C_1(K) + C_2(K)) / 2 - C_b(K),
        PVP(K) = (P_1(K) + P_2(K)) / 2 - P_b(K),

    where C_b and P_b are call and put prices on the arithmetic basket
        (S_T^1 + S_T^2) / 2.

    The two payoffs are equal pathwise when the individual options and the
    basket option have the same strike and maturity. They are computed from
    their respective call and put decompositions to provide a numerical
    consistency check.

    Here
        xi(u) = E[exp(u^T log(S_T / S_0))]

    is the joint Fourier-Laplace transform of the log-returns.

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
        Array of common strike prices for the individual and basket options.

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

    option_type : {"call_vs_call", "put_vs_put", "both"}
        Specifies which prices to return:
        - "call_vs_call": return only call-vs-call prices,
        - "put_vs_put": return only put-vs-put prices,
        - "both": return both put-vs-put and call-vs-call prices.

    Returns
    -------
    call_vs_call_prices : np.ndarray
        Returned if option_type == "call_vs_call".

    put_vs_put_prices : np.ndarray
        Returned if option_type == "put_vs_put".

    put_vs_put_prices, call_vs_call_prices : tuple of np.ndarray
        Returned if option_type == "both".
    """
    if option_type not in {"call_vs_call", "put_vs_put", "both"}:
        raise ValueError(
            "option_type must be one of "
            "{'call_vs_call', 'put_vs_put', 'both'}"
        )

    S_0 = np.asarray(S_0, dtype=float)
    K_array = np.atleast_1d(np.asarray(K_array, dtype=float))
    marginal_fourier_laplace_values = np.asarray(
        marginal_fourier_laplace_values,
        dtype=complex,
    )

    component_option_type = {
        "call_vs_call": "call",
        "put_vs_put": "put",
        "both": "both",
    }[option_type]

    basket_prices = basket_call_put_price_fourier(
        joint_fourier_laplace_values=joint_fourier_laplace_values,
        marginal_fourier_laplace_values=marginal_fourier_laplace_values,
        S_0=S_0,
        K_array=K_array,
        alpha=alpha,
        joint_sample_points=joint_sample_points,
        joint_weights=joint_weights,
        marginal_sample_points=marginal_sample_points,
        marginal_weights=marginal_weights,
        option_type=component_option_type,
    )

    prices_1 = call_put_price_fourier(
        fourier_laplace_values=marginal_fourier_laplace_values[0],
        S_0=S_0[0],
        K_array=K_array,
        sample_points=marginal_sample_points,
        weights=marginal_weights,
        option_type=component_option_type,
    )

    prices_2 = call_put_price_fourier(
        fourier_laplace_values=marginal_fourier_laplace_values[1],
        S_0=S_0[1],
        K_array=K_array,
        sample_points=marginal_sample_points,
        weights=marginal_weights,
        option_type=component_option_type,
    )

    if option_type == "call_vs_call":
        return 0.5 * (prices_1 + prices_2) - basket_prices
    elif option_type == "put_vs_put":
        return 0.5 * (prices_1 + prices_2) - basket_prices
    else:
        put_prices_1, call_prices_1 = prices_1
        put_prices_2, call_prices_2 = prices_2
        basket_put_prices, basket_call_prices = basket_prices

        call_vs_call_prices = (
            0.5 * (call_prices_1 + call_prices_2) - basket_call_prices
        )
        put_vs_put_prices = (
            0.5 * (put_prices_1 + put_prices_2) - basket_put_prices
        )

        return put_vs_put_prices, call_vs_call_prices
