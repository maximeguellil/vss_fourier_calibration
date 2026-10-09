import numpy as np


def vol_swap_price_fourier(
    fourier_laplace_values,
    expected_integrated_variance,
    sample_points,
    weights,
):
    """
    Computes the price of a volatility swap of maturity T using a Fourier/Laplace inversion
    and Gauss–Laguerre quadrature.

    The pricing identity (fractional-moment representation) is
        VS_T = sqrt(int_0^T G_0(t)^2 dt) + (1 / (2*sqrt(pi))) * int_0^inf (exp(-s int_0^T G_0(t)^2 dt) - xi(s)) / s^(3/2) ds,
    where xi(s) = E[exp(-s * V_bar_T)] is the Laplace transform of the integrated variance

    Parameters
    ----------
    fourier_laplace_values : np.ndarray
        Values of xi(s) = E[exp(-s * V_bar_T)] evaluated at s = sample_points.

    expected_integrated_variance : float
        E[V_bar_T], available in closed form from the calibration
        function G_0:
        E[V_bar_T] = int_0^T G_0(s)^2 ds.

    sample_points : 1D numpy array of float
        Gauss-Laguerre quadrature points (strictly positive).

    weights : 1D numpy array of float
        Gauss-Laguerre weights corresponding to each sample point.

    Returns
    -------
    vol_swap_price : float
        Non-annualized volatility-swap price,
        E[sqrt(V_bar_T)].

        The corresponding annualized quote is
        E[sqrt(V_bar_T)] / sqrt(T), where T denotes the maturity.
        T is not an argument of this function, so the annualization
        must be performed outside this function.
    """
    a = float(expected_integrated_variance)
    if a < 0.0:
        raise ValueError("expected_integrated_variance must be non-negative")

    s = np.asarray(sample_points, dtype=float)
    w = np.asarray(weights, dtype=float)
    xi = np.asarray(fourier_laplace_values, dtype=complex)

    if s.ndim != 1 or w.ndim != 1 or xi.ndim != 1:
        raise ValueError("sample_points, weights, and fourier_laplace_values must be 1D arrays")
    if not (len(s) == len(w) == len(xi)):
        raise ValueError("sample_points, weights, and fourier_laplace_values must have the same length")
    if np.any(s <= 0.0):
        raise ValueError("sample_points must be strictly positive for the s^{-3/2} integrand")

    # Gauss–Laguerre approximates int_0^inf e^{-s} f(s) ds ≈ sum_i w_i f(s_i),
    # so we multiply by exp(s) to recover the bare integral of the residual integrand
    #   g(s) = (exp(-s*a) - xi(s)) / s^{3/2}   (regular at 0 and decaying at +inf),  a = E[V_bar_T].
    integrand = np.exp(s) * (np.exp(-s * a) - xi).real / (s ** 1.5)
    integral = np.sum(w * integrand)

    # Closed-form add-back sqrt(E[V_bar_T]) (the variance-swap strike).
    vol_swap_price = integral / (2.0 * np.sqrt(np.pi)) + np.sqrt(a)

    return float(vol_swap_price)
