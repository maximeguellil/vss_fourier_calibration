# Implementation of the VSS Model with Gaussian Jumps

## Objective

This extension adds common and idiosyncratic Gaussian jumps to the
two-asset diagonal Volterra Stein--Stein model. It preserves the existing
diffusion code and separates three responsibilities:

1. computing and validating the jump parameters;
2. computing the Fourier transform of the jump component;
3. simulating the jump component with Monte Carlo and combining it with the
   diffusion component.

The intended calibration is performed using Fourier methods. Monte Carlo can
then be used as an independent validation method.

## Implemented Model

For the two assets, the counting processes are

\[
N_t^1=N_t^c+N_t^{1,\perp},\qquad
N_t^2=N_t^c+N_t^{2,\perp},
\]

with respective intensities

\[
\lambda_c,\qquad \lambda_1-\lambda_c,\qquad
\lambda_2-\lambda_c,
\]

and the constraint

\[
0\leq\lambda_c\leq\min(\lambda_1,\lambda_2).
\]

The idiosyncratic marks are Gaussian with unit variance. The common marks
have means \(m_J^1,m_J^2\), unit variances, and correlation \(\rho_J\). The
jump process for the logarithm of asset \(i\) is therefore constructed from

\[
J_t^i=
\sum_{k=1}^{N_t^c}\zeta_k^{i,c}
+
\sum_{k=1}^{N_t^{i,\perp}}\zeta_k^{i,\perp}.
\]

The marginal normalization is

\[
\alpha_i=
\frac{1}{\sqrt{1+\lambda_i\left(1+(m_J^i)^2\right)}}.
\]

The diffusion loading and the jump loading are both
\(\alpha_i g_0^i(t)\). The log compensator is

\[
d\kappa_J^i(t)=
\lambda_i\left[
\exp\left(
\alpha_i g_0^i(t)m_J^i
+\frac12\alpha_i^2(g_0^i(t))^2
\right)-1
\right]dt.
\]

This construction preserves the marginal variance swap:

\[
\frac{d}{dt}\mathbb E[\langle\log S^i\rangle_t]
=(g_0^i(t))^2.
\]

## Added Directory Structure

```text
src/vss/
├── jumps/
│   ├── __init__.py
│   └── parameters.py
├── fourier/
│   └── gaussian_jumps.py
└── monte_carlo/
    ├── simulate_gaussian_jumps.py
    └── log_price_and_volatility_paths_with_jumps.py
```

Two existing Monte Carlo pricers have also been extended:

```text
src/vss/monte_carlo/skew.py
src/vss/monte_carlo/call_vs_call_put_vs_put.py
```

When no jump arguments are supplied, they follow exactly their original
diffusion-only path.

## Jump Parameters

The file `src/vss/jumps/parameters.py` contains the computations and
validations specific to the jump parameters:

- `gaussian_jump_normalization` computes \(\alpha_i\) for one asset;
- `gaussian_jump_normalizations` performs the same calculation for several
  assets;
- `validate_common_idiosyncratic_jump_parameters` validates the intensities,
  means, \(\lambda_c\), and \(\rho_J\);
- `gaussian_jump_quadratic_variation_matrix` computes the matrix \(Q_J\)
  defined by \(\mathbb E[d[J]_t]=Q_Jdt\).

In the two-dimensional case,

\[
Q_J=
\begin{pmatrix}
\lambda_1(1+(m_J^1)^2) &
\lambda_c(m_J^1m_J^2+\rho_J)\\
\lambda_c(m_J^1m_J^2+\rho_J) &
\lambda_2(1+(m_J^2)^2)
\end{pmatrix}.
\]

## Fourier Transforms

The file `src/vss/fourier/gaussian_jumps.py` exposes:

- `marginal_gaussian_jump_transform`;
- `joint_gaussian_jump_transform`.

These functions return only the transform factor associated with the
compensated jumps. By independence, the full transform is

\[
\Phi_{\log S}(u)
=
\Phi_{\mathrm{diffusion}}(u;\alpha g_0)
\Phi_{\mathrm{jumps}}(u;g_0,\lambda,m_J,\lambda_c,\rho_J).
\]

The existing diffusion routines, notably the computation of the \(M\) and
\(V\) matrices and the VSS transform, have not been modified.

### Marginal Transform

On an interval where \(h_i(t)=\alpha_i g_0^i(t)\) is constant, the exponent is

\[
\lambda_i\left[
\exp\left(uh_im_J^i+\frac12u^2h_i^2\right)-1
-u\left(
\exp\left(h_im_J^i+\frac12h_i^2\right)-1
\right)
\right].
\]

### Joint Transform

The joint transform adds:

- a common contribution with intensity \(\lambda_c\) and correlated Gaussian
  marks;
- two idiosyncratic contributions with intensities
  \(\lambda_i-\lambda_c\);
- the two marginal compensators.

The two \(g_0^i\) curves are aligned on the union of their boundaries. Since
they are piecewise constant, all time integrals are exact sums. The last value
is extrapolated flat when the maturity exceeds the last supplied boundary.

### Usage with the Existing Fourier Code

The diffusion computation must receive the normalized curve:

```python
from vss.jumps.parameters import gaussian_jump_normalizations

alpha = gaussian_jump_normalizations(jump_intensities, jump_means)
diffusive_marginal_g0 = marginal_g0 * alpha[None, :]
```

The `M` matrix is then constructed with `diffusive_marginal_g0`. After the
diffusion transform has been computed, the jump factor is multiplied
pointwise:

```python
from vss.fourier.gaussian_jumps import joint_gaussian_jump_transform

jump_transform = joint_gaussian_jump_transform(
    u=u_array,
    T=T,
    time_boundaries=(spx_boundaries, sx5e_boundaries),
    g0_values=(spx_g0_values, sx5e_g0_values),
    jump_intensities=jump_intensities,
    jump_means=jump_means,
    common_jump_intensity=lambda_c,
    jump_size_correlation=rho_J,
)
full_transform = diffusive_transform * jump_transform
```

The marginal factor is used in the same way for skew calibration.

## Monte Carlo Simulation

### Jump Simulation

The file `src/vss/monte_carlo/simulate_gaussian_jumps.py` contains:

- `simulate_marginal_gaussian_jump_log_returns`;
- `simulate_common_idiosyncratic_gaussian_jump_log_returns`.

For a jump count \(N\), the sum of the Gaussian marks is simulated directly
as

\[
Nm_J+\sqrt{N}Z.
\]

It is therefore unnecessary to simulate every mark separately. In the common
case, the two normal variables are correlated through

\[
Z_1=U_1,\qquad
Z_2=\rho_JU_1+\sqrt{1-\rho_J^2}\,U_2.
\]

Each Monte Carlo time step is split at the boundaries of the \(g_0^i\) curves.
This avoids approximating a piecewise-constant curve with a value taken from
the wrong side of a boundary.

With the antithetic option, both paths in a pair share the same Poisson counts
and use centered Gaussian variables with opposite signs. Pair averages remain
the independent observations used to calculate standard errors.

### Combining Diffusion and Jumps

`simulate_log_prices_and_sigmas_with_jumps_batch` first calls
`simulate_log_prices_and_sigmas_batch`, then simulates the jumps using an
independent random stream and adds the two cumulative log returns.

The diffusion seed is left unchanged. A reproducible child seed is derived
for the jumps. Consequently, the diffusion random numbers remain comparable
with those of the model without jumps.

The `MM` matrix passed to this function must already have been constructed
using the diffusion curve \(\alpha_i g_0^i\). The curves supplied through the
`jump_g0_values` arguments remain the total \(g_0^i\) curves: normalization is
applied inside the jump simulator.

## Extended Monte Carlo Pricers

`mc_skew` now accepts the following four optional arguments:

```text
jump_time_boundaries
jump_g0_values
jump_intensity
jump_mean
```

`mc_call_vs_call_put_vs_put_price` accepts the following six optional
arguments:

```text
jump_time_boundaries
jump_g0_values
jump_intensities
jump_means
common_jump_intensity
jump_size_correlation
```

To avoid ambiguous partial configurations, all arguments in the relevant
group must be supplied together. If none is supplied, the previous behavior
is preserved.

## Recommended Calibration Procedure

The intended organization for the future notebook is as follows:

1. calibrate the variance swap curves \(g_0^i\);
2. use Fourier methods to calibrate the marginal diffusion parameters and
   \((\lambda_i,m_J^i)\);
3. construct the diffusion matrices using \(\alpha_i g_0^i\);
4. use Fourier methods to calibrate the CVCs by optimizing the joint
   parameters \((\rho,\lambda_c,\rho_J)\);
5. validate the resulting prices using Monte Carlo.

This procedure avoids directly optimizing a realization of a Poisson process
as a function of \(\lambda_c\). The Fourier transform is smooth in this
parameter, whereas a simulation with common random numbers is a step function
of the intensity.

## Added Tests

The tests are distributed across:

```text
tests/test_gaussian_jump_parameters.py
tests/test_gaussian_jump_fourier.py
tests/test_gaussian_jump_monte_carlo.py
```

They notably verify:

- the quadratic-variation normalization;
- the theoretical \(Q_J\) matrix;
- rejection of inadmissible joint parameters;
- the identities \(\Phi_J(0)=1\) and \(\Phi_J(e_i)=1\);
- consistency between the marginal and joint transforms;
- exact disappearance of jumps when the intensities are zero;
- agreement between the Fourier transform and Monte Carlo simulation;
- the theoretical covariance of common jumps;
- correct addition of the diffusion and jump log returns;
- non-regression of the existing Monte Carlo tests.

The suite can be run from the project root with:

```bash
PYTHONPATH=src UV_CACHE_DIR=/tmp/vss-uv-cache \
    uv run python -m unittest discover -s tests -v
```
