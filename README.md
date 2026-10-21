# VSS Fourier calibration

Research code and two-year calibration notebooks for multidimensional Volterra
Stein--Stein models. The repository contains Fourier and Monte Carlo pricing
code, calibration notebooks, numerical sanity checks and the figures produced
by the retained two-year analyses.

Market-data files are intentionally excluded from the repository and from its
complete Git history.

## Repository contents

- `src/vss`: reusable calibration, Fourier, kernel, market-data and Monte Carlo
  code, including Gaussian common/idiosyncratic jump transforms and
  simulations;
- `vss_with_jump_implementation.md`: model specification and implementation
  guide for the Gaussian jump extension;
- `vss_calibration_CACIB_SP500_SX5E_skew_vars_vols_2y.ipynb`: sequential
  variance-swap, ATM-skew and volatility-swap calibrations;
- `vss_calibration_CACIB_SP500_SX5E_skew_vars_cvc_2y.ipynb`: sequential
  variance-swap, ATM-skew and Call-vs-Call calibrations;
- `vss_joint_calibration_CACIB_SP500_SX5E_skew_vars_cvc_2y.ipynb`: joint
  ATM-skew and Call-vs-Call calibrations after the variance-swap construction;
- `vss_fourier_sanity_check.ipynb`: Fourier-versus-Monte-Carlo numerical checks;
- `vss_fractional_atm_skew.ipynb`: fractional-kernel ATM-skew study;
- `Plots ...`: figures associated with the retained notebooks.

All CACIB market-data files, ten-year calibration notebooks and unrelated
experiments are intentionally outside the repository scope. Authorized users
must provide the calibration inputs locally under
`Market datas/CACIB/processed`; that directory is ignored by Git.

## Installation with uv

Install [uv](https://docs.astral.sh/uv/), clone the repository and run:

```bash
uv sync --frozen
```

The project uses Python 3.12, recorded in `.python-version`. The package in
`src/vss` is installed into the project environment by `uv sync`.

For Jupyter or an editor that discovers kernels by name, register the project
kernel once:

```bash
uv run python -m ipykernel install --user \
  --name volterra-stein-stein \
  --display-name "Volterra Stein-Stein"
```

In VS Code or another notebook editor, selecting `.venv/bin/python` directly is
also sufficient.

## Working safely

The `main` branch is the reviewed reference state. Repository policy is to work
on a topic branch and open a pull request:

```bash
git switch main
git pull --ff-only
git switch -c student/short-description
```

Do not add raw market data, ten-year notebooks or files outside the documented
scope.

The complete calibrations are intentionally not run in continuous integration
because they are computationally expensive. Automated checks validate the
environment, Python sources, notebook structure and repository scope.
