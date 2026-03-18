# BroadPy Documentation

`broadpy` provides fast, lightweight tools for spectral-line broadening:
- instrumental broadening (Gaussian, Lorentzian, Voigt, variable Gaussian)
- rotational broadening
- JWST NIRSpec broadening using polynomial resolution curves

## Quickstart

1. Install the package: see [`installation.md`](installation.md)
2. Run the example notebook: [`instrumental_broadening_examples.ipynb`](instrumental_broadening_examples.ipynb)

## Example Notebook

The docs notebook includes:
- constant-resolution instrumental broadening
- variable-resolution broadening with linearly increasing `R(λ)`
- JWST NIRSpec broadening (`g140h`)
- comparison plots and residual panels

Open it directly from the repository root:

```bash
jupyter lab docs/instrumental_broadening_examples.ipynb
```
