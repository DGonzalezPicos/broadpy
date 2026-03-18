# Installation Guide

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `matplotlib`, `astropy`

## Option 1: Install from source (recommended)

```bash
git clone https://github.com/DGonzalezPicos/broadpy
cd broadpy
pip install .
```

## Option 2: Conda environment

```bash
conda env create -f environment.yml
conda activate broadpy-env
pip install .
```

## Verify installation

```bash
python -c "from broadpy import InstrumentalBroadening, RotationalBroadening; print('broadpy import OK')"
```

## Run the docs quickstart notebook

From the repository root:

```bash
jupyter lab docs/instrumental_broadening_examples.ipynb
```
