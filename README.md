# broadpy

`broadpy` is a Python package for the broadening of spectral lines from instrumental and rotational effects. It is designed to be simple, fast and customizable.

## Features

- Calculate different broadening effects
- Plot spectral lines with various profiles
- Simple API for integration into larger projects

## Installation

You can install `broadpy` by cloning the repository and running `pip install .`:

```bash
git clone https://github.com/DGonzalezPicos/broadpy
cd broadpy
pip install .
```

## Usage

Here's an example of how to use BroadPy:

```python
from broadpy import InstrumentalBroadening
wave, flux = load_data()
flux_broadpy_fwhm = InstrumentalBroadening(wave, flux)(fwhm=3.0, kernel='gaussian')
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue.

## License

This project is licensed under the MIT License.

## Conda Environment Setup

To create a Conda environment for BroadPy, use the following command:

```bash
conda env create -f environment.yml
```

This will create an environment named `broadpy-env` with all necessary dependencies.

To activate the environment, use:

```bash
conda activate broadpy-env
```
