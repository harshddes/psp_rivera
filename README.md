# psp_rivera

This repository contains a script `mva_1.py` that performs a minimum
variance analysis on data from the Parker Solar Probe. The script
requires a number of Python packages which are not included in this
repository.

## Installation

Create a virtual environment and install the dependencies from the
`requirements.txt` file:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the script

After installing the dependencies, execute the script with:

```bash
python mva_1.py
```

The script downloads PSP MAG-RTN data via the `pyspedas` package and
saves its results in the `output_rivera_mva` directory.

## Running in Google Colab

If you are using [Google Colab](https://colab.research.google.com/), you
can install the requirements and execute the script directly in a
notebook cell:

```python
!pip install -r requirements.txt
!python mva_1.py
```

The figures and `mva_results.npz` file will be written to a folder named
`output_rivera_mva` in the Colab working directory.
