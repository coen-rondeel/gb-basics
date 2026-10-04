# Galactic Binary LISA Tutorial

This repository contains computational tools for performing a galactic binary analysis within LISA.

## Run on Google Colab

Open the notebook in Colab: [GBExploration.ipynb](https://colab.research.google.com/github/coen-rondeel/gb-basics/blob/main/GBExploration.ipynb).

## Run locally

Choose either environment setup below, then open the notebook in Jupyter or VS Code and select the environment's Python kernel. Download the folder

```
git clone https://github.com/coen-rondeel/gb-basics.git
cd gb-basics
```

The source catalog is read directly from the bundled `VGB.hdf5` (via `h5py`); keep the file beside `GBEx_utils.py`.

### Option 1: Conda

Create and activate the environment, then install the notebook dependencies:
```bash
conda create -n lisa_env -c conda-forge -y python=3.12
conda activate lisa_env
python -m pip install --upgrade pip
python -m pip install h5py corner healpy eryn lisaorbits segwo pytdi jaxgb lisaconstants ipywidgets jupyter "numpy==2.3.3" ipympl
```

### Option 2: Python virtual environment

Use Python 3.12 and the built-in `venv` module:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install h5py corner healpy eryn lisaorbits segwo pytdi jaxgb lisaconstants ipywidgets jupyter "numpy==2.3.3" ipympl
```


## Notebook outline

1. The source: what a galactic binary is, and how its parameters map to a signal.
2. The data: signal + noise.
3. Matched-filter and intrinsic SNR across frequency and sky.
4. Likelihood, priors, and fitting one binary with known noise.
5. Extra: fitting the noise as well.

Change `SOURCE_NAME` near the top of the notebook to rerun the whole analysis for another reference binary.


