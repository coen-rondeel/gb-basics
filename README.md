# Galactic Binary LISA Tutorial

This repository contains computational tools for performing a galactic binary analysis within LISA.

## Run locally

Keep `GBExploration.ipynb` and `GBEx_utils.py` in the same folder. Choose either environment setup below, then open the notebook in Jupyter or VS Code and select the environment's Python kernel.

### Option 1: Conda

Create and activate the environment, then install the notebook dependencies:
```bash
conda create -n lisa_env -c conda-forge -y python=3.12
conda activate lisa_env
python -m pip install --upgrade pip
python -m pip install corner healpy eryn lisaorbits segwo pytdi jaxgb lisaconstants ipywidgets jupyter "numpy==2.3.3"
```

### Option 2: Python virtual environment

Use Python 3.12 and the built-in `venv` module:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install corner healpy eryn lisaorbits segwo pytdi jaxgb lisaconstants ipywidgets jupyter "numpy==2.3.3"
```

In either local environment, click-to-select in the interactive sky map is optional. Enable it with:
```bash
python -m pip install ipympl
```
Then run `%matplotlib widget` in a notebook cell. Without this backend, the sky map uses sliders instead.

## Run on Google Colab

1. Push `GBExploration.ipynb` and `GBEx_utils.py` to a GitHub repository.
2. Open the notebook using this link template, replacing the placeholders with your GitHub username and repository name:

   `https://colab.research.google.com/github/<GITHUB_USER>/<REPOSITORY>/blob/main/gb-basics/GBExploration.ipynb`

   Alternatively, use **Colab -> File -> Open notebook -> GitHub** and select the notebook.
3. In the first notebook cell, uncomment the `%pip install ...` and `!wget ...` lines. Replace the placeholders in the `wget` URL with your GitHub username and repository so `GBEx_utils.py` is downloaded next to the notebook.
4. Run the installation cell, choose **Runtime -> Restart session**, then run the notebook from the top.
5. A GPU is not required. The CPU runtime is sufficient, though MCMC cells take several minutes; reduce `N_ITER` and `N_BURN` for a quicker run.

For click-to-select on the interactive sky map in Colab, install `ipympl`, run `%matplotlib widget`, then enable the Colab widget manager:
```python
from google.colab import output
output.enable_custom_widget_manager()
```
Alternatively, upload `GBEx_utils.py` using the Colab file browser or clone the repository and change into `gb-basics`.

## Notebook outline

1. The source: what a galactic binary is, and how its parameters map to a signal.
2. The data: signal + noise.
3. Matched-filter and intrinsic SNR across frequency and sky.
4. Likelihood, priors, and fitting one binary with known noise.
5. Extra: fitting the noise as well.

Change `SOURCE_NAME` near the top of the notebook to rerun the whole analysis for another reference binary.
