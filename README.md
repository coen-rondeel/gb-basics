# ResponseRequirements

This repository contains computational tools for performing a galactic binary analysis within LISA.

## Installation
Follow these steps to set up the project on your local machine:

**Install dependencies**:
Ensure you have conda installed. Then run:
```bash
conda create -n lisa_env -c conda-forge -y python=3.12
conda activate lisa_env
pip install --upgrade pip && pip cache purge && pip install corner healpy eryn lisaorbits segwo pytdi jaxgb lisaconstants ipywidgets jupyter "numpy==2.3.3"
```

If the environment already exists, just activate it and make sure `ipywidgets` is installed:
```bash
conda activate lisa_env
pip install ipywidgets
```

Then open `GBExploration.ipynb` in Jupyter or VS Code and select the `lisa_env` kernel. The notebook imports its helper functions from `GBEx_utils.py`, which must stay in the same folder.

Optional: click-to-select in the interactive sky map needs `pip install ipympl` and `%matplotlib widget` in the notebook (on Colab also run `from google.colab import output; output.enable_custom_widget_manager()`). Without it the map uses sliders.

## Running on Google Colab

1. Push this folder (`GBExploration.ipynb` and `GBEx_utils.py`) to a GitHub repository.
2. Open the notebook in Colab: `https://colab.research.google.com/github/<USER>/<REPO>/blob/main/gb-basics/GBExploration.ipynb`
   (or Colab -> File -> Open notebook -> GitHub tab).
3. In the first code cell, uncomment the `%pip install ...` line and the `!wget ...` line. Replace `<USER>/<REPO>` in the `wget` URL with your repository so that `GBEx_utils.py` is downloaded next to the notebook.
4. Run the install cell, then **Runtime -> Restart session**, and run the notebook from the top.
5. A GPU is not required. The CPU runtime is enough, but the MCMC cells take several minutes; reduce `N_ITER` and `N_BURN` for a quick pass.

Alternative to `wget`: upload `GBEx_utils.py` through the Colab file browser (left sidebar), or `git clone` the repository and `%cd` into `gb-basics`.

## Notebook outline

1. The source: what a galactic binary is, and how its parameters map to a signal.
2. The data: signal + noise.
3. Matched-filter and intrinsic SNR across frequency and sky.
4. Likelihood, priors, and fitting one binary with known noise.
5. Extra: fitting the noise as well.

Change `SOURCE_NAME` near the top of the notebook to rerun the whole analysis for another reference binary.
