# Installation & GPU Setup

This repository contains tools and notebooks for Galactic Binary (GB) data analysis with LISA, accelerated using JAX.

---

## 1. Installation Procedure

### Option A: Google Colab

1. **Upload the Notebook**:
   - Open [Google Colab](https://colab.research.google.com/).
   - Click **File** > **Upload notebook** and upload `GBExploration.ipynb`.

2. **Upload the Utilities File**:
   - In the left sidebar of Colab, click the **Files** icon (📁).
   - Upload `GBEx_utils.py` directly into the session root folder (`/content`).

3. **Install Dependencies**:
   - Run the installation cell near the top of `GBExploration.ipynb` (uncomment the `%pip install` command in Cell 2) to install the required packages directly inline.


---

### Option B: Local Installation

Ensure you have **Python 3.10+** (Python 3.13 is recommended) and `git` installed.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/coen-rondeel/gb-basics.git
   cd gb-basics
   ```

2. **Install dependencies**:

   - **Using `uv` (Recommended):**
     ```bash
     # Install uv if not already installed: https://docs.astral.sh/uv/
     uv sync

     # Activate the virtual environment
     source .venv/bin/activate
     ```

   - **Using standard `venv` and `pip`:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate  # On Windows: .venv\Scripts\activate
     pip install --upgrade pip
     pip install -e .
     ```

---

## 2. Running on GPUs

The waveform calculations and likelihood evaluations in this repository are implemented using **JAX** (`jax.jit`, `jax.vmap`). When JAX detects an available GPU, it automatically offloads and compiles operations on the GPU without needing manual code changes.

### GPU on Google Colab

1. **Enable GPU Runtime:**
   - In the Colab top menu, go to **Runtime** > **Change runtime type**.
   - Under **Hardware accelerator**, select **T4 GPU** (or A100 / L4 if available).
   - Click **Save**.

2. **Install CUDA-enabled JAX:**
   Colab usually has CUDA pre-configured, but to ensure JAX is built with CUDA 12 support, run:
   ```python
   %pip install -U "jax[cuda12]"
   ```

3. **Verify GPU Detection:**
   Run the Imports cell (Cell 3) in `GBExploration.ipynb`, which includes:
   ```python
   print(f"Available devices: {jax.devices()}")
   ```
   You should see a CUDA device listed (e.g., `[cuda(id=0)]`).

---

### GPU on Local Machines

> **System Requirements:** Official JAX GPU acceleration requires an **NVIDIA GPU** running on **Linux** or **Windows via WSL2**. (Native Windows and macOS run on CPU; Apple Silicon Metal acceleration for JAX is experimental and not fully compatible with all 64-bit primitives).

1. **Install CUDA-enabled JAX:**
   Ensure you have compatible NVIDIA drivers installed on your host/WSL2 system, then install JAX with CUDA 12 wheels inside your virtual environment:

   - **With `uv`:**
     ```bash
     uv pip install -U "jax[cuda12]"
     ```

   - **With standard `pip`:**
     ```bash
     pip install -U "jax[cuda12]"
     ```

2. **Verify GPU Detection:**
   Check device availability from your terminal or in Python:
   ```bash
   python -c "import jax; print('Available devices:', jax.devices())"
   ```
   The output should confirm the CUDA device, e.g.:
   ```text
   Available devices: [cuda(id=0)]
   ```

3. **Optional: GPU Memory Configuration:**
   By default, JAX pre-allocates up to 75% of total GPU memory when the first operation runs. To enable dynamic memory allocation as needed (preventing out-of-memory errors when sharing the GPU):
   ```bash
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   ```
   Or limit the allocated memory fraction (e.g., 50%):
   ```bash
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50
   ```
