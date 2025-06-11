# CINN\_spline: Scalar Submodule

This submodule implements the **scalar** pipeline of the Conditional Invertible Neural Network (cINN) framework for modeling galaxy cluster merger properties. It covers:

1. **Data preprocessing** (`data_filter.py`)
2. **Data loading** (`data_loader.py`)
3. **Model definition** (`model.py`)
4. **Training loop** (`trainer.py`, `main.py`)
5. **Plotting utilities** (`plot_utils.py` and notebook scripts)

---

## Table of Contents

* [Installation](#installation)
* [Directory Structure](#directory-structure)
* [Usage](#usage)

  * [1. Data Preprocessing](#1-data-preprocessing)
  * [2. Creating DataLoaders](#2-creating-dataloaders)
  * [3. Model Architecture](#3-model-architecture)
  * [4. Training](#4-training)
  * [5. Plotting Results](#5-plotting-results)
* [Configuration](#configuration)
* [Contributing](#contributing)
* [License](#license)

---

## Installation

1. **Clone the repository**

   ```bash
   git clone <repo_url>
   cd CINN_spline/scalar
   ```

2. **Create & activate a Python environment** (recommended)

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

*Required libraries include:* `numpy`, `pandas`, `scikit-learn`, `torch`, `FrEIA`, `matplotlib`, `scipy`, `joblib`, `pyyaml`.

---

## Directory Structure

```text
scalar/
├── data_filter.py              # Preprocess raw CSVs into scaled datasets
├── data_loader.py              # Build PyTorch DataLoaders
├── model.py                    # Defines VBLinear, Subnet, transformations, CINN
├── trainer.py                  # Training logic, logging, checkpointing
├── main.py                     # CLI entry point for training runs
├── params.yaml                 # Hyperparameter configuration
├── plot_utils.py               # Common routines for posterior sampling & KDE
├── 1.posterior_distribution.ipynb  # Notebook generating Figure 1
├── 1.posterior_distribution.png   # Posterior distribution figure
├── 2.prediction_performance.ipynb # Notebook for performance evaluation
├── 2.prediction_performance1.png  # Performance heatmap (prior vs posterior)
├── 2.prediction_performance2.png  # MAP vs truth & error vs truth plots
├── 3.uncertainities.ipynb         # Notebook analyzing predictive uncertainties
├── 3.uncertainities.png           # Uncertainty vs error scatter
├── 4.cross_correlations.ipynb     # Notebook for cross-correlation analysis
├── 4.cross_correlations.png        # Cross-correlation heatmap
├── params.yaml                 # YAML file with model & training settings
├── processed_data/             # Generated: scaled X.csv, Y.csv, meta.csv, scalers
└── runs/                       # Output directory for checkpoints & logs
```

---

## Usage

### 1. Data Preprocessing

Merge observables & unobservables, drop missing targets, scale features:

```bash
python data_filter.py \
  --obs_csv observables1.csv \
  --unobs_csv unobservables1.csv \
  --out_dir processed_data
```

* **Input:** `observables1.csv`, `unobservables1.csv` (must share `HaloID, Snapshot`).
* **Outputs in `processed_data/`:**

  * `X.csv`: scaled observables
  * `Y.csv`: scaled targets
  * `meta.csv`: `HaloID, Snapshot`
  * `obs_scaler.pkl`, `tar_scaler.pkl`

### 2. Creating DataLoaders

```python
from data_loader import get_loaders
train_loader, val_loader = get_loaders(
    processed_dir="processed_data", 
    batch_size=512,
    val_frac=0.1,
)
```

* Splits by **unique** `HaloID` to avoid leakage.
* Returns PyTorch `DataLoader` with fields `.data` (targets) and `.cond` (observables).

### 3. Model Architecture

Defined in **`model.py`**:

* **`VBLinear`**: Bayesian linear layer with local reparameterization and optional MAP inference.
* **`Subnet`**: Configurable fully connected or Bayesian subnet for coupling blocks.
* **`LogTransformation`**: Simple invertible log/exp transform module.
* **`RationalQuadraticSplineBlock`**: Invertible coupling block using rational-quadratic splines.
* **`CINN`**: Builds a sequence of coupling blocks conditioned on observables:

  * Supports `affine` or `rational_quadratic` coupling (via `params.yaml`).
  * Optional Bayesian layers with KL regularization.
  * Methods for forward (`log_prob`), reverse sampling (`sample`), and MAP.

*Essential hyperparameters* (see `params.yaml`): number of blocks (`n_blocks`), bins (`num_bins`), hidden sizes, Bayesian flags, learning rate.

### 4. Training

Run the CLI (**`main.py`**):

```bash
python main.py params.yaml
```

* Loads YAML config, sets up an output folder under `runs/` with timestamp.
* Initializes `Trainer`, which:

  1. Loads data via `get_loaders`
  2. Builds and moves `CINN` model to device
  3. Sets up optimizer + scheduler (supports `one_cycle`, `step`, or `reduce_on_plateau`)
  4. Executes training epochs: computes NLL loss (`-log_prob`) + KL (if Bayesian)
  5. Logs train & val losses, learning rates, and checkpointing at intervals

*Checkpoint files* (`model.pt`, `model_20.pt`, ..., `model_last.pt`) are saved via the `Documenter` utility in `runs/<timestamp>_<run_name>/`.

### 5. Plotting Results

Use the provided notebooks or call functions in **`plot_utils.py`** to regenerate figures:

1. **Posterior Distribution**

   ![Figure 1](1.posterior_distribution.png)

   ```python
   from plot_utils import posterior_distribution
   posterior_distribution(
       model_checkpoint="runs/.../model_last.pt",
       params_path="params.yaml",
       processed_dir="processed_data",
       n_rows=10,
       n_samples=600
   )
   ```

2. **Prediction Performance**

   * *Heatmap of prior vs posterior bins* (`2.prediction_performance1.png`)
   * *MAP vs ground truth & error-vs-truth* (`2.prediction_performance2.png`)

   ```bash
   jupyter nbconvert --to notebook --execute 2.prediction_performance.ipynb
   ```

3. **Uncertainty Analysis**

   ![Figure 3](3.uncertainities.png)

   ```bash
   python 3.uncertainities.ipynb
   ```

4. **Cross-Correlations**

   ![Figure 4](4.cross_correlations.png)

   ```bash
   python 4.cross_correlations.ipynb
   ```

---
