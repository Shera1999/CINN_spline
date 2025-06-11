# plot_utils.py

import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KernelDensity
import torch
from model import CINN
import os
import yaml


def load_model_and_data(model_checkpoint, params_path, processed_dir, device):
    X_full = pd.read_csv(os.path.join(processed_dir, "X.csv")).values
    Y_full = pd.read_csv(os.path.join(processed_dir, "Y.csv")).values
    tar_sc = joblib.load(os.path.join(processed_dir, "tar_scaler.pkl"))

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    model = CINN(params=params,
                 data=torch.tensor(Y_full, dtype=torch.float32),
                 cond=torch.tensor(X_full, dtype=torch.float32)).to(device)

    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["net"] if "net" in checkpoint else checkpoint)
    model.eval()

    return model, X_full, Y_full, tar_sc

def sample_posteriors(model, observables, scaler, n_samples=100, device='cpu'):
    """
    Sample the posterior for a batch of observables.

    Parameters:
        model       : trained CINN model
        observables : np.ndarray of shape (N, D_obs)
        scaler      : sklearn StandardScaler for targets
        n_samples   : number of samples per observable
        device      : 'cpu' or 'cuda'

    Returns:
        samples_phys : np.ndarray of shape (N, n_samples, D_tar)
    """
    model.eval()
    with torch.no_grad():
        obs_torch = torch.tensor(observables, dtype=torch.float32, device=device)
        samples = model.sample(n_samples, obs_torch)  # (N, n_samples, D_tar)
        samples_np = samples.cpu().numpy()
    samples_phys = np.array([scaler.inverse_transform(s) for s in samples_np])
    return samples_phys


def compute_map_estimates(samples_phys, bandwidth=0.3):
    """
    Compute MAP estimates for each variable using KDE.

    Parameters:
        samples_phys : np.ndarray of shape (N, n_samples, D_tar)

    Returns:
        maps : np.ndarray of shape (N, D_tar)
    """
    N, n_samples, D = samples_phys.shape
    maps = np.zeros((N, D))
    for i in range(N):
        for d in range(D):
            samp = samples_phys[i, :, d][:, None]
            kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(samp)
            grid = np.linspace(samp.min(), samp.max(), 200)[:, None]
            log_dens = kde.score_samples(grid)
            maps[i, d] = grid[np.argmax(np.exp(log_dens)), 0]
    return maps


def build_priors_from_training(Y_train_phys, bandwidth=0.5, n_grid=200):
    """
    Estimate priors from training data using KDE.

    Parameters:
        Y_train_phys : np.ndarray of shape (N, D_tar)
        bandwidth    : float, KDE bandwidth
        n_grid       : int, number of points on the grid

    Returns:
        List of (grid, density) tuples per dimension
    """
    D = Y_train_phys.shape[1]
    priors = []
    for d in range(D):
        data = Y_train_phys[:, d][:, None]
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(data)
        grid = np.linspace(data.min(), data.max(), n_grid)[:, None]
        dens = np.exp(kde.score_samples(grid))
        dens /= dens.max()
        priors.append((grid.ravel(), dens))
    return priors
