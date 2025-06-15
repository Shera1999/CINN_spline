# Add these imports at the top
from sklearn.metrics import pairwise_distances_argmin
from glob import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KernelDensity
import torch
from model import CINN
import os
import yaml

def load_model_and_data(model_checkpoint, params_path, processed_dir, device):
    """
    If model_checkpoint is a single file: load unified model (non-MoE).
    If model_checkpoint is a directory (like 'experts/'): assume MoE and load expert models on demand.
    """
    # === Shared: load targets, embeddings, and scalers ===
    meta      = pd.read_csv(os.path.join(processed_dir, "meta.csv"))
    Y_full    = pd.read_csv(os.path.join(processed_dir, "Y.csv")).values
    tar_scaler = joblib.load(os.path.join(processed_dir, "tar_scaler.pkl"))

    embeddings = np.load(os.path.join(processed_dir, "..", "embeddings.npy"))
    filenames  = np.load(os.path.join(processed_dir, "..", "filenames.npy"))

    def normalize_fname(fname):
        base = os.path.splitext(os.path.basename(fname))[0]
        m = __import__("re").match(r"^snap_(\d+)_halo_(\d+)_proj_(\d+)$", base)
        if not m:
            raise ValueError(f"Unexpected filename: {base}")
        snap, halo, proj = map(int, m.groups())
        return f"{halo}_{snap}_{proj}"

    keys = [normalize_fname(f) for f in filenames]
    emb_map = {k: emb for k, emb in zip(keys, embeddings)}

    projs     = [1, 2, 3]
    meta_rep  = pd.concat([meta.assign(proj=p) for p in projs], ignore_index=True)

    missing = []
    E_list  = []
    for _, row in meta_rep.iterrows():
        key = f"{int(row.HaloID)}_{int(row.Snapshot)}_{int(row.proj)}"
        emb = emb_map.get(key)
        if emb is None:
            missing.append(key)
        else:
            E_list.append(emb)
    if missing:
        raise KeyError(f"Missing embeddings for keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    E_full = np.vstack(E_list)

    emb_scaler_path = os.path.join(processed_dir, "emb_scaler.pkl")
    if os.path.exists(emb_scaler_path):
        emb_scaler = joblib.load(emb_scaler_path)
        E_full = emb_scaler.transform(E_full)

    Y_full_rep = np.repeat(Y_full, len(projs), axis=0)

    # === Detect if we're in MoE mode ===
    is_moe = os.path.isdir(model_checkpoint)
    if not is_moe:
        # Load single CINN model
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        model = CINN(
            params = params,
            data   = torch.tensor(Y_full_rep, dtype=torch.float32, device=device),
            cond   = torch.tensor(E_full,    dtype=torch.float32, device=device)
        ).to(device)
        ckpt = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(ckpt["net"] if "net" in ckpt else ckpt)
        model.eval()
        return model, E_full, Y_full_rep, tar_scaler

    # === MoE mode ===
    centers = np.load(os.path.join(processed_dir, "kmeans_centers.npy"))
    expert_ids = pairwise_distances_argmin(E_full, centers)

    # Load all expert models into a list
    expert_models = []
    for expert_id in range(centers.shape[0]):
        ckpt_path = os.path.join(model_checkpoint, f"expert_{expert_id}", "model_last.pt")
        if not os.path.exists(ckpt_path):
            expert_models.append(None)
            continue
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        model = CINN(
            params = params,
            data   = torch.tensor(Y_full_rep, dtype=torch.float32, device=device),
            cond   = torch.tensor(E_full,    dtype=torch.float32, device=device)
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["net"] if "net" in ckpt else ckpt)
        model.eval()
        expert_models.append(model)

    # Return special MoE-aware "model" that routes input correctly
    def expert_router_model(x, n_samples=100):
        x = torch.tensor(x, dtype=torch.float32, device=device)
        batch_expert_ids = pairwise_distances_argmin(x.cpu().numpy(), centers)
        all_samples = []
        for i, expert_id in enumerate(batch_expert_ids):
            expert = expert_models[expert_id]
            if expert is None:
                raise RuntimeError(f"No model found for expert {expert_id}")
            with torch.no_grad():
                sample_i = expert.sample(n_samples, x[i:i+1])  # (1, n_samples, D)
                all_samples.append(sample_i)
        return torch.cat(all_samples, dim=0)

    return expert_router_model, E_full, Y_full_rep, tar_scaler


def sample_posteriors(model, embeddings, scaler, n_samples=100, device="cpu"):
    """
    Supports both a unified model and MoE router function.
    """
    if callable(model):  # MoE
        samples = model(embeddings, n_samples)
    else:  # unified
        model.eval()
        with torch.no_grad():
            obs_torch = torch.tensor(embeddings, dtype=torch.float32, device=device)
            samples = model.sample(n_samples, obs_torch)
    samples_np = samples.cpu().numpy()
    samples_phys = np.array([scaler.inverse_transform(s) for s in samples_np])
    return samples_phys


def compute_map_estimates(samples_phys, bandwidth=0.3):
    """
    Compute MAP estimates for each variable using 1D KDE per dimension.

    Parameters:
        samples_phys : np.ndarray of shape (N_proj, n_samples, D_tar_phys)

    Returns:
        maps : np.ndarray of shape (N_proj, D_tar_phys)
    """
    N, n_samples, D = samples_phys.shape
    maps = np.zeros((N, D))
    for i in range(N):
        for d in range(D):
            samp = samples_phys[i, :, d][:, None]
            kde  = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(samp)
            grid = np.linspace(samp.min(), samp.max(), 200)[:, None]
            log_dens = kde.score_samples(grid)
            # pick the grid point with maximum density
            maps[i, d] = grid[np.argmax(np.exp(log_dens)), 0]
    return maps


def build_priors_from_training(Y_train_phys, bandwidth=0.5, n_grid=200):
    """
    Estimate marginal priors from training data using 1D KDE per dimension.

    Parameters:
        Y_train_phys : np.ndarray of shape (N_train, D_tar_phys)
        bandwidth    : float, KDE bandwidth
        n_grid       : int, number of grid points

    Returns:
        List of (grid, density) tuples per dimension
    """
    D = Y_train_phys.shape[1]
    priors = []
    for d in range(D):
        data = Y_train_phys[:, d][:, None]
        kde  = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(data)
        grid = np.linspace(data.min(), data.max(), n_grid)[:, None]
        dens = np.exp(kde.score_samples(grid))
        dens /= dens.max()
        priors.append((grid.ravel(), dens))
    return priors
