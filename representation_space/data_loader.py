#!/usr/bin/env python3
import os
import re
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def normalize_fname(fname: str) -> str:
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.match(r"^snap_(\d+)_halo_(\d+)_proj_(\d+)$", base)
    if not m:
        raise ValueError(f"Unexpected filename: {base}")
    snap, halo, proj = map(int, m.groups())
    return f"{halo}_{snap}_{proj}"

def get_embeddings_for_meta(meta_df: pd.DataFrame, emb_map: dict) -> np.ndarray:
    missing, E = [], []
    for _, row in meta_df.iterrows():
        key = f"{int(row.HaloID)}_{int(row.Snapshot)}_{int(row.proj)}"
        emb = emb_map.get(key)
        if emb is None:
            missing.append(key)
        else:
            E.append(emb)
    if missing:
        raise KeyError(f"Missing embeddings for keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    return np.vstack(E)

def make_loader(Y: np.ndarray, E: np.ndarray, batch_size=512, shuffle=True):
    Y_t = torch.tensor(Y, dtype=torch.float32)
    E_t = torch.tensor(E, dtype=torch.float32)
    ds = TensorDataset(Y_t, E_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    loader.data = Y_t
    loader.cond = E_t
    return loader

def get_loaders(processed_dir="processed_data",
                emb_path="embeddings.npy",
                fname_path="filenames.npy",
                batch_size=512,
                val_frac=0.1,
                random_state=42,
                emb_scaling="standard"):
    # 1) Load targets + meta
    dfY   = pd.read_csv(os.path.join(processed_dir, "Y.csv"))
    meta  = pd.read_csv(os.path.join(processed_dir, "meta.csv"))
    Y_all = dfY.values  # (N, D_feat)

    # 2) Replicate meta×3 projections
    projs    = [1,2,3]
    meta_rep = pd.concat([meta.assign(proj=p) for p in projs], ignore_index=True)
    Y_rep    = np.repeat(Y_all, len(projs), axis=0)

    # 3) Load & align raw embeddings
    all_emb   = np.load(emb_path)
    all_files = np.load(fname_path)
    keys      = [normalize_fname(f) for f in all_files]
    emb_map   = {k:emb for k,emb in zip(keys, all_emb)}
    E_raw     = get_embeddings_for_meta(meta_rep, emb_map)  # (3N, D_emb)

    # 4) Split by HaloID for train/val
    halos      = meta_rep["HaloID"].unique()
    _, halos_val = train_test_split(halos, test_size=val_frac, random_state=random_state)
    mask_val   = meta_rep["HaloID"].isin(halos_val).to_numpy()
    mask_train = ~mask_val

    Y_tr_raw   = Y_rep[mask_train]
    Y_val_raw  = Y_rep[mask_val]
    E_tr_raw   = E_raw[mask_train]
    E_val_raw  = E_raw[mask_val]

    # 5) Embedding preprocessing
    if emb_scaling == "standard":
        emb_sc = StandardScaler().fit(E_tr_raw)
        E_tr  = emb_sc.transform(E_tr_raw)
        E_val = emb_sc.transform(E_val_raw)
        joblib.dump(emb_sc, os.path.join(processed_dir, "emb_scaler.pkl"))

    elif emb_scaling == "center":
        emb_mean = E_tr_raw.mean(axis=0)
        E_tr     = E_tr_raw - emb_mean
        E_val    = E_val_raw - emb_mean
        np.save(os.path.join(processed_dir, "emb_mean.npy"), emb_mean)

    elif emb_scaling == "none":
        E_tr, E_val = E_tr_raw, E_val_raw

    else:
        raise ValueError(f"Unknown emb_scaling mode: {emb_scaling}")

    # 6) Build DataLoaders
    train_loader = make_loader(Y_tr_raw, E_tr, batch_size=batch_size, shuffle=True)
    val_loader   = make_loader(Y_val_raw, E_val, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
