#!/usr/bin/env python3
import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def make_loader(Y: np.ndarray, X: np.ndarray, batch_size=512, shuffle=True):
    Y_t = torch.tensor(Y, dtype=torch.float32)
    X_t = torch.tensor(X, dtype=torch.float32)
    ds = TensorDataset(Y_t, X_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    loader.data = Y_t
    loader.cond = X_t
    return loader

def get_loaders(processed_dir="processed_data",
                batch_size=512,
                val_frac=0.1,
                random_state=42):

    # 1) Load processed data
    dfX  = pd.read_csv(os.path.join(processed_dir, "X.csv"))
    dfY  = pd.read_csv(os.path.join(processed_dir, "Y.csv"))
    meta = pd.read_csv(os.path.join(processed_dir, "meta.csv"))

    X_all = dfX.values  # (N, D_obs)
    Y_all = dfY.values  # (N, D_tar)

    # 2) Split by unique HaloID
    halos = meta["HaloID"].unique()
    _, halos_val = train_test_split(halos, test_size=val_frac, random_state=random_state)
    mask_val   = meta["HaloID"].isin(halos_val).to_numpy()
    mask_train = ~mask_val

    X_tr = X_all[mask_train]
    X_val = X_all[mask_val]
    Y_tr = Y_all[mask_train]
    Y_val = Y_all[mask_val]

    # 3) Build DataLoaders
    train_loader = make_loader(Y_tr, X_tr, batch_size=batch_size, shuffle=True)
    val_loader   = make_loader(Y_val, X_val, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
