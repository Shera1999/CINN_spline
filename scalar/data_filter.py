#!/usr/bin/env python3
"""
cluster_data_filter_simple.py

Load your raw observables+unobservables CSVs, drop any row with NaNs
in the five target columns (i.e. only keep clusters that actually
had a merger), scale both X and Y, and write out processed_data/.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

OBS_COLS = [
    'bcg_SubhaloBHMass',
    'bcg_SubhaloMass',
    'bcg_StellarMass',
    'Group_M_Crit500',
    'Group_R_Crit500',
    'lookback_time_Gyr',
    'GroupGasMass'
]

TARGET_COLS = [
    'last_T_coll',
    'last_V_coll',
    'last_M_Crit500_coll',
    'last_Subcluster_mass',
    'last_Mass_ratio',
    'last_d_peri',
]

def generate_processed_data(obs_csv,
                            unobs_csv,
                            out_dir="processed_data"):
    # 1) Load & merge on (HaloID, Snapshot)
    df_obs   = pd.read_csv(obs_csv)
    df_unobs = pd.read_csv(unobs_csv)
    df       = pd.merge(df_obs, df_unobs, on=['HaloID','Snapshot'], how='inner')

    # 2) Drop any rows where *any* target is NaN
    before = len(df)
    df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before - after} / {before} clusters with missing targets.")

    # 3) Extract numpy arrays
    X_raw = df[OBS_COLS].to_numpy()
    Y_raw = df[TARGET_COLS].to_numpy()
    meta  = df[['HaloID','Snapshot']]

    # 4) Fit + apply StandardScalers
    obs_scaler = StandardScaler().fit(X_raw)
    tar_scaler = StandardScaler().fit(Y_raw)

    X_scaled = obs_scaler.transform(X_raw)
    Y_scaled = tar_scaler.transform(Y_raw)

    # 5) Make output directory
    os.makedirs(out_dir, exist_ok=True)

    # 6) Dump CSVs & scalers
    pd.DataFrame(X_scaled, columns=OBS_COLS)    .to_csv(os.path.join(out_dir,"X.csv"), index=False)
    pd.DataFrame(Y_scaled, columns=TARGET_COLS).to_csv(os.path.join(out_dir,"Y.csv"), index=False)
    meta.to_csv(os.path.join(out_dir,"meta.csv"), index=False)

    joblib.dump(obs_scaler, os.path.join(out_dir,"obs_scaler.pkl"))
    joblib.dump(tar_scaler, os.path.join(out_dir,"tar_scaler.pkl"))

    print(f"Processed data written to '{out_dir}/' with {len(df)} clusters.")

if __name__ == "__main__":
    # ** adjust paths here if needed **
    generate_processed_data(
        obs_csv   = "observables1.csv",
        unobs_csv = "unobservables1.csv",
        out_dir   = "processed_data"
    )


# Explanation of each step:

#Load & Merge
#Reads both CSVs and inner‐joins on HaloID, Snapshot.

#Drop NaNs
#Keeps only rows where all five target columns are present.

#Extract Arrays + Meta
#Builds X_raw (observables), Y_raw (targets), and a small meta‐DataFrame for IDs.

#Scale
#Fits StandardScaler on the raw X and Y and transforms them.

#Dump Outputs
#Writes out:

#processed_data/X.csv (scaled observables)
#processed_data/Y.csv (scaled targets)
#processed_data/meta.csv (HaloID & Snapshot)
#processed_data/obs_scaler.pkl & tar_scaler.pkl