#!/usr/bin/env python3
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster     import KMeans
from sklearn.model_selection import train_test_split
from data_loader    import normalize_fname

def main():
    processed_dir = "processed_data"
    os.makedirs(processed_dir, exist_ok=True)

    # 1) Load & replicate Y + meta
    dfY  = pd.read_csv(os.path.join(processed_dir, "Y.csv"))
    df_meta = pd.read_csv(os.path.join(processed_dir, "meta.csv"))
    projs = [1,2,3]
    df_meta_rep = pd.concat([df_meta.assign(proj=p) for p in projs],
                             ignore_index=True)
    Y_rep = np.repeat(dfY.values, len(projs), axis=0)

    # 2) Load raw embeddings & map into replication order
    all_emb    = np.load("embeddings.npy")   # shape (N_proj, D_emb)
    all_files  = np.load("filenames.npy")    # shape (N_proj,)
    df_emb     = pd.DataFrame(all_emb)
    df_emb["key"] = [normalize_fname(fn) for fn in all_files]
    emb_map    = {row["key"]: row.drop("key").values
                  for _,row in df_emb.iterrows()}

    keys = (df_meta_rep["HaloID"].astype(int).astype(str) + "_" +
            df_meta_rep["Snapshot"].astype(int).astype(str) + "_" +
            df_meta_rep["proj"].astype(int).astype(str))
    E_rep = np.vstack([emb_map[k] for k in keys])

    # 3) Scale embeddings
    emb_sc = StandardScaler().fit(E_rep)
    E_rep_s = emb_sc.transform(E_rep)
    joblib.dump(emb_sc, os.path.join(processed_dir, "emb_scaler.pkl"))

    # 4) Global train/val/test split by HaloID (80/10/10)
    halos = df_meta_rep["HaloID"].unique()
    halos_tmp, halos_te = train_test_split(halos, test_size=0.1, random_state=0)
    val_frac = 0.1/0.9
    halos_tr, halos_va = train_test_split(halos_tmp, test_size=val_frac, random_state=0)
    mask = df_meta_rep["HaloID"]
    idx = np.arange(len(df_meta_rep))
    idx_tr = idx[mask.isin(halos_tr)]
    idx_va = idx[mask.isin(halos_va)]
    idx_te = idx[mask.isin(halos_te)]

    np.save(os.path.join(processed_dir, "idx_tr.npy"), idx_tr)
    np.save(os.path.join(processed_dir, "idx_va.npy"), idx_va)
    np.save(os.path.join(processed_dir, "idx_te.npy"), idx_te)

    # 5) K-means on training embeddings
    M = 50  # number of experts
    km = KMeans(n_clusters=M, random_state=0).fit(E_rep_s[idx_tr])
    centers = km.cluster_centers_
    labels  = km.predict(E_rep_s)

    np.save(os.path.join(processed_dir, "kmeans_centers.npy"), centers)
    np.save(os.path.join(processed_dir, "labels.npy"), labels)

    # 6) Save the full stacked arrays
    np.save(os.path.join(processed_dir, "E_rep.npy"), E_rep_s)
    np.save(os.path.join(processed_dir, "Y_rep.npy"), Y_rep)

    print("✅  cluster_setup complete")

if __name__ == "__main__":
    main()
