#!/usr/bin/env python3
import os
import numpy as np
import torch
import yaml
from trainer import Trainer

class DummyDoc:
    """Saves expert model to experts/expert_{m}/model.pt."""
    def __init__(self, expert_root, expert_id):
        self.base_dir = os.path.join(expert_root, f"expert_{expert_id}")
        os.makedirs(self.base_dir, exist_ok=True)

    def get_file(self, name):
        return os.path.join(self.base_dir, name)

def main():
    # === 1) Load training artifacts ===
    processed_dir = "processed_data"
    expert_save_dir = "experts"  

    E_rep    = np.load(os.path.join(processed_dir, "E_rep.npy"))
    Y_rep    = np.load(os.path.join(processed_dir, "Y_rep.npy"))
    labels   = np.load(os.path.join(processed_dir, "labels.npy"))
    idx_tr   = np.load(os.path.join(processed_dir, "idx_tr.npy"))
    idx_va   = np.load(os.path.join(processed_dir, "idx_va.npy"))

    # === 2) Load training config ===
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    # === 3) Loop through experts ===
    M = int(labels.max()) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for m in range(M):
        print(f"\n🔹 Training expert {m}")
        n_tr = np.sum(labels[idx_tr] == m)
        if n_tr < 20:
            print(f"⏭️  Skipping expert {m}: only {n_tr} training samples.")
            continue

        doc = DummyDoc(expert_save_dir, m)

        # Standard Trainer (uses get_loaders internally)
        trainer = Trainer(params, device, doc)
        trainer.train()

if __name__ == "__main__":
    main()
