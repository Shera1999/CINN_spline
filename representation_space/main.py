#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime

import yaml
import torch

from data_loader import get_loaders
from model import CINN
from trainer import Trainer

class Documenter:
    """
    Simple helper to manage output paths.
    doc.get_file('foo.png') -> '<basedir>/foo.png'
    """
    def __init__(self, basedir: str):
        self.basedir = basedir
        os.makedirs(basedir, exist_ok=True)

    def get_file(self, relpath: str) -> str:
        full = os.path.join(self.basedir, relpath)
        d = os.path.dirname(full)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
        return full

def main():
    p = argparse.ArgumentParser(
        description="Train a conditional INN on cluster embeddings→features"
    )
    p.add_argument("param_file", help="YAML file with training parameters")
    args = p.parse_args()

    # 1) Load params.yaml
    with open(args.param_file) as f:
        params = yaml.safe_load(f)

    # 2) Build output directory
    run_name  = params.get("run_name",
                   os.path.splitext(os.path.basename(args.param_file))[0])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base      = params.get("output_dir", "runs")
    basedir   = os.path.join(base, f"{timestamp}_{run_name}")
    doc       = Documenter(basedir)

    # 3) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4) Data loaders
    train_loader, val_loader = get_loaders(
        processed_dir = params["processed_dir"],
        emb_path      = params["emb_path"],
        fname_path    = params["fname_path"],
        batch_size    = params["batch_size"],
        val_frac      = params["val_frac"],
        random_state  = params.get("seed", 42),
    )

    # 5) Initialize the CINN model (inside Trainer)
    trainer = Trainer(params, device, doc)
    # the Trainer __init__ will internally call get_loaders again;
    # if you prefer, you can refactor Trainer to accept existing loaders.

    # 6) Train!
    trainer.train()

if __name__ == "__main__":
    main()
