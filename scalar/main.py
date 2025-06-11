#!/usr/bin/env python3
import os
import sys
import argparse
from datetime import datetime

import yaml
import torch

from trainer import Trainer


class Documenter:
    """
    Helper to manage output paths.
    doc.get_file('foo.png') → '<basedir>/foo.png'
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
    p = argparse.ArgumentParser(description="Train a conditional INN on observables → unobservables")
    p.add_argument("param_file", help="YAML file with training parameters")
    args = p.parse_args()

    # 1) Load YAML
    with open(args.param_file) as f:
        params = yaml.safe_load(f)

    # 2) Setup output folder
    run_name  = params.get("run_name", os.path.splitext(os.path.basename(args.param_file))[0])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base      = params.get("output_dir", "runs")
    basedir   = os.path.join(base, f"{timestamp}_{run_name}")
    doc       = Documenter(basedir)

    # 3) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4) Train model
    trainer = Trainer(params, device, doc)
    trainer.train()

if __name__ == "__main__":
    main()
