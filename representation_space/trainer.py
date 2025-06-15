import sys
import time
import torch
import torch.distributions as dist
import numpy as np
from data_loader import get_loaders
from model import CINN


class LogUniform(dist.TransformedDistribution):
    def __init__(self, lb, ub):
        super(LogUniform, self).__init__(dist.Uniform(torch.log(lb), torch.log(ub)),
                                            dist.ExpTransform())
class Trainer:
    """ This class is responsible for training and testing the model.  """

    def __init__(self, params, device, doc):
        """
            Initializes train_loader, test_loader, model, optimizer and scheduler.

            Parameters:
            params: Dict containing the network and training parameter
            device: Device to use for the training
            doc: An instance of the documenter class responsible for documenting the run
        """

        self.params = params
        self.device = device
        self.doc = doc

        self.train_loader, self.val_loader = get_loaders(
            processed_dir   = params['processed_dir'],
            emb_path        = params['emb_path'],
            fname_path      = params['fname_path'],
            batch_size      = params['batch_size'],
            val_frac        = params['val_frac'],
            random_state    = params.get('seed', 42),
            emb_scaling     = params.get('emb_scaling', 'standard'),
        )
        self.single_energy = params.get("single_energy", None)
        self.avg_gen_time = {}

        # ─── 1) Build DataLoaders ───────────────────────────────────────────────
        self.train_loader, self.val_loader = get_loaders(
            processed_dir = params['processed_dir'],
            emb_path      = params['emb_path'],
            fname_path    = params['fname_path'],
            batch_size    = params['batch_size'],
            val_frac      = params['val_frac'],
            random_state  = params.get('seed', 42),
        )

        # ─── 2) Dimensions ─────────────────────────────────────────────────────
        # Number of features per sample
        self.num_dim = self.train_loader.data.shape[1]

        # ─── 3) Prepare raw tensors for CINN init ─────────────────────────────
        data = torch.clone(self.train_loader.data).to(self.device)
        cond = torch.clone(self.train_loader.cond).to(self.device)

        # ─── 4) Instantiate model ──────────────────────────────────────────────
        model = CINN(self.params, data, cond)
        self.model = model.to(self.device)

        # ─── 5) Optimizer & scheduler ─────────────────────────────────────────
        self.set_optimizer(steps_per_epoch=len(self.train_loader))

        # ─── 6) Logging containers ─────────────────────────────────────────────
        self.losses_train = {'inn': [], 'kl': [], 'total': []}
        self.losses_val   = {'inn': [], 'kl': [], 'total': []}
        self.learning_rates = []

        # Optional: print model size
        param_mb = sum(p.numel()*p.element_size() for p in self.model.parameters())
        buf_mb   = sum(b.numel()*b.element_size() for b in self.model.buffers())
        total_mb = (param_mb + buf_mb) / 1024**2
        print(f"Model size: {total_mb:.2f} MB")


    def train(self):
        """Trains the cINN on (feature, embedding) pairs."""
        self.epoch = 0
        self.save()  # save initial state

        N_train = len(self.train_loader.data)

        # If Bayesian, enable MAP mode for training
        if getattr(self.model, "bayesian", False):
            self.model.enable_map()

        for epoch in range(1, self.params["n_epochs"] + 1):
            self.epoch = epoch
            self.model.train()

            # Accumulators
            total_loss = 0.0
            inn_loss_acc = 0.0
            kl_loss_acc  = 0.0
            max_grad     = 0.0
            batch_count  = 0

            # —— Training loop —— 
            for x, c in self.train_loader:
                x, c = x.to(self.device), c.to(self.device)
                self.optim.zero_grad()

                inn_loss = -torch.mean(self.model.log_prob(x, c))
                if getattr(self.model, "bayesian", False):
                    kl = self.model.get_kl() / N_train
                    loss = inn_loss + kl
                    kl_loss_acc += kl.item() * x.size(0)
                else:
                    loss = inn_loss

                loss.backward()
                if "grad_clip" in self.params:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.params_trainable,
                        self.params["grad_clip"]
                    )
                self.optim.step()

                # Scheduler step (for OneCycleLR)
                if hasattr(self.scheduler, "step"):
                    try:
                        self.scheduler.step()
                    except TypeError:
                        # some schedulers expect a metric (ReduceLROnPlateau)
                        pass

                inn_loss_acc  += inn_loss.item() * x.size(0)
                total_loss    += loss.item()    * x.size(0)
                self.learning_rates.append(self.optim.param_groups[0]["lr"])
                batch_count   += x.size(0)

                # track max gradient
                for p in self.model.params_trainable:
                    if p.grad is not None:
                        max_grad = max(max_grad, p.grad.abs().max().item())

            # —— Validation loop —— 
            self.model.eval()
            val_inn_loss = 0.0
            val_kl_loss  = 0.0
            with torch.no_grad():
                for x, c in self.val_loader:
                    x, c = x.to(self.device), c.to(self.device)
                    inn_loss = -torch.mean(self.model.log_prob(x, c))
                    if getattr(self.model, "bayesian", False):
                        kl = self.model.get_kl() / N_train
                        loss = inn_loss + kl
                        val_kl_loss += kl.item() * x.size(0)
                    else:
                        loss = inn_loss
                    val_inn_loss += inn_loss.item() * x.size(0)

            # —— Compute epoch averages —— 
            train_inn = inn_loss_acc / N_train
            train_tot = total_loss   / N_train
            val_inn   = val_inn_loss / len(self.val_loader.data)
            val_tot   = (val_inn_loss + val_kl_loss) / len(self.val_loader.data) \
                        if getattr(self.model, "bayesian", False) else val_inn

            # —— Logging —— 
            self.losses_train["inn"].append(train_inn)
            self.losses_train["total"].append(train_tot)
            self.losses_val["inn"].append(val_inn)
            self.losses_val["total"].append(val_tot)
            if getattr(self.model, "bayesian", False):
                train_kl = kl_loss_acc / N_train
                val_kl   = val_kl_loss  / len(self.val_loader.data)
                self.losses_train["kl"].append(train_kl)
                self.losses_val["kl"].append(val_kl)

            print(f"\n=== Epoch {epoch} ===")
            print(f"Train  → inn: {train_inn:.4f}" +
                  (f", kl: {train_kl:.4f}" if getattr(self.model, "bayesian", False) else ""))
            print(f"         total: {train_tot:.4f}")
            print(f"Val    → inn: {val_inn:.4f}" +
                  (f", kl: {val_kl:.4f}"   if getattr(self.model, "bayesian", False) else ""))
            print(f"         total: {val_tot:.4f}")
            print(f"LR: {self.optim.param_groups[0]['lr']:.3e}, max grad: {max_grad:.3e}")

            # Save checkpoint every save_interval
            if epoch % self.params.get("save_interval", 20) == 0:
                self.save(f"_{epoch}")

        # Save final model
        self.save("_last")

 
    def set_optimizer(self, steps_per_epoch=1, no_training=False, params=None):
        if params is None:
            params = self.params

        # —— Debug: print out what we loaded —— 
        print("Optimizer params:", {
            "lr":      params.get("lr"),
            "betas":   params.get("betas"),
            "eps":     params.get("eps"),
            "weight_decay": params.get("weight_decay")
        })

        # —— Cast everything to numeric types —— 
        lr = float(params.get("lr", 2e-4))
        betas = tuple(float(b) for b in params.get("betas", [0.9, 0.999]))
        eps = float(params.get("eps", 1e-6))
        wd  = float(params.get("weight_decay", 0.0))

        self.optim = torch.optim.AdamW(
            self.model.params_trainable,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=wd,
        )
        if no_training:
            return

        mode = params.get("lr_scheduler", "reduce_on_plateau")
        if mode == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optim,
                step_size=params["lr_decay_epochs"],
                gamma=float(params["lr_decay_factor"]),
            )
        elif mode == "reduce_on_plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim,
                factor=float(params.get("reduce_factor", 0.8)),
                patience=int(params.get("reduce_patience", 20)),
                threshold=float(params.get("reduce_threshold", 1e-4)),
            )
        elif mode == "one_cycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optim,
                max_lr=float(params.get("max_lr", lr * 10)),
                epochs=int(params.get("n_epochs", 100)),
                steps_per_epoch=steps_per_epoch,
            )
        else:
            raise ValueError(f"Unknown lr_scheduler mode: {mode}")


    def save(self, epoch=""):
        """Save model state (and optionally scaler state)."""
        out = {
            "net": self.model.state_dict(),
        }
        torch.save(out, self.doc.get_file(f"model{epoch}.pt"))

    def load(self, epoch=""):
        """Load model state."""
        path = self.doc.get_file(f"model{epoch}.pt")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["net"])
        self.model.to(self.device)

    def sample(self, embeddings: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Generates `n_samples` feature‐vectors per input embedding.

        Args:
            embeddings: Tensor of shape (batch_size, D_emb) on self.device
            n_samples:  number of samples to draw per embedding

        Returns:
            Tensor of shape (batch_size, n_samples, D_feat)
        """
        self.model.eval()
        with torch.no_grad():
            # sample from base Normal
            # .sample() on your model returns shape (batch, n_samples, D_feat)
            return self.model.sample(n_samples, embeddings)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encodes `features` to latent z-space: z = f(x | c).
        Useful for latent‐space visualization/debugging if you need it.

        Args:
            features: Tensor of shape (batch_size, D_feat)

        Returns:
            Tensor of shape (batch_size, D_feat) latent codes
        """
        self.model.eval()
        with torch.no_grad():
            # we need a dummy condition; if you have embeddings for these features you can pass them in
            raise NotImplementedError("Provide corresponding embeddings to encode.")
