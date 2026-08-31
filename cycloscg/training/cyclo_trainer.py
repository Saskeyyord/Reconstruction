from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from cycloscg.losses import CycloSCGLoss
from cycloscg.models import CycloSCGNet

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment]


class CycloTrainer:
    """Trainer for cardiac-phase matrix reconstruction and structured losses."""

    def __init__(
        self,
        model: CycloSCGNet,
        loss_function: CycloSCGLoss,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: str | Path,
        patience: int = 12,
        min_delta: float = 1e-4,
        gradient_clip_norm: float | None = 5.0,
        mixed_precision: bool = False,
    ):
        self.model = model.to(device)
        self.loss_function = loss_function.to(device)
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.gradient_clip_norm = gradient_clip_norm
        self.use_amp = bool(mixed_precision and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.best_val_loss = float("inf")
        self.bad_epochs = 0
        self.start_epoch = 0
        self.writer = SummaryWriter(self.log_dir / "tensorboard") if SummaryWriter else None

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        totals: dict[str, float] = {}
        samples = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                noisy = batch["noisy"].to(self.device, non_blocking=True)
                clean = batch["clean"].to(self.device, non_blocking=True)
                identity = batch["identity"].to(self.device, non_blocking=True)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    output = self.model(noisy, return_aux=True)
                # Structured linear algebra losses remain FP32 for stable SVD,
                # Gram matrices, and STFT on consumer GPUs.
                reconstruction = output.reconstruction.float()
                clean_float = clean.float()
                noisy_float = noisy.float()
                loss, components = self.loss_function(reconstruction, clean_float, identity)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite CycloSCG loss")
                if training:
                    self.scaler.scale(loss).backward()
                    if self.gradient_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                batch_size = int(noisy.shape[0])
                samples += batch_size
                error_power = (reconstruction - clean_float).square().mean(dim=(-2, -1))
                raw_error_power = (noisy_float - clean_float).square().mean(dim=(-2, -1))
                rmse = torch.sqrt(error_power.clamp_min(1e-12)).mean()
                per_sample_snr_improvement = (
                    10.0
                    * torch.log10(raw_error_power.clamp_min(1e-12) / error_power.clamp_min(1e-12))
                )
                non_identity = ~identity.bool()
                snr_improvement = (
                    per_sample_snr_improvement[non_identity].mean()
                    if torch.any(non_identity)
                    else per_sample_snr_improvement.sum() * 0.0
                )
                estimate_centered = reconstruction.flatten(1) - reconstruction.flatten(1).mean(
                    dim=1, keepdim=True
                )
                target_centered = clean_float.flatten(1) - clean_float.flatten(1).mean(
                    dim=1, keepdim=True
                )
                pearson = (
                    (estimate_centered * target_centered).sum(dim=1)
                    / (
                        torch.linalg.vector_norm(estimate_centered, dim=1)
                        * torch.linalg.vector_norm(target_centered, dim=1)
                    ).clamp_min(1e-8)
                ).mean()
                batch_values = {
                    "total": loss.detach(),
                    **{k: v.detach() for k, v in components.items()},
                    "rmse": rmse.detach(),
                    "snr_improvement_db": snr_improvement.detach(),
                    "pearson_r": pearson.detach(),
                }
                for name, value in batch_values.items():
                    totals[name] = totals.get(name, 0.0) + float(value) * batch_size
        if samples == 0:
            raise RuntimeError("DataLoader produced no samples")
        return {name: value / samples for name, value in totals.items()}

    def _save(self, name: str, epoch: int) -> None:
        target = self.checkpoint_dir / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "bad_epochs": self.bad_epochs,
                "scaler_state": self.scaler.state_dict(),
            },
            temporary,
        )
        temporary.replace(target)

    def resume(self, checkpoint: str | Path) -> None:
        payload = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state"])
        self.optimizer.load_state_dict(payload["optimizer_state"])
        self.best_val_loss = float(payload.get("best_val_loss", float("inf")))
        self.bad_epochs = int(payload.get("bad_epochs", 0))
        if "scaler_state" in payload:
            self.scaler.load_state_dict(payload["scaler_state"])
        self.start_epoch = int(payload["epoch"]) + 1

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        resume: str | Path | None = None,
    ) -> list[dict[str, float | int]]:
        if resume:
            self.resume(resume)
        log_path = self.log_dir / "training.csv"
        history: list[dict[str, float | int]] = []
        fieldnames = [
            "epoch",
            "train_total",
            "val_total",
            "train_wave",
            "val_wave",
            "train_cycle",
            "val_cycle",
            "train_cov",
            "val_cov",
            "train_svd",
            "val_svd",
            "train_spec",
            "val_spec",
            "train_identity",
            "val_identity",
            "train_rmse",
            "val_rmse",
            "train_snr_improvement_db",
            "val_snr_improvement_db",
            "train_pearson_r",
            "val_pearson_r",
        ]
        new_log = self.start_epoch == 0 or not log_path.exists()
        with log_path.open("w" if new_log else "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if new_log:
                writer.writeheader()
            for epoch in range(self.start_epoch, int(epochs)):
                if hasattr(train_loader.dataset, "set_epoch"):
                    train_loader.dataset.set_epoch(epoch)  # type: ignore[attr-defined]
                train = self._run_epoch(train_loader, training=True)
                validation = self._run_epoch(val_loader, training=False)
                row: dict[str, float | int] = {"epoch": epoch}
                for name in (
                    "total",
                    "wave",
                    "cycle",
                    "cov",
                    "svd",
                    "spec",
                    "identity",
                    "rmse",
                    "snr_improvement_db",
                    "pearson_r",
                ):
                    row[f"train_{name}"] = train[name]
                    row[f"val_{name}"] = validation[name]
                history.append(row)
                writer.writerow(row)
                handle.flush()
                if self.writer:
                    for name, value in train.items():
                        self.writer.add_scalar(f"train/{name}", value, epoch)
                    for name, value in validation.items():
                        self.writer.add_scalar(f"validation/{name}", value, epoch)
                if validation["total"] < self.best_val_loss - self.min_delta:
                    self.best_val_loss = validation["total"]
                    self.bad_epochs = 0
                    self._save("best.pt", epoch)
                else:
                    self.bad_epochs += 1
                self._save("last.pt", epoch)
                if self.bad_epochs >= self.patience:
                    break
        if self.writer:
            self.writer.flush()
            self.writer.close()
        return history
