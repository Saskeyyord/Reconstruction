from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # CSV logging remains available if tensorboard is absent
    SummaryWriter = None  # type: ignore[assignment]


@dataclass(frozen=True)
class EpochResult:
    loss: float
    samples: int


class BaselineTrainer:
    """Single-device trainer with checkpoints, resume, early stopping, and logs."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: str | Path,
        patience: int = 12,
        min_delta: float = 1e-4,
        gradient_clip_norm: float | None = 5.0,
        mixed_precision: bool = False,
    ):
        self.model = model.to(device)
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
        self.criterion = nn.SmoothL1Loss()
        self.best_val_loss = float("inf")
        self.bad_epochs = 0
        self.start_epoch = 0
        self.writer = SummaryWriter(self.log_dir / "tensorboard") if SummaryWriter else None

    def _run_epoch(self, loader: DataLoader, training: bool) -> EpochResult:
        self.model.train(training)
        total_loss = 0.0
        total_samples = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                noisy = batch["noisy"].to(self.device, non_blocking=True)
                clean = batch["clean"].to(self.device, non_blocking=True)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    reconstructed = self.model(noisy)
                    loss = self.criterion(reconstructed, clean)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                if training:
                    self.scaler.scale(loss).backward()
                    if self.gradient_clip_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                batch_size = int(noisy.shape[0])
                total_loss += float(loss.detach()) * batch_size
                total_samples += batch_size
        if total_samples == 0:
            raise RuntimeError("DataLoader produced no samples")
        return EpochResult(total_loss / total_samples, total_samples)

    def _save_checkpoint(self, name: str, epoch: int) -> Path:
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
        return target

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
        if resume is not None:
            self.resume(resume)
        log_path = self.log_dir / "training.csv"
        write_header = not log_path.exists() or self.start_epoch == 0
        history: list[dict[str, float | int]] = []
        with log_path.open("a" if not write_header else "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
            if write_header:
                writer.writeheader()
            for epoch in range(self.start_epoch, int(epochs)):
                if hasattr(train_loader.dataset, "set_epoch"):
                    train_loader.dataset.set_epoch(epoch)  # type: ignore[attr-defined]
                train_result = self._run_epoch(train_loader, training=True)
                val_result = self._run_epoch(val_loader, training=False)
                row: dict[str, float | int] = {
                    "epoch": epoch,
                    "train_loss": train_result.loss,
                    "val_loss": val_result.loss,
                }
                history.append(row)
                writer.writerow(row)
                handle.flush()
                if self.writer:
                    self.writer.add_scalar("loss/train", train_result.loss, epoch)
                    self.writer.add_scalar("loss/validation", val_result.loss, epoch)

                improved = val_result.loss < self.best_val_loss - self.min_delta
                if improved:
                    self.best_val_loss = val_result.loss
                    self.bad_epochs = 0
                    self._save_checkpoint("best.pt", epoch)
                else:
                    self.bad_epochs += 1
                self._save_checkpoint("last.pt", epoch)
                if self.bad_epochs >= self.patience:
                    break
        if self.writer:
            self.writer.flush()
            self.writer.close()
        return history

    def evaluate(self, loader: DataLoader) -> EpochResult:
        return self._run_epoch(loader, training=False)
