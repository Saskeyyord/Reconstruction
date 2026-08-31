from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cycloscg.models import CycloSCGNet, ResidualUNet1D
from cycloscg.utils.config import load_yaml


def load_baseline_model(
    config_path: str | Path, checkpoint_path: str | Path, device: torch.device
) -> tuple[ResidualUNet1D, dict]:
    config = load_yaml(config_path)
    model = ResidualUNet1D(**config["model"])
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, config


def load_cycloscg_model(
    config_path: str | Path, checkpoint_path: str | Path, device: torch.device
) -> tuple[CycloSCGNet, dict]:
    config = load_yaml(config_path)
    model = CycloSCGNet(**config["model"])
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, config


@torch.no_grad()
def reconstruct_cycle_matrix(
    model: CycloSCGNet,
    cycles: np.ndarray,
    beats_per_window: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct an arbitrary number of beats using non-overlapping K windows."""
    values = np.asarray(cycles, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("cycles must be a non-empty [num_cycles, phase_length] matrix")
    outputs: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for start in range(0, len(values), beats_per_window):
        chunk = values[start : start + beats_per_window]
        actual = len(chunk)
        if actual < beats_per_window:
            padding = np.repeat(chunk[-1:], beats_per_window - actual, axis=0)
            chunk = np.concatenate((chunk, padding), axis=0)
        tensor = torch.from_numpy(chunk[None]).to(device)
        output = model(tensor, return_aux=True)
        outputs.append(output.reconstruction[0, :actual].cpu().numpy())
        weights.append(output.reliability_weights[0, :actual].cpu().numpy())
    return np.concatenate(outputs, axis=0), np.concatenate(weights, axis=0)

