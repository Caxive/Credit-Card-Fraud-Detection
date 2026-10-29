import numpy as np
from pathlib import Path

import joblib
import torch
from torch import nn

from src.model import build_autoencoder


def load_model_and_scaler(models_dir: str = "models", latent_dim: int = 16):
    """
    Load the trained autoencoder and the fitted scaler.
    """
    models_path = Path(models_dir)

    # Load scaler
    scaler = joblib.load(models_path / "scaler.pkl")

    # We need input_dim to build the model
    # Option 1: infer from scaler.mean_.shape
    input_dim = scaler.mean_.shape[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    state_dict = torch.load(models_path / "autoencoder_best.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler, device


def compute_reconstruction_error(model, x_scaled: np.ndarray, device: str) -> np.ndarray:
    """
    Compute reconstruction error (MSE per sample) for already scaled data.

    x_scaled: shape (n_samples, n_features)
    """
    if x_scaled.ndim == 1:
        x_scaled = x_scaled.reshape(1, -1)

    model.eval()
    with torch.no_grad():
        tensor_x = torch.from_numpy(x_scaled.astype(np.float32)).to(device)
        recon = model(tensor_x)
        errors = torch.mean((tensor_x - recon) ** 2, dim=1)

    return errors.cpu().numpy()


def score_samples(
    X_raw: np.ndarray,
    models_dir: str = "models",
    latent_dim: int = 16,
    threshold: float = 0.588957,  # from eval.py best F1 threshold
):
    """
    High-level helper to:
      1) Load model + scaler
      2) Scale raw input
      3) Compute reconstruction errors
      4) Flag anomalies based on threshold

    Args:
        X_raw: np.ndarray of shape (n_samples, n_features)
        threshold: reconstruction error above which a sample is flagged as anomaly

    Returns:
        errors: np.ndarray of shape (n_samples,)
        is_anomaly: np.ndarray of shape (n_samples,), values 0 or 1
    """
    if X_raw.ndim == 1:
        X_raw = X_raw.reshape(1, -1)

    model, scaler, device = load_model_and_scaler(models_dir=models_dir, latent_dim=latent_dim)

    # Scale using training scaler
    X_scaled = scaler.transform(X_raw)

    # Compute reconstruction errors
    errors = compute_reconstruction_error(model, X_scaled, device=device)

    # Anomaly flag
    is_anomaly = (errors > threshold).astype(int)

    return errors, is_anomaly


def demo_using_saved_test_sample():
    """
    Demo function (optional):
    Uses one sample from the saved X_test_scaled.npy but
    NOTE: that file is already scaled, so here we bypass the scaler.
    """
    models_path = Path("models")

    # Load already scaled test data
    X_test_scaled = np.load(models_path / "X_test_scaled.npy")
    y_test = np.load(models_path / "y_test.npy")

    model, scaler, device = load_model_and_scaler(models_dir="models", latent_dim=16)

    # Take a few samples
    idxs = [0, 1, 2, 100, 200]
    X_subset = X_test_scaled[idxs]
    y_subset = y_test[idxs]

    errors = compute_reconstruction_error(model, X_subset, device=device)
    is_anomaly = (errors > 0.588957).astype(int)

    for i, idx in enumerate(idxs):
        print(f"Sample index {idx}:")
        print(f"  True label: {y_subset[i]}")
        print(f"  Reconstruction error: {errors[i]:.6f}")
        print(f"  Predicted anomaly (1=yes, 0=no): {is_anomaly[i]}")
        print("")


if __name__ == "__main__":
    demo_using_saved_test_sample()
