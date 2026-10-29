import numpy as np
from pathlib import Path

import torch
from torch import nn

from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report)
from src.model import build_autoencoder

def load_artifacts(models_dir: str = "models"):
    """
    Load saved test data and trained model weights.

    Expects:
      - models/scaler.pkl        (optional here, more useful for inference)
      - models/X_test_scaled.npy
      - models/y_test.npy
      - models/autoencoder_best.pth
    """
    model_path = Path(models_dir)
    X_test_scaled = np.load(model_path/"X_test_scaled.npy")
    y_test = np.load(model_path/"y_test.npy")

    # infer input_dim from X_test
    input_dim = X_test_scaled.shape[1]
    latent_dim = 16

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    state_dim = torch.load(model_path/"autoencoder_best.pth", map_location=device)
    model.load_state_dict(state_dim)
    model.eval()

    return model, X_test_scaled, y_test, device

def compute_reconstruction_errors(model, X: np.ndarray, device: str, batch_size: int = 1024):
    """
    Compute per-sample reconstruction error (MSE) for all samples in X.
    """
    model.eval()
    dataset = torch.from_numpy(X).to(device)
    errors = []

    with torch.no_grad():
        for i in range (0, len(dataset), batch_size):
            batch = dataset[i : i + batch_size]
            recon = model(batch)
            batch_errors = torch.mean((batch - recon) ** 2, dim=1)
            errors.append(batch_errors.cpu().numpy())

    return np.concatenate(errors, axis=0)

def choose_threshold_by_f1(y_true: np.ndarray, scores: np.ndarray):
    """
    Choose an anomaly threshold that approximately maximizes F1 score.
    We treat higher reconstruction error = more anomalous.
    """
    # Use precision-recall curve as a function of threshold
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    # Avoid division by zero
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1, precision, recall, thresholds

def evaluate_anomaly_detection(models_dir: str = "models", percentile_fallback: float = 99.5):
    """
    Full evaluation pipeline:
      1) Load model + test data
      2) Compute reconstruction errors
      3) Compute ROC AUC
      4) Choose threshold via PR-F1; fallback to percentile of normal errors if needed
      5) Print confusion matrix and classification report
    """
    print("loading Artifacts...")
    model, X_test_scaled, y_test, device = load_artifacts(models_dir=models_dir)

    print("Computing reconstruction errors on test set...")
    errors_test = compute_reconstruction_errors(model, X_test_scaled, device=device)
    assert errors_test.shape[0] == y_test.shape[0]

    # ROC AUC
    roc_auc = roc_auc_score(y_test, errors_test)
    print(f"\nROC AUC (using reconstruction error as anomaly score): {roc_auc:.4f}")

    # Best F1-based threshold
    best_threshold, best_f1, precision, recall, thresholds = choose_threshold_by_f1(
        y_true=y_test,
        scores=errors_test,
    )

    print(f"\nBest F1-based threshold: {best_threshold:.6f}")
    print(f"Best F1 score at this threshold: {best_f1:.4f}")

    # If something weird happens (no thresholds), fallback to percentile of normal errors
    if np.isnan(best_threshold) or np.isinf(best_threshold):
        print("\nF1-based threshold invalid, falling back to percentile-based threshold...")
        normal_errors = errors_test[y_test == 0]
        best_threshold = float(np.percentile(normal_errors, percentile_fallback))
        print(f"Percentile-based threshold ({percentile_fallback}% of normal errors): {best_threshold:.6f}")

    # Predictions: 1 = anomaly, 0 = normal
    y_pred = (errors_test > best_threshold).astype(int)

    # Confusion Metrics
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)

    print("\nClassification Report:")
    print(report)

    num_anom_true = int((y_test == 1).sum())    
    num_anom_pred = int((y_pred == 1).sum())   

    print(f"True anomalies in test set: {num_anom_true}")
    print(f"Predicted anomalies (y_pred == 1): {num_anom_pred}")

    return {
        "errors_test": errors_test,
        "y_test": y_test,
        "threshold": best_threshold,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


if __name__ == "__main__":
    evaluate_anomaly_detection() 
