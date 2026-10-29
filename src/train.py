import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.optim import Adam

from src.model import build_autoencoder


def load_creditcard_data(csv_path: str):
    """
    Load the Credit Card Fraud dataset from a CSV file.
    Expected to have a 'Class' column where:
      - 0 = normal transaction
      - 1 = fraud (anomaly)
    """
    df = pd.read_csv(csv_path)
    if 'Class' not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column for labels.")
    
    X = df.drop(columns=['Class']).values.astype(np.float32)
    y = df['Class'].values.astype(np.float32)
    return X, y

def prepare_datasets(X , y, test_size= 0.2, val_size = 0.1, random_state=42):
    """
    Prepare train/val/test splits.

    Strategy:
      - Train and validation sets use only normal samples (y == 0).
      - Test set contains a mix of normal and anomaly samples.
    """
    normal_mask = (y==0)
    anomaly_mask = (y==1)

    X_normal = X[normal_mask]
    y_normal = y[normal_mask]
    
    X_anom = X[anomaly_mask]
    y_anom = y[anomaly_mask]

    # train/val split on normal data only
    X_train_norm, X_val_norm, y_train_norm, y_val_train = train_test_split(
        X_normal,
        y_normal,
        test_size=val_size,
        random_state=random_state,
        stratify=y_normal
    )

    # build a test set that includes all anomalies and a subset of normals
    X_normal_for_test, _, y_normal_for_test, _ = train_test_split(
        X_normal,
        y_normal,
        test_size=1 - test_size,
        random_state=random_state,
        stratify=y_normal
    )

    X_test = np.concatenate([X_normal_for_test, X_anom], axis=0)
    y_test = np.concatenate([y_normal_for_test, y_anom], axis=0)

    return (X_train_norm, y_train_norm), (X_val_norm, y_val_train), (X_test, y_test)

def scale_data(X_train, X_val, X_test):
    """
    Fit a StandardScaler on the training data and transform train/val/test.
    """
    scaler = StandardScaler()
    X_train_scaler = scaler.fit_transform(X_train)
    X_val_scaler = scaler.fit_transform(X_val)
    X_test_scaler = scaler.fit_transform(X_test)
    return scaler, X_train_scaler, X_val_scaler, X_test_scaler

def make_dataloader(X, batch_size:512, shuffle=True):
    tensor_X = torch.from_numpy(X)
    dataset = TensorDataset(tensor_X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def train_autoencoder(
    csv_path: str = "data/creditcard.csv",
    models_dir: str = "models",
    latent_dim: int = 16,
    batch_size: int = 512,
    lr: float = 1e-3,
    num_epochs: int = 30,
    device: str = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Path
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. load data
    print(f"Loading data from: {csv_path}")
    X, y = load_creditcard_data(csv_path)

    input_dim = X.shape[1]
    print(f"Input features: {input_dim}")

    # 2. prepare splits
    (X_train_norm, _), (X_val_norm, _), (X_test, y_test) = prepare_datasets(X, y)
    print(f"Train normal: {X_train_norm.shape[0]}, "
          f"Val normal: {X_val_norm.shape[0]}, "
          f"Test: {X_test.shape[0]}")
    
    # 3. scale
    scaler, X_train_scaled, X_val_scaled, X_test_scaled = scale_data(
        X_train_norm, X_val_norm, X_test
    )
    scaler_path = models_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to: {scaler_path}")

    # 4. dataloaders
    train_loader = make_dataloader(X_train_scaled, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(X_val_scaled, batch_size=batch_size, shuffle=False)

    # 5. model, loss, optimizer
    model = build_autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_model_path = models_dir / "autoencoder_best.pth"

    # 6. training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch_x,) in val_loader:
                batch_x = batch_x.to(device)
                recon = model(batch_x)
                loss = criterion(recon, batch_x)
                val_losses.append(loss.item())

        avg_val_loss = float(np.mean(val_losses))

        print(f"Epoch [{epoch}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.6f}  "
              f"Val Loss: {avg_val_loss:.6f}")

        # save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✔ New best model saved at val_loss={best_val_loss:.6f}")

    # also save final model
    final_model_path = models_dir / "autoencoder_last.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Best model: {best_model_path}, "
          f"Last model: {final_model_path}")

    # save test set arrays for later evaluation
    np.save(models_dir / "X_test_scaled.npy", X_test_scaled)
    np.save(models_dir / "y_test.npy", y_test)
    print("Saved X_test_scaled.npy and y_test.npy for eval.")

if __name__ == "__main__":
    train_autoencoder()



