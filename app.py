
import streamlit as st
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.metrics import precision_recall_curve
from src.model import build_autoencoder


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Credit Card Fraud Detection")
st.write("Deep learning using an Autoencoder for anomaly detection")

MODEL_DIR = Path("models")


@st.cache_resource
def load_model_and_data():
    X = np.load(MODEL_DIR / "X_test_scaled.npy")
    y = np.load(MODEL_DIR / "y_test.npy")

    model = build_autoencoder(
        input_dim=X.shape[1],
        latent_dim=16,
    )

    state = torch.load(
        MODEL_DIR / "autoencoder_best.pth",
        map_location="cpu",
    )
    model.load_state_dict(state)
    model.eval()

    return model, X, y


@st.cache_data
def calculate_errors(X):
    model, _, _ = load_model_and_data()
    errors = []

    with torch.no_grad():
        for start in range(0, len(X), 1024):
            batch = torch.tensor(
                X[start:start + 1024],
                dtype=torch.float32,
            )
            reconstructed = model(batch)
            batch_errors = torch.mean(
                (batch - reconstructed) ** 2,
                dim=1,
            )
            errors.extend(batch_errors.numpy().tolist())

    return np.array(errors)


try:
    model, X, y = load_model_and_data()
    errors = calculate_errors(X)

except Exception as error:
    st.error(f"Could not load the model or test data: {error}")
    st.stop()


# Find a threshold from the test labels for demonstration.
from sklearn.model_selection import train_test_split

# Split saved samples into threshold-selection and evaluation sets.
# The autoencoder itself is not retrained.
indices = np.arange(len(y))

validation_idx, test_idx = train_test_split(
    indices,
    test_size=0.5,
    random_state=42,
    stratify=y,
)

# Select threshold using validation samples only.

from sklearn.model_selection import train_test_split

# Split saved samples into threshold-selection and evaluation sets.
# The autoencoder itself is not retrained.
indices = np.arange(len(y))

validation_idx, test_idx = train_test_split(
    indices,
    test_size=0.5,
    random_state=42,
    stratify=y,
)

# Select threshold using validation samples only.
validation_errors = errors[validation_idx]
validation_labels = y[validation_idx]

p, r, candidate_thresholds = precision_recall_curve(
    validation_labels,
    validation_errors,
)

if len(candidate_thresholds) > 0:
    f1_values = (
        2 * p[:-1] * r[:-1]
        / (p[:-1] + r[:-1] + 1e-12)
    )
    best_threshold = float(
        candidate_thresholds[np.argmax(f1_values)]
    )
else:
    best_threshold = float(np.median(validation_errors))

# Keep only the separate evaluation samples from here onward.
X = X[test_idx]
y = y[test_idx]
errors = errors[test_idx]

st.sidebar.caption(
    "Threshold selected on validation samples; "
    "metrics calculated on separate evaluation samples."
)

threshold = st.sidebar.slider(
    "Anomaly threshold",
    min_value=0.0,
    max_value=float(
        max(errors.max(), best_threshold * 1.1, 0.001)
    ),
    value=best_threshold,
    step=float(
        max(errors.max(), best_threshold * 1.1, 0.001)
        / 1000
    ),
    format="%.6f",
)

predictions = (errors > threshold).astype(int)



# Performance dashboard
st.header("1. Model Performance")

roc_auc = roc_auc_score(y, errors)
precision = precision_score(y, predictions, zero_division=0)
recall = recall_score(y, predictions, zero_division=0)
f1 = f1_score(y, predictions, zero_division=0)

col1, col2, col3, col4 = st.columns(4)

col1.metric("ROC-AUC", f"{roc_auc:.4f}")
col2.metric("Precision", f"{precision:.4f}")
col3.metric("Recall", f"{recall:.4f}")
col4.metric("F1-score", f"{f1:.4f}")

st.caption(
    "Metrics are recalculated on the saved test set. "
    "The threshold is selected using test labels for this "
    "educational demonstration, so these are not independent "
    "unseen-test performance estimates."
)


# Confusion matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y, predictions, labels=[0, 1])

cm_df = pd.DataFrame(
    cm,
    index=["Actual legitimate", "Actual fraud"],
    columns=["Predicted legitimate", "Predicted fraud"],
)

st.dataframe(cm_df, use_container_width=True)

tn, fp, fn, tp = cm.ravel()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Correct legitimate", f"{tn:,}")
c2.metric("False alarms", f"{fp:,}")
c3.metric("Missed fraud", f"{fn:,}")
c4.metric("Fraud detected", f"{tp:,}")


# Transaction explorer
st.header("2. Transaction Explorer")

st.write(
    "Select a saved test transaction to inspect its "
    "reconstruction error and prediction."
)

index = st.slider(
    "Test transaction index",
    min_value=0,
    max_value=len(X) - 1,
    value=0,
)

actual = int(y[index])
predicted = int(predictions[index])

left, right = st.columns(2)

left.metric("Reconstruction error", f"{errors[index]:.6f}")
right.metric("Current threshold", f"{threshold:.6f}")

if predicted == 1:
    st.warning("⚠️ Flagged as an anomaly")
else:
    st.success("✅ Not flagged as an anomaly")

st.write(
    "Actual dataset label:",
    "Fraud" if actual == 1 else "Legitimate",
)

st.write(
    "Prediction:",
    "Anomaly" if predicted == 1 else "Not flagged",
)


# Browse several transactions
st.header("3. Sample Transaction Table")

number_of_rows = st.selectbox(
    "Number of transactions to display",
    [10, 25, 50, 100],
    index=1,
)

table = pd.DataFrame({
    "Transaction index": np.arange(number_of_rows),
    "Reconstruction error": errors[:number_of_rows],
    "Predicted anomaly": predictions[:number_of_rows],
    "Actual label": [
        "Fraud" if value == 1 else "Legitimate"
        for value in y[:number_of_rows]
    ],
})

st.dataframe(table, use_container_width=True)

st.caption(
    "Educational demonstration using saved test samples. "
    "An anomaly flag is not proof of fraud."
)

