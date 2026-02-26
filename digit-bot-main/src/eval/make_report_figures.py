# src/eval/make_report_figures.py
from __future__ import annotations

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score

from src.config import CFG
from src.data.mnist_numpy import load_mnist_numpy
from src.data.features import hog_features
from src.models.cnn_lenet import LeNetMNIST
from src.models.mlp import MLP


# -------------------------
# Plot helpers
# -------------------------
def save_cm_figure(cm: np.ndarray, title: str, out_path: str) -> None:
    # make sure the output folder exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Plot confusion matrix as an image
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    
    
# Write numbers into each cell to make the matrix is readable 
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def save_learning_curve(log_path: str, title: str, out_path: str) -> None:
    # Learning curve data is saved during training (JSON files in outputs/logs/)
    if not os.path.exists(log_path):
        print(f"Skipping learning curve (missing log): {log_path}")
        return
    
    # Load training history from JSON
    with open(log_path, "r", encoding="utf-8") as f:
        hist = json.load(f)
        
    # Extract train/val accuracy arrays
    train_acc = hist["train_acc"]
    val_acc = hist["val_acc"]
    epochs = list(range(1, len(train_acc) + 1))

    # Ensure output folder exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Plot accuracy vs epoch
    plt.figure()
    plt.plot(epochs, train_acc, label="Train accuracy")
    plt.plot(epochs, val_acc, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# PyTorch helpers
def to_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    X_t = torch.from_numpy(X).unsqueeze(1)  # (N,1,28,28)
    y_t = torch.from_numpy(y)
    return TensorDataset(X_t, y_t)


@torch.no_grad()
def torch_predict(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    preds = []
    targets = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(pred)
        targets.append(yb.numpy())
    return np.concatenate(targets), np.concatenate(preds)


def main():
    # Load MNIST split
    split = load_mnist_numpy(data_dir=CFG.data_dir, val_ratio=CFG.val_ratio, seed=CFG.seed)
    y_test = split.y_test

    os.makedirs("outputs/figures", exist_ok=True)

    # 1) Confusion Matrix: SVM + HOG
    svm_path = "outputs/models/svm_hog.pkl"
    if os.path.exists(svm_path):
        svm = joblib.load(svm_path)
        X_test_hog = hog_features(split.X_test)
        svm_pred = svm.predict(X_test_hog)
        cm = confusion_matrix(y_test, svm_pred)
        acc = accuracy_score(y_test, svm_pred)
        save_cm_figure(cm, f"SVM + HOG confusion matrix (acc={acc:.4f})", "outputs/figures/cm_svm_hog.png")
    else:
        print(f"Skipping SVM CM (missing model): {svm_path}")

    # Prepare torch loader once for CNN+MLP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = to_tensor_dataset(split.X_test, split.y_test)
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

    # 2) Confusion Matrix: CNN
    cnn_path = "outputs/models/cnn_lenet.pt"
    if os.path.exists(cnn_path):
        cnn = LeNetMNIST().to(device)
        cnn.load_state_dict(torch.load(cnn_path, map_location=device))
        y_true, y_pred = torch_predict(cnn, test_loader, device)
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        save_cm_figure(cm, f"CNN confusion matrix (acc={acc:.4f})", "outputs/figures/cm_cnn.png")
    else:
        print(f"Skipping CNN CM (missing model): {cnn_path}")

    # 3) Confusion Matrix: MLP
    mlp_path = "outputs/models/mlp.pt"
    if os.path.exists(mlp_path):
        mlp = MLP().to(device)
        mlp.load_state_dict(torch.load(mlp_path, map_location=device))
        y_true, y_pred = torch_predict(mlp, test_loader, device)
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        save_cm_figure(cm, f"MLP confusion matrix (acc={acc:.4f})", "outputs/figures/cm_mlp.png")
    else:
        print(f"Skipping MLP CM (missing model): {mlp_path}")

    # 4) Learning curves (from saved logs)
    save_learning_curve(
        log_path="outputs/logs/cnn_history.json",
        title="CNN learning curve (accuracy)",
        out_path="outputs/figures/cnn_learning_curve_accuracy.png",
    )

    save_learning_curve(
        log_path="outputs/logs/mlp_history.json",
        title="MLP learning curve (accuracy)",
        out_path="outputs/figures/mlp_learning_curve_accuracy.png",
    )

    print("\nDone. Check outputs/figures/ for all report figures.")


if __name__ == "__main__":
    main()
