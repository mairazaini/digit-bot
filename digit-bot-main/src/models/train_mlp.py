
# Trains a simple Multi-Layer Perceptron (MLP) on MNIST
# Saves the best model and training history for report figures
from __future__ import annotations
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix

from src.config import CFG
from src.data.mnist_numpy import load_mnist_numpy
from src.models.mlp import MLP


def to_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    # Convert NumPy arrays into PyTorch tensors
    X_t = torch.from_numpy(X).unsqueeze(1)  # (N,1,28,28)
    y_t = torch.from_numpy(y)
    return TensorDataset(X_t, y_t)


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None = None) -> float:
    # Evaluate model accuracy on a dataset (no gradient tracking)
    model.eval()
    preds, targets = [], []
    for i, (xb, yb) in enumerate(loader):
        # Optionally limit number of batches for faster evaluation
        if max_batches is not None and i >= max_batches:
            break
        xb = xb.to(device)
        logits = model(xb)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        targets.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    return accuracy_score(y_true, y_pred)


def main():
    # Select GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(CFG.seed)

    split = load_mnist_numpy(data_dir=CFG.data_dir, val_ratio=CFG.val_ratio, seed=CFG.seed)

    train_ds = to_tensor_dataset(split.X_train, split.y_train)
    val_ds = to_tensor_dataset(split.X_val, split.y_val)
    test_ds = to_tensor_dataset(split.X_test, split.y_test)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
    test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    history = {"train_acc": [], "val_acc": []}
    best_val = -1.0

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_acc = evaluate(model, val_loader, device)
        train_acc = evaluate(model, train_loader, device, max_batches=200)

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{epochs} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "outputs/models/mlp.pt")

    # Test evaluation
    best_model = MLP().to(device)
    best_model.load_state_dict(torch.load("outputs/models/mlp.pt", map_location=device))
    best_model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = best_model(xb)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets.append(yb.numpy())

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)

    test_acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\nBest val_acc:", best_val)
    print("Test accuracy:", test_acc)
    print("Confusion matrix:\n", cm)

    with open("outputs/logs/mlp_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\nSaved model to outputs/models/mlp.pt")
    print("Saved history to outputs/logs/mlp_history.json")


if __name__ == "__main__":
    main()
