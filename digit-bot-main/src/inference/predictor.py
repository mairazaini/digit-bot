# src/inference/predictor.py
# One simple prediction API for the GUI:
#   pred, conf = predict_pil(pil_image, model_name="svm_hog")
#
# Supports:
# - "svm_hog"  (joblib .pkl)
# - "mlp"      (torch .pt)
# - "cnn"      (torch .pt)

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps

import joblib

import torch
import torch.nn.functional as F

from src.data.features import hog_features
from src.models.mlp import MLP
from src.models.cnn_lenet import LeNetMNIST


# Small utilities
def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Stable softmax for numpy arrays."""
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return bounding box (xmin, ymin, xmax, ymax) of True pixels, or None if empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def preprocess_pil_to_mnist(pil_img: Image.Image) -> np.ndarray:
    """
    Convert any PIL image into an MNIST-like numpy image:
    - grayscale
    - invert if background seems white
    - crop to digit bounding box
    - resize to 28x28
    - normalize to float32 in [0,1]
    Output shape: (28, 28) float32
    """
    # 1) Grayscale
    img = pil_img.convert("L")

    # 2) Convert to numpy [0..255]
    arr = np.array(img, dtype=np.uint8)

    # 3) Heuristic invert:
    # If the image is mostly bright, assume white background + dark stroke -> invert to MNIST style.
    if arr.mean() > 127:
        arr = 255 - arr

    # 4) Create a mask of "ink" pixels (digit)
    # Threshold is small because after inversion digit pixels are brighter.
    mask = arr > 30
    bbox = _bbox_from_mask(mask)

    # If we found a digit region, crop to it; otherwise keep full image
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        arr = arr[y0:y1, x0:x1]

    # 5) Convert back to PIL for resizing
    img2 = Image.fromarray(arr, mode="L")

    # 6) Make square by padding (helps keep digit aspect ratio)
    w, h = img2.size
    side = max(w, h)
    pad_left = (side - w) // 2
    pad_top = (side - h) // 2
    pad_right = side - w - pad_left
    pad_bottom = side - h - pad_top
    img2 = ImageOps.expand(img2, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)

    # 7) Resize to 28x28
    img2 = img2.resize((28, 28), resample=Image.BILINEAR)

    # 8) Normalize to [0,1] float32
    out = np.array(img2, dtype=np.float32) / 255.0
    return out

# Model bundle (loads once)
@dataclass
class ModelBundle:
    device: torch.device
    svm_hog: Optional[object] = None
    mlp: Optional[torch.nn.Module] = None
    cnn: Optional[torch.nn.Module] = None


_BUNDLE: Optional[ModelBundle] = None


def load_models(
    model_dir: str = "outputs/models",
    device: Optional[str] = None,
) -> ModelBundle:
    """
    Load all available models once and cache them for GUI usage.
    """
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    bundle = ModelBundle(device=dev)

    # SVM + HOG (joblib)
    svm_path = os.path.join(model_dir, "svm_hog.pkl")
    if os.path.exists(svm_path):
        bundle.svm_hog = joblib.load(svm_path)

    # MLP (torch)
    mlp_path = os.path.join(model_dir, "mlp.pt")
    if os.path.exists(mlp_path):
        m = MLP().to(dev)
        m.load_state_dict(torch.load(mlp_path, map_location=dev))
        m.eval()
        bundle.mlp = m

    # CNN (torch)
    cnn_path = os.path.join(model_dir, "cnn_lenet.pt")
    if os.path.exists(cnn_path):
        c = LeNetMNIST().to(dev)
        c.load_state_dict(torch.load(cnn_path, map_location=dev))
        c.eval()
        bundle.cnn = c

    _BUNDLE = bundle
    return bundle


def available_models(model_dir: str = "outputs/models") -> Dict[str, str]:
    """
    Return {model_key: human_label} for models that exist on disk.
    """
    models = {}
    if os.path.exists(os.path.join(model_dir, "svm_hog.pkl")):
        models["svm_hog"] = "SVM (HOG)"
    if os.path.exists(os.path.join(model_dir, "mlp.pt")):
        models["mlp"] = "MLP"
    if os.path.exists(os.path.join(model_dir, "cnn_lenet.pt")):
        models["cnn"] = "CNN (LeNet)"
    return models


# Public prediction function
def predict_pil(
    pil_img: Image.Image,
    model_name: str,
    model_dir: str = "outputs/models",
) -> Tuple[int, float]:
    """
    Predict digit from a PIL image using the selected model.

    Returns:
        pred_digit (int), confidence (float in [0,1])
    """
    bundle = load_models(model_dir=model_dir)

    x28 = preprocess_pil_to_mnist(pil_img)  # (28,28) float32 [0,1]

    if model_name == "svm_hog":
        if bundle.svm_hog is None:
            raise FileNotFoundError("SVM model not found (outputs/models/svm_hog.pkl).")

        # HOG features expect batch input (N,28,28)
        X_feat = hog_features(x28[None, ...])  # (1,D)
        svm = bundle.svm_hog

        # LinearSVC does not provide calibrated probabilities.
        # Use decision_function scores and softmax them to get a confidence-like value.
        scores = svm.decision_function(X_feat)
        if scores.ndim == 1:
            scores = scores[None, :]
        probs = _softmax_np(scores)

        pred = int(np.argmax(probs, axis=1)[0])
        conf = float(np.max(probs, axis=1)[0])
        return pred, conf

    elif model_name in ("mlp", "cnn"):
        model = bundle.mlp if model_name == "mlp" else bundle.cnn
        if model is None:
            raise FileNotFoundError(f"{model_name} model not found in outputs/models/.")

        # Torch model expects (N,1,28,28)
        xb = torch.from_numpy(x28).unsqueeze(0).unsqueeze(0).to(bundle.device)  # (1,1,28,28)
        with torch.no_grad():
            logits = model(xb)
            probs = F.softmax(logits, dim=1).cpu().numpy()

        pred = int(np.argmax(probs, axis=1)[0])
        conf = float(np.max(probs, axis=1)[0])
        return pred, conf

    else:
        raise ValueError(f"Unknown model_name='{model_name}'. Use one of {list(available_models(model_dir).keys())}.")
