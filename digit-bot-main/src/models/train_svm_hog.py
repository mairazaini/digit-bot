# src/models/train_svm_hog.py
from __future__ import annotations
import os
import joblib
import numpy as np

# We import CalibratedClassifierCV to fix the "predict_proba" error
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.config import CFG
from src.data.mnist_numpy import load_mnist_numpy
from src.data.features import hog_features

def main():
    print("1. Loading Data...")
    split = load_mnist_numpy(
        data_dir=CFG.data_dir,
        val_ratio=CFG.val_ratio,
        seed=CFG.seed
    )

    print("2. Extracting HOG Features (This takes a moment)...")
    X_train = hog_features(split.X_train)
    X_test = hog_features(split.X_test)
    
    # We combine train and validation for the final training to get maximum accuracy
    if split.X_val is not None:
        X_val_feats = hog_features(split.X_val)
        X_train = np.concatenate((X_train, X_val_feats))
        split.y_train = np.concatenate((split.y_train, split.y_val))

    print("3. Training Calibrated SVM...")
    # This gives the model the ability to output "confidence percentages"
    base_svm = LinearSVC(dual=False, C=1.0, max_iter=5000, random_state=CFG.seed)
    
    clf = CalibratedClassifierCV(estimator=base_svm, cv=3)
    clf.fit(X_train, split.y_train)

    # 4) Evaluation
    print("Evaluating on Test Set...")
    test_pred = clf.predict(X_test)
    acc = accuracy_score(split.y_test, test_pred)

    print(f"\nTest accuracy: {acc:.4f}")
    
    print("\nClassification report (test):")
    print(classification_report(split.y_test, test_pred, digits=4))

    print("\nConfusion matrix (test):")
    print(confusion_matrix(split.y_test, test_pred))

    # 5) Save model
    os.makedirs("outputs/models", exist_ok=True)
    save_path = "outputs/models/svm_hog.pkl"
    joblib.dump(clf, save_path)
    print(f"\nSUCCESS! Model saved to {save_path}")

if __name__ == "__main__":
    main()