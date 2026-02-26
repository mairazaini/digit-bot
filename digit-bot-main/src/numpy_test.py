# src/numpy_test.py
from src.config import CFG
from src.data.mnist_numpy import load_mnist_numpy


def main():
    split = load_mnist_numpy(
        data_dir=CFG.data_dir,
        val_ratio=CFG.val_ratio,
        seed=CFG.seed
    )

    print("X_train:", split.X_train.shape,
          split.X_train.dtype,
          split.X_train.min(),
          split.X_train.max())

    print("y_train:", split.y_train.shape, split.y_train.dtype)
    print("X_val:", split.X_val.shape)
    print("X_test:", split.X_test.shape)


if __name__ == "__main__":
    main()
