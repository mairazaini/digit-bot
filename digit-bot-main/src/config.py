# src/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    seed: int = 42
    data_dir: str = "data"
    batch_size: int = 128
    num_workers: int = 2
    val_ratio: float = 0.1  # validation split

# THIS OBJECT MUST EXIST
CFG = Config()
