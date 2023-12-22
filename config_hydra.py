from dataclasses import dataclass


@dataclass
class Seed:
    seed: int


@dataclass
class Loader:
    batch_size: int
    shuffle: bool
    pin_memory: bool


@dataclass
class Optimizer:
    lr: float


@dataclass
class VineConfig:
    seed: Seed
    loader_train: Loader
    loader_test: Loader
    device: str
    num_epochs: int
    optimizer: Optimizer
