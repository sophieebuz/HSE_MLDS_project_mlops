from dataclasses import dataclass


@dataclass
class Paths:
    dftrain: str
    dftest: str
    model: str
    pred: str


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
class RunMlflow:
    run_mlflow: bool
    exp_name: str
    run_name: str


@dataclass
class VineConfig:
    paths: Paths
    seed: Seed
    loader_train: Loader
    loader_test: Loader
    device: str
    num_epochs: int
    plot: bool
    optimizer: Optimizer
    run_mlflow: RunMlflow
