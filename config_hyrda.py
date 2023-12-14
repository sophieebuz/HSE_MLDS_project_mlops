from dataclasses import dataclass


@dataclass
class Params:
    n_estimators: int
    learning_rate: float
    depth: int
    random_state: int
    loss_function: str
    eval_metric: str
    verbose: int


@dataclass
class CatBoostConfig:
    params: Params
