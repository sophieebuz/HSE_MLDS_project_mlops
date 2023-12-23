import torch
from torch import nn


class Classification(nn.Module):
    def __init__(self, in_features: int, num_classes: int, hidden_dim: int = 100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(
                in_features=in_features, out_features=hidden_dim, dtype=torch.float64
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=hidden_dim, out_features=hidden_dim, dtype=torch.float64
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=hidden_dim, out_features=num_classes, dtype=torch.float64
            ),
        )

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)
