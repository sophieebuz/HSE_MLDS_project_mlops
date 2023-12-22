import pandas as pd
import torch
from torch.utils.data import Dataset


class VineDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.df = torch.tensor(pd.read_csv(path).values)
        self.names = pd.read_csv(path).columns
        self.num_classes = pd.read_csv(path)["target"].nunique()
        self.num_features = len(self.names) - 1

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        return self.df[idx]


def collate_fn(batch):
    batch_x, batch_y = [], []
    for elem in batch:
        batch_x += [elem[:-1]]
        batch_y += [elem[-1]]

    batch_x = torch.vstack(batch_x)
    batch_y = torch.vstack(batch_y)
    return batch_x, batch_y.to(torch.long)
