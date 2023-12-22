import os

import hydra
import pandas as pd
import torch
from hydra.core.config_store import ConfigStore
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from config_hydra import VineConfig
from utils.dataset import VineDataset, collate_fn
from utils.model import Classification
from utils.seed import seed_everything, seed_worker
from utils.trainer import inference


cs = ConfigStore.instance()
cs.store(name="vine_config", node=VineConfig)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: VineConfig) -> None:
    SEED = cfg.seed.seed
    seed_everything(SEED)
    g = torch.Generator()
    g.manual_seed(0)

    os.system("dvc pull")

    path_test = cfg.paths.dftest
    test_set = VineDataset(path_test)
    test_loader = DataLoader(
        test_set, collate_fn=collate_fn, worker_init_fn=seed_worker, **cfg.loader_test
    )

    model_save_file = cfg.paths.model
    ckpt = torch.load(model_save_file)

    DEVICE = cfg.device
    model = Classification(
        in_features=test_set.num_features, num_classes=test_set.num_classes
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), **cfg.optimizer)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])

    y_true, y_pred = inference(test_loader, model, DEVICE)
    target_names = ["class 0", "class 1", "class 2"]
    print(
        classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    )

    pd.DataFrame(y_true).to_csv(cfg.paths.pred, index=False)


if __name__ == "__main__":
    main()
