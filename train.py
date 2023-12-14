import os

import hydra
import pandas as pd
from catboost import CatBoostClassifier
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from config_hyrda import CatBoostConfig


cs = ConfigStore.instance()
cs.store(name="catboost_config", node=CatBoostConfig)


@hydra.main(config_path="conf_hydra", config_name="config", version_base=None)
def main(cfg: CatBoostConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    dftrain = pd.read_csv("data/dftrain.csv")

    clf = CatBoostClassifier(**cfg.params)

    clf.fit(dftrain, dftrain["target"])

    model_save_file = "Catboost_model.cbm"

    if os.path.exists(model_save_file):
        os.remove(model_save_file)

    clf.save_model(model_save_file, format="cbm")


if __name__ == "__main__":
    main()
