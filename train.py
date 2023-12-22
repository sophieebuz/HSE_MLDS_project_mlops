import os
from pathlib import Path

import hydra
import mlflow
import torch
from hydra.core.config_store import ConfigStore
from mlflow.utils.git_utils import get_git_branch, get_git_commit
from torch import nn
from torch.utils.data import DataLoader

from config_hydra import VineConfig
from utils.dataset import VineDataset, collate_fn
from utils.model import Classification
from utils.seed import seed_everything, seed_worker
from utils.trainer import train


cs = ConfigStore.instance()
cs.store(name="vine_config", node=VineConfig)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: VineConfig) -> None:
    SEED = cfg.seed.seed
    seed_everything(SEED)
    g = torch.Generator()
    g.manual_seed(0)

    os.system("dvc pull")

    path_train = cfg.paths.dftrain
    train_set = VineDataset(path_train)
    path_test = cfg.paths.dftest
    test_set = VineDataset(path_test)

    train_loader = DataLoader(
        train_set, collate_fn=collate_fn, worker_init_fn=seed_worker, **cfg.loader_train
    )
    test_loader = DataLoader(
        test_set, collate_fn=collate_fn, worker_init_fn=seed_worker, **cfg.loader_test
    )

    DEVICE = cfg.device
    NUM_EPOCHS = cfg.num_epochs

    model = Classification(
        in_features=train_set.num_features, num_classes=train_set.num_classes
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), **cfg.optimizer)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    # print(mlflow.get_tracking_uri())
    # mlflow.set_tracking_uri("http://128.0.1.1:8080")
    # tracking_uri = mlflow.get_tracking_uri()
    # print(f"Current tracking uri: {tracking_uri}")
    run_mlflow = cfg.run_mlflow.run_mlflow
    if run_mlflow:
        exp_name = cfg.run_mlflow.exp_name
        experiment_id = mlflow.create_experiment(exp_name)
        print(experiment_id)
        mlflow.set_experiment(exp_name)
        # experiment_id=973347247450672054
        # experiment_id = 658355158496838367

        with mlflow.start_run(
            run_name=cfg.run_mlflow.run_name, experiment_id=experiment_id
        ):
            train_losses, test_losses, train_metrics, test_metrics = train(
                model,
                optimizer,
                None,
                criterion,
                train_loader,
                test_loader,
                NUM_EPOCHS,
                DEVICE,
                cfg.plot,
                run_mlflow,
            )
            mlflow.log_param("git commit id", get_git_commit(Path.cwd()))
            mlflow.log_param("git branch", get_git_branch(Path.cwd()))
            mlflow.log_params(dict(cfg))
    else:
        train_losses, test_losses, train_metrics, test_metrics = train(
            model,
            optimizer,
            None,
            criterion,
            train_loader,
            test_loader,
            NUM_EPOCHS,
            DEVICE,
            cfg.plot,
            run_mlflow,
        )

    # print(test_metrics["f1 macro"])

    model_save_file = cfg.paths.model
    if os.path.exists(model_save_file):
        os.remove(model_save_file)

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        model_save_file,
    )


if __name__ == "__main__":
    main()
