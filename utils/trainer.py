from collections import defaultdict
from functools import partial

import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from utils.plots import plot_losses


def training_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.modules.loss,
    train_loader: DataLoader,
    device: str,
    run_mlflow: bool,
):
    num_batches = 0.0
    train_loss = 0.0
    metrics = defaultdict(float)
    model.train()

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, torch.flatten(target))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        for m_name, m in {
            "accuracy": partial(accuracy_score),
            "precision": partial(precision_score, average="macro", zero_division=0),
            "recall": partial(recall_score, average="macro", zero_division=0),
            "f1 macro": partial(f1_score, average="macro"),
        }.items():
            metrics[m_name] += m(
                target.detach().numpy(), np.argmax(logits.detach().numpy(), axis=-1)
            )

        num_batches += 1

    train_loss /= num_batches
    for m_name in metrics:
        metrics[m_name] /= num_batches
    if run_mlflow:
        mlflow.log_metric("accuracy_train", metrics["accuracy"])
        mlflow.log_metric("precision_train", metrics["precision"])
        mlflow.log_metric("recall_train", metrics["recall"])
        mlflow.log_metric("f1 macro_train", metrics["f1 macro"])
    return train_loss, metrics


@torch.no_grad()
def validation_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.modules.loss,
    test_loader: DataLoader,
    device: str,
    run_mlflow: bool,
):
    num_batches = 0.0
    test_loss = 0.0
    metrics = defaultdict(float)
    model.eval()
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        logits = model(data)
        loss = criterion(logits, torch.flatten(target))

        test_loss += loss.item()
        for m_name, m in {
            "accuracy": partial(accuracy_score),
            "precision": partial(precision_score, average="macro", zero_division=0),
            "recall": partial(recall_score, average="macro", zero_division=0),
            "f1 macro": partial(f1_score, average="macro"),
        }.items():
            metrics[m_name] += m(
                target.detach().numpy(), np.argmax(logits.detach().numpy(), axis=-1)
            )

        num_batches += 1

    test_loss /= num_batches
    for m_name in metrics:
        metrics[m_name] /= num_batches
    if run_mlflow:
        mlflow.log_metric("accuracy_test", metrics["accuracy"])
        mlflow.log_metric("precision_test", metrics["precision"])
        mlflow.log_metric("recall_test", metrics["recall"])
        mlflow.log_metric("f1 macro_test", metrics["f1 macro"])
    return test_loss, metrics


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: torch.nn.modules.loss,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    device: str,
    plot: bool,
    run_mlflow: bool,
):
    train_losses = []
    test_losses = []
    train_metrics, test_metrics = defaultdict(list), defaultdict(list)

    for _epoch in range(1, num_epochs + 1):
        train_loss, train_metric = training_epoch(
            model,
            optimizer,
            criterion,
            train_loader,
            device,
            run_mlflow,
        )
        test_loss, test_metric = validation_epoch(
            model,
            criterion,
            test_loader,
            device,
            run_mlflow,
        )

        if scheduler is not None:
            scheduler.step()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        for m_name, m_value in train_metric.items():
            train_metrics[m_name].append(m_value.item())
        for m_name, m_value in test_metric.items():
            test_metrics[m_name].append(m_value.item())

    if plot:
        plot_losses(train_losses, test_losses, train_metrics, test_metrics)

    return train_losses, test_losses, train_metrics, test_metrics


def inference(loader: DataLoader, model: torch.nn.Module, device: str):
    y_preds = []
    y_true = []
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        logits = model(data)
        y_pred = np.argmax(logits.detach().numpy(), axis=-1)

        y_preds.extend(y_pred)
        y_true.extend(target.detach().numpy())
    return y_true, y_preds
