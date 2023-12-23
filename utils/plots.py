from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 15})


def plot_losses(
    train_losses: list,
    test_losses: list,
    train_metrics: defaultdict[str,],
    valid_metrics: defaultdict[str,],
) -> None:
    fig, axs = plt.subplots(3, 2, figsize=(13, 8))
    axs[0][0].plot(range(1, len(train_losses) + 1), train_losses, label="train")
    axs[0][0].plot(range(1, len(test_losses) + 1), test_losses, label="test")
    axs[0][0].set_ylabel("loss")

    for (ax1, ax2), train_m_name, valid_m_name in zip(
        ((0, 1), (1, 0), (1, 1), (2, 0), (2, 1)), train_metrics, valid_metrics
    ):
        train_m, valid_m = train_metrics[train_m_name], valid_metrics[valid_m_name]
        axs[ax1][ax2].plot(range(1, len(train_m) + 1), train_m, label="train")
        axs[ax1][ax2].plot(range(1, len(valid_m) + 1), valid_m, label="test")
        axs[ax1][ax2].set_ylabel(train_m_name)

    for ax1 in axs:
        for ax2 in ax1:
            ax2.set_xlabel("epoch")
            ax2.legend()

    plt.show()
