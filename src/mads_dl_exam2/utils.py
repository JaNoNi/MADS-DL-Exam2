import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


def plot_multi_conf(results, labels, title):
    """Plot ConfusionMatrix.

    Args:
        results (Dictionary): Dictionary with confusion matrix arrays.
        labels (List): Humanreadable ticklabel names.
        title (String): Title for the subplots.
    """
    plots = len(results.keys())
    fig, axes = plt.subplots(1, plots, figsize=(12, 6), dpi=200)
    fig.suptitle(title)
    for i, (key, item) in enumerate(results.items()):
        ConfusionMatrixDisplay(item, display_labels=labels).plot(ax=axes[i], colorbar=False)
        axes[i].set_title(key)
        axes[i].grid(False)
        if i:
            axes[i].get_yaxis().set_visible(False)
    fig.tight_layout()


def compute_metrics(model, x: torch.Tensor, y) -> dict:
    y_pred = model.predict(x)
    y_pred_proba = model.predict_proba(x)
    return {
        "Accuracy": round(accuracy_score(y, y_pred), 4),
        "Balanced Accuracy": round(balanced_accuracy_score(y, y_pred), 4),
        "ROC AUC": round(roc_auc_score(y, y_pred_proba, multi_class="ovr"), 4),
        "F1": round(f1_score(y, y_pred, average="macro"), 4),
        "Precision": round(
            precision_score(y, y_pred, average="macro", labels=np.unique(y_pred)), 4
        ),
        "confusion_matrix": confusion_matrix(y, y_pred),
    }


class Dataset(Dataset):
    def __init__(
        self, data: pd.DataFrame, classes: list, device: torch.device = torch.device("cpu")
    ):
        self.data = data
        self.classes = classes
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        raw_data = data_row.loc[list(set(data_row.index) - set(self.classes))]
        labels = data_row[self.classes]
        return dict(
            data=torch.tensor(raw_data, dtype=torch.float).to(self.device),
            labels=torch.tensor(labels, dtype=torch.long).to(self.device).squeeze_(),
        )


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        test_df,
        val_df,
        classes: list,
        batch_size: int = 8,
        num_workers: int = 2,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df
        self.classes = classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    def setup(self, stage=None):
        self.train_dataset = Dataset(data=self.train_df, classes=self.classes, device=self.device)
        self.val_dataset = Dataset(data=self.val_df, classes=self.classes, device=self.device)
        self.test_dataset = Dataset(data=self.test_df, classes=self.classes, device=self.device)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


def main():
    return ()


if __name__ == "__main__":
    main()
