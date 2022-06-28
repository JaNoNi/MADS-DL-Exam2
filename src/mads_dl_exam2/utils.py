import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader, Dataset


def compute_metrics(model, x: torch.Tensor, y) -> dict:
    y_pred = model(x).argmax(dim=1)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred, average="macro"),
        "precision": precision_score(y, y_pred, average="macro", labels=np.unique(y_pred)),
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
