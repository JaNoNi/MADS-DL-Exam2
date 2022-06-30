import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, Precision, Recall


def load_best_model(best_params: dict, model_class, feature_n, classes_n, device, cv=None):
    version = (
        f"layer={best_params['hidden_layer'][0]}-"
        f"lr={best_params['learning_rate'][0]}-"
        f"batchsize={best_params['batch_size'][0]}-"
        f"cv={cv}-"
        f"weights={best_params['weights'][0]}"
    )
    ckpt_path = f"logs/DL-Exam2/{version}/checkpoints/best-checkpoint.ckpt"
    trained_model = model_class.load_from_checkpoint(
        ckpt_path,
        feature_n=feature_n,
        classes_n=classes_n,
        learning_rate=best_params["learning_rate"][0],
        layer_count=best_params["hidden_layer"][0],
        device=device,
    )
    trained_model.eval()
    trained_model.freeze()
    return trained_model


class MultiClassifier(nn.Module):
    def __init__(self, feature_n, classes_n, layer_count: int = 1) -> None:
        super().__init__()

        layers = list()
        input_size = 2 ** (4 + layer_count)
        layers.append(nn.Linear(feature_n, input_size))
        layers.append(nn.LeakyReLU())
        for layer in reversed(range(layer_count)):
            output_size = 2 ** (4 + layer)
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.LeakyReLU())
            input_size = output_size
        layers.append(nn.Linear(output_size, classes_n))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        outputs = self.classifier(x)
        return F.log_softmax(outputs, dim=1)

    def predict(self, x):
        return self.forward(x).argmax(dim=1)

    def predict_proba(self, x):
        return torch.exp(self.forward(x))


class NeuralNet(pl.LightningModule):
    def __init__(
        self,
        feature_n,
        classes_n,
        learning_rate,
        layer_count,
        weights=None,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.model = MultiClassifier(feature_n, classes_n, layer_count)
        self.lossfunc = F.nll_loss
        self.lr = learning_rate
        self.weights = weights
        self.metrics = {
            "accuracy": Accuracy().to(device),
            "balanced_accuracy": Recall(num_classes=classes_n, average="macro").to(device),
            "f1": F1Score(num_classes=classes_n, average="macro").to(device),
            "precision": Precision(num_classes=classes_n, average="macro").to(device),
        }

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx: int):
        x, y = batch["data"], batch["labels"]
        y_hat = self(x)

        loss = self.lossfunc(y_hat, y, self.weights)
        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        for key, metric in self.metrics.items():
            y_pred = y_hat.argmax(dim=1)
            metric_res = metric(y_pred, y)
            self.log(f"train_{key}", metric_res, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        x, y = batch["data"], batch["labels"]
        y_hat = self(x)

        loss = self.lossfunc(y_hat, y, self.weights)
        self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        for key, metric in self.metrics.items():
            y_pred = y_hat.argmax(dim=1)
            metric_res = metric(y_pred, y)
            self.log(f"val_{key}", metric_res, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx: int):
        x, y = batch["data"], batch["labels"]
        y_hat = self(x)

        loss = self.lossfunc(y_hat, y, self.weights)
        self.log("test_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        for key, metric in self.metrics.items():
            y_pred = y_hat.argmax(dim=1)
            metric_res = metric(y_pred, y)
            self.log(f"test_{key}", metric_res, prog_bar=True, logger=True, sync_dist=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, y = batch["data"], batch["labels"]
        return {"y_pred": self(x).argmax(dim=1), "y_true": y}
