from modeling import SinusoidalPositionEmbedding, WarmupLR, _convolutions, _linear
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch.nn import Dropout, Linear, TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryFBetaScore


class ConvolutionDetector(LightningModule):
    """Use only convolutional layers. Based on FCN"""

    def __init__(
        self,
        window: int,
        convolution_features: list[int],
        convolution_width: list[int],
        convolution_dropout: float = 0.0,
        pooling: str = "avg",
        loss: str = "mask",
        loss_boost_fp: float = 0.2,
        batch_normalization: bool = False,
        warmup: int = 30,
    ):
        super().__init__()
        self.window = window
        self.loss = loss
        self.loss_boost_fp = loss_boost_fp
        self.save_hyperparameters()
        self.convolutions = _convolutions(
            convolution_features=[1] + convolution_features,
            convolution_width=convolution_width,
            convolution_dropout=convolution_dropout,
            batch_normalization=batch_normalization,
            activation="relu",
            pad=True,
        )
        self.pooling = pooling
        # final linear block to convert each
        # feature to a single value
        self.warmup = warmup
        self.f1_score = BinaryF1Score(multidim_average="global")
        self.accuracy = BinaryAccuracy(multidim_average="global")
        self.fbeta_score = BinaryFBetaScore(beta=0.5)

    def forward(self, x):
        """
        :param x: input of size (batch, window)
        :return: output of size (batch, window)
        """
        x = x.unsqueeze(1)
        x = self.convolutions(x)
        if self.pooling == "max":
            x = x.max(1)[0]
        else:
            x = x.mean(1)
        return x.squeeze()

    def training_step(self, batch, _):
        """
        """
        # Your normal training operations
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss]
        loss = mse_loss(y, m)
        if self.loss_boost_fp > 0 and self.loss_boost_fp <= 1:
            loss_fp = mse_loss(y[m == 0], m[m == 0])
            loss += self.loss_boost_fp * loss_fp
            self.log("train_fp", loss_fp.item())
        accuracy = self.accuracy(y, m)
        f1_score = self.f1_score(y, m)
        fbeta_score = self.fbeta_score(y, m)

        self.log_dict(
            {
                "train_loss": loss.item(),
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
                "train_fbeta_score": fbeta_score,
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, _):
        """
        validation step
        """
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss]
        loss = mse_loss(y, m)
        if self.loss_boost_fp > 0 and self.loss_boost_fp <= 1:
            loss_fp = mse_loss(y[m == 0], m[m == 0])
            loss += self.loss_boost_fp * loss_fp
        fbeta_score = self.fbeta_score(y, m)
        self.log("validation", loss.item())
        self.log("validation_fp", loss_fp.item())
        self.log("val_fbeta", fbeta_score)

        return loss

    def configure_optimizers(self):
        """
        :return: dict
        """
        optimizer = Adam(self.parameters(), lr=1e-2)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=3,
            min_lr=1e-6,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 500,
                "name": "learning_rate",
                "strict": True,
                "monitor": "validation",
            },
        }
    
    def on_before_optimizer_step(self, _):
        self.log_dict(grad_norm(self, norm_type=2))


class WindowTransformerDetector(LightningModule):
    """Use convolutional layers to extract features, then use them
    in a transformer block"""

    def __init__(
        self,
        window: int,
        convolution_features: list[int],
        convolution_width: list[int],
        convolution_dropout: float,
        transformer_heads: int,
        transformer_feedforward: int,
        transformer_layers: int,
        transformer_dropout: float,
        loss: str = "mask",
        loss_boost_fp: float = 0.2,
        warmup: int = 30,
    ):
        super().__init__()
        self.window = window
        self.loss = loss
        self.loss_boost_fp = loss_boost_fp
        self.save_hyperparameters()
        # convolutional layers to extract features
        self.convolutions = _convolutions(
            convolution_features=[1] + convolution_features,
            convolution_width=convolution_width,
            convolution_dropout=convolution_dropout,
            pad=True,
        )
        # positional encoding for transformer
        self.position = SinusoidalPositionEmbedding(convolution_features[-1], window)
        self.dropout = Dropout(transformer_dropout)
        # transformer block
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                convolution_features[-1],
                transformer_heads,
                dim_feedforward=transformer_feedforward,
                batch_first=True,
            ),
            num_layers=transformer_layers,
        )
        # final linear block to convert each
        # feature to a single value
        self.linear = Linear(convolution_features[-1], 1)
        self.warmup = warmup
        self.f1_score = BinaryF1Score()
        self.accuracy = BinaryAccuracy()
        self.fbeta_score = BinaryFBetaScore(beta=0.5)

    def forward(self, x):
        """
        :param x: input of size (batch, window)
        :return: output of size (batch, window)
        """
        x = x.unsqueeze(1)
        x = self.convolutions(x)
        x = self.dropout(x)
        # transpose to use convolution features
        # as datapoint embeddings
        x = x.transpose(1, 2)
        # apply position and transformer block
        x = self.position(x)
        # apply dropout before transformer
        x = self.dropout(x)
        x = self.transformer(x)
        # convert output tokens back to predictions
        x = self.linear(x)

        return x.squeeze()

    def training_step(self, batch, _):
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss]
        loss = mse_loss(y, m)
        if self.loss_boost_fp > 0 and self.loss_boost_fp <= 1:
            loss_fp = mse_loss(y[m == 0], m[m == 0])
            loss += self.loss_boost_fp * loss_fp
            self.log("train_fp", loss_fp.item())
        accuracy = self.accuracy(y, m)
        f1_score = self.f1_score(y, m)
        fbeta_score = self.fbeta_score(y, m)
        self.log_dict(
            {
                "train_loss": loss.item(),
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
                "train_fbeta_score": fbeta_score,
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log("train", loss.item())

        return loss

    def validation_step(self, batch, _):
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss]
        loss = mse_loss(y, m)
        if self.loss_boost_fp > 0 and self.loss_boost_fp <= 1:
            loss_fp = mse_loss(y[m == 0], m[m == 0])
            loss += self.loss_boost_fp * loss_fp
            self.log("validation_fp", loss_fp.item())
        fbeta_score = self.fbeta_score(y, m)
        self.log("validation", loss.item())
        self.log("val_fbeta", fbeta_score)
        
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-1)
        scheduler = WarmupLR(
            optimizer,
            warmup_steps=self.warmup,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 500,
                "name": "learning_rate",
                "strict": True,
                "monitor": "val_fbeta",
            },
        }

    def on_before_optimizer_step(self, _):
        self.log_dict(grad_norm(self, norm_type=2))


class WindowLinearDetector(LightningModule):
    """Use convolutional layers to extract features, then use them
    in a dense block"""

    def __init__(
        self,
        window: int,
        convolution_features: list[int],
        convolution_width: list[int],
        convolution_dropout: float,
        linear_dropout: float = 0,
        linear_layers: list[int] | None = None,
        loss: str = "mask",
        loss_boost_fp: float = 0.2,
        warmup: int = 30,
    ):
        super().__init__()
        self.window = window
        self.loss = loss
        self.loss_boost_fp = loss_boost_fp
        self.save_hyperparameters()
        # convolutional layers to extract features
        self.convolutions = _convolutions(
            convolution_features=[1] + convolution_features,
            convolution_width=convolution_width,
            convolution_dropout=convolution_dropout,
            pad=True,
        )
        # dropout layer before dense block
        self.dropout = Dropout(linear_dropout)
        # final linear block to convert each
        # feature to a single value
        self.linear = _linear(
            [convolution_features[-1] * window, *(linear_layers or list()), window]
        )
        self.warmup = warmup
        self.f1_score = BinaryF1Score()
        self.accuracy = BinaryAccuracy()
        self.fbeta_score = BinaryFBetaScore(beta=0.5)

    def forward(self, x):
        """
        :param x: input of size (batch, window)
        :return: output of size (batch, window)
        """
        x = x.unsqueeze(1)
        x = self.convolutions(x)
        x = self.dropout(x.view(x.shape[0], -1))
        x = self.linear(x)
        # convert output tokens back to predictions
        return x.squeeze()

    def training_step(self, batch, _):
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss]
        loss = mse_loss(y, m)
        if self.loss_boost_fp > 0 and self.loss_boost_fp <= 1:
            loss_fp = mse_loss(y[m == 0], m[m == 0])
            loss += self.loss_boost_fp * loss_fp
            self.log("train_fp", loss_fp.item())
        accuracy = self.accuracy(y, m)
        f1_score = self.f1_score(y, m)
        fbeta_score = self.fbeta_score(y, m)
        self.log_dict(
            {
                "train_loss": loss.item(),
                "train_accuracy": accuracy,
                "train_f1_score": f1_score,
                "train_fbeta_score": fbeta_score,
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        self.log("train", loss.item())
        return loss

    def validation_step(self, batch, _):
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss]
        loss = mse_loss(y, m)
        if self.loss_boost_fp > 0 and self.loss_boost_fp <= 1:
            loss_fp = mse_loss(y[m == 0], m[m == 0])
            loss += self.loss_boost_fp * loss_fp
            self.log("val_fp", loss_fp.item())
        self.accuracy(y, m)
        self.f1_score(y, m)
        fbeta_score = self.fbeta_score(y, m)
        self.log("validation", loss.item())
        self.log("val_fbeta", fbeta_score)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-2)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            min_lr=1e-6,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 500,
                "name": "learning_rate",
                "strict": True,
                "monitor": "val_fbeta",
            },
        }

    def on_before_optimizer_step(self, _):
        self.log_dict(grad_norm(self, norm_type=2))
