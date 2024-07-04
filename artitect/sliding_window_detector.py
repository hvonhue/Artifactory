from mask_detector import WindowTransformerDetector
from modeling import SinusoidalPositionEmbedding, WarmupLR, _convolutions, _linear
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch.nn import (
    Dropout,
    Linear,
    MaxPool1d,
    ReLU,
    Sigmoid,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch.nn.functional import adaptive_avg_pool2d, binary_cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryFBetaScore


class SlidingWindowTransformerDetector(LightningModule):
    """
    Use convolutional layers to extract features, then use them
    in a transformer block.

    Parameters
    ----------
    window : int
        The window size for the input sequence.
    convolution_features : list[int]
        The number of features for each convolutional layer.
    convolution_width : list[int]
        The kernel width for each convolutional layer.
    convolution_dropout : float
        The dropout rate for the convolutional layers.
    transformer_heads : int
        The number of heads in the transformer encoder.
    transformer_feedforward : int
        The dimension of the feedforward network model in the transformer.
    transformer_layers : int
        The number of sub-encoder-layers in the transformer encoder.
    transformer_dropout : float
        The dropout rate for the transformer encoder.
    conv_pool_kernel : int, optional
        The kernel size for the max pooling layer after convolutional layers (default is 4).
    pooling : str, optional
        The pooling method to use ('avg', 'max', 'cls', or 'total') (default is 'avg').
    loss : str, optional
        The key for the loss calculation from the batch dictionary (default is 'label').
    loss_boost_fp : float, optional
        The boosting factor for false positive loss (default is 0.5).
    act_fct : LightningModule, optional
        The activation function to use (default is Sigmoid()).
    warmup : int, optional
        The number of warmup steps for the learning rate scheduler (default is 30).
    """

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
        conv_pool_kernel: int = 4,
        pooling: str = "avg",
        loss: str = "label",
        loss_boost_fp: float = 0.5,
        act_fct: LightningModule = Sigmoid(),
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
        self.conv_pooling = MaxPool1d(conv_pool_kernel)
        # positional encoding for transformer
        self.position = SinusoidalPositionEmbedding(
            convolution_features[-1], int(window / conv_pool_kernel)
        )
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
        self.pooling = pooling
        self.linear = Linear(convolution_features[-1], 1)
        self.act_fct = act_fct
        self.warmup = warmup
        self.f1_score = BinaryF1Score(multidim_average="global")
        self.accuracy = BinaryAccuracy(multidim_average="global")
        self.fbeta_score = BinaryFBetaScore(beta=0.5)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch, window).

        Returns
        -------
        Tensor
            Output tensor of shape (batch, window).
        """
        x = x.unsqueeze(1)
        x = self.convolutions(x)
        x = self.conv_pooling(x)
        # transpose to use convolution features
        # as datapoint embeddings
        x = x.transpose(1, 2)
        # apply position and transformer block
        x = self.position(x)
        # apply dropout before transformer
        x = self.dropout(x)
        x = self.transformer(x)
        # different options of pooling over sequence length
        if self.pooling == "cls":
            x = x[:, 0, :]
        elif self.pooling == "max":
            x = x.max(1)[0]
        elif self.pooling == "avg":
            x = x.mean(1)
        elif self.pooling == "total":
            x = adaptive_avg_pool2d(x, (1, 1))
            x = x.squeeze()
            return self.act_fct(x)
        else:
            # another linear layer instead of pooling
            x = x.transpose(1, 2)
            x = Linear(int(self.window / self.conv_pooling), 1)(x)
            x = x.squeeze()
        # convert output tokens back to predictions
        x = self.linear(x)
        x = x.squeeze()
        # activation function for binary classification
        return self.act_fct(x)

    def training_step(self, batch, _):
        """
        Training step.

        Parameters
        ----------
        batch : dict
            Dictionary containing the input data and targets.
        _ : Any
            Placeholder for unused argument.

        Returns
        -------
        Tensor
            Loss value for the training step.
        """
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss].float()
        loss = binary_cross_entropy(y, m)
        if self.loss_boost_fp > 0 and self.loss_boost_fp <= 1:
            loss_fp = binary_cross_entropy(y[m == 0], m[m == 0])
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
                "train_fbeta": fbeta_score,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train", loss.item())
        return loss

    def validation_step(self, batch, _):
        """
        Validation step.

        Parameters
        ----------
        batch : dict
            Dictionary containing the input data and targets.
        _ : Any
            Placeholder for unused argument.

        Returns
        -------
        Tensor
            Loss value for the validation step.
        """
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss].float()
        loss = binary_cross_entropy(y, m)
        if self.loss_boost_fp > 0 and self.loss_boost_fp <= 1:
            loss_fp = binary_cross_entropy(y[m == 0], m[m == 0])
            loss += self.loss_boost_fp * loss_fp
            self.log("val_fp", loss_fp.item())
        fbeta_score = self.fbeta_score(y, m)
        self.log("validation", loss.item())
        self.log("val_fbeta", fbeta_score)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate scheduler.

        Returns
        -------
        dict
            Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = Adam(self.parameters(), lr=1e-2)
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
        """
        Log gradient norms before the optimizer step.

        Parameters
        ----------
        _ : Any
            Placeholder for unused argument.
        """
        self.log_dict(grad_norm(self, norm_type=2))


class FineTunedSlidingWindowDetector(LightningModule):
    """
    Fine-tuned Sliding Window Detector using a pretrained WindowTransformerDetector.

    Parameters
    ----------
    pretrainedModel : WindowTransformerDetector
        The pretrained model to fine-tune.
    pooling : str, optional
        The pooling method to use ('avg', 'max', 'cls', or 'linear') (default is 'avg').
    act_fct : callable, optional
        The activation function to use (default is Sigmoid()).
    warmup : int, optional
        The number of warmup steps for the learning rate scheduler (default is 10000).
    """

    def __init__(
        self,
        pretrainedModel: WindowTransformerDetector,
        pooling: str = "avg",
        act_fct=Sigmoid(),
        warmup: int = 10000,
    ):
        super().__init__()
        self.window = pretrainedModel.window
        self.loss = "label"
        self.loss_boost_fp = pretrainedModel.loss_boost_fp
        self.save_hyperparameters()
        # convolutional layers to extract features
        self.convolutions = pretrainedModel.convolutions
        # positional encoding for transformer
        self.position = pretrainedModel.position
        self.dropout = pretrainedModel.dropout
        # transformer block
        self.transformer = pretrainedModel.transformer
        # final linear block to convert each
        # feature to a single value
        self.pooling = pooling
        self.linear = Linear(pretrainedModel.convolutions[-2].out_channels, 1)
        self.act_fct = act_fct
        self.warmup = warmup
        self.accuracy = BinaryAccuracy()
        self.fbeta_score = BinaryFBetaScore(beta=0.5)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch, window).

        Returns
        -------
        Tensor
            Output tensor of shape (batch, window).
        """
        x = x.unsqueeze(1)
        x = self.convolutions(x)
        # transpose to use convolution features
        # as datapoint embeddings
        x = x.transpose(1, 2)
        # apply position and transformer block
        x = self.position(x)
        # apply dropout before transformer
        x = self.dropout(x)
        x = self.transformer(x)
        # different options of pooling
        if self.pooling == "cls":
            x = x[:, 0, :]
        elif self.pooling == "max":
            x = x.max(1)[0]
        elif self.pooling == "avg":
            x = x.mean(1)
        else:
            # another linear layer instead of pooling
            x = x.transpose(1, 2)
            x = Linear(self.window, 1)(x)
            x = x.squeeze()
        # convert output tokens back to predictions
        x = self.linear(x)
        x = x.squeeze()
        # activation function for binary classification
        x = self.act_fct(x)
        return x

    def training_step(self, batch, _):
        """
        Training step.

        Parameters
        ----------
        batch : dict
            Dictionary containing the input data and targets.
        _ : Any
            Placeholder for unused argument.

        Returns
        -------
        Tensor
            Loss value for the training step.
        """
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss].float()
        loss = binary_cross_entropy(y, m)
        if self.loss_boost_fp > 0 and self.loss_boost_fp <= 1:
            loss_fp = binary_cross_entropy(y[m == 0], m[m == 0])
            loss += self.loss_boost_fp * loss_fp
            self.log("train_fp", loss_fp.item())
        accuracy = self.accuracy(y, m)
        fbeta_score = self.fbeta_score(y, m)
        self.log_dict(
            {
                "train_loss": loss.item(),
                "train_accuracy": accuracy,
                "train_fbeta_score": fbeta_score,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, _):
        """
        Validation step.

        Parameters
        ----------
        batch : dict
            Dictionary containing the input data and targets.
        _ : Any
            Placeholder for unused argument.

        Returns
        -------
        Tensor
            Loss value for the validation step.
        """
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss].float()
        loss = binary_cross_entropy(y, m)
        loss_fp = binary_cross_entropy(y[m == 0], m[m == 0])
        fbeta_score = self.fbeta_score(y, m)
        self.log("validation", loss.item())
        self.log("validation_fp", loss_fp.item())
        self.log("val_fbeta", fbeta_score)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate scheduler.

        Returns
        -------
        dict
            Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = Adam(self.parameters(), lr=1e-3)
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
        """
        Log gradient norms before the optimizer step.

        Parameters
        ----------
        _ : Any
            Placeholder for unused argument.
        """
        self.log_dict(grad_norm(self, norm_type=2))



class SlidingWindowLinearDetector(LightningModule):
    """
    Use convolutional layers to extract features, then use them in a dense block.

    Parameters
    ----------
    window : int
        The size of the sliding window.
    convolution_features : list[int]
        List of the number of features for each convolutional layer.
    convolution_width : list[int]
        List of kernel sizes for each convolutional layer.
    convolution_dropout : float
        Dropout rate for the convolutional layers.
    pooling_kernel : int, optional
        Kernel size for the max pooling layer (default is 8).
    linear_dropout : float, optional
        Dropout rate for the linear layers (default is 0).
    linear_layers : list[int], optional
        List of sizes for the linear layers (default is None).
    loss : str, optional
        Loss function to use (default is "label").
    loss_boost_fp : float, optional
        Boost factor for false positives in the loss function (default is 0.2).
    warmup : int, optional
        Number of warmup steps for the learning rate scheduler (default is 15000).
    act_fct : LightningModule, optional
        Activation function to use in the output layer (default is Sigmoid()).
    """

    def __init__(
        self,
        window: int,
        convolution_features: list[int],
        convolution_width: list[int],
        convolution_dropout: float,
        pooling_kernel: int = 8,
        linear_dropout: float = 0,
        linear_layers: list[int] | None = None,
        loss: str = "label",
        loss_boost_fp: float = 0.2,
        warmup: int = 15000,
        act_fct: LightningModule = Sigmoid(),
    ):
        super().__init__()
        self.window = window
        self.loss = loss
        self.loss_boost_fp = loss_boost_fp
        self.save_hyperparameters()
        self.convolution_features = convolution_features
        # convolutional layers to extract features
        self.convolutions = _convolutions(
            convolution_features=[1] + convolution_features,
            convolution_width=convolution_width,
            convolution_dropout=convolution_dropout,
            pad=True,
        )
        # dropout layer before dense block
        self.dropout = Dropout(linear_dropout)
        self.pooling = MaxPool1d(kernel_size=pooling_kernel)
        # final linear block to convert each
        # feature to a single value
        self.linear = _linear(
            [
                int(window / pooling_kernel) * convolution_features[-1],
                *(linear_layers or list()),
                1,
            ],
            activation=ReLU(),
            last=False,
        )
        self.warmup = warmup
        self.act_fct = act_fct
        self.f1_score = BinaryF1Score(multidim_average="global")
        self.accuracy = BinaryAccuracy(multidim_average="global")
        self.fbeta_score = BinaryFBetaScore(beta=0.5)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch, window).

        Returns
        -------
        Tensor
            Output tensor of shape (batch,).
        """
        x = x.unsqueeze(1)
        x = self.convolutions(x)
        # Max 1D Pooling
        x = self.pooling(x)
        # flatten
        x = x.view(x.shape[0], -1)
        # dense block
        x = self.linear(x)
        x = x.squeeze()
        return self.act_fct(x)

    def training_step(self, batch, _):
        """
        Training step.

        Parameters
        ----------
        batch : dict
            Dictionary containing the input data and targets.
        _ : Any
            Placeholder for unused argument.

        Returns
        -------
        Tensor
            Loss value for the training step.
        """
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss].float()
        loss = binary_cross_entropy(y, m)
        if self.loss_boost_fp > 0 and self.loss_boost_fp <= 1:
            loss_fp = binary_cross_entropy(y[m == 0], m[m == 0])
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
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, _):
        """
        Validation step.

        Parameters
        ----------
        batch : dict
            Dictionary containing the input data and targets.
        _ : Any
            Placeholder for unused argument.

        Returns
        -------
        Tensor
            Loss value for the validation step.
        """
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss].float()
        loss = binary_cross_entropy(y, m)
        if self.loss_boost_fp > 0 and self.loss_boost_fp <= 1:
            loss_fp = binary_cross_entropy(y[m == 0], m[m == 0])
            loss += self.loss_boost_fp * loss_fp
            self.log("validation_fp", loss_fp.item())
        fbeta_score = self.fbeta_score(y, m)
        self.log("validation", loss.item())
        self.log("val_fbeta", fbeta_score)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate scheduler.

        Returns
        -------
        dict
            Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = Adam(self.parameters(), lr=1e-2)
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


class ConvolutionalSlidingWindowDetector(LightningModule):
    """
    Use convolutional layers to extract features, then use them in a dense block.

    Parameters
    ----------
    window : int
        The size of the sliding window.
    convolution_features : list[int]
        List of the number of features for each convolutional layer.
    convolution_width : list[int]
        List of kernel sizes for each convolutional layer.
    convolution_dropout : float
        Dropout rate for the convolutional layers.
    pooling : str, optional
        Pooling method to use. Options are "avg", "max", "cls", or "none". (default is "avg")
    loss : str, optional
        Loss function to use (default is "label").
    loss_boost_fp : float, optional
        Boost factor for false positives in the loss function (default is 0.2).
    batch_normalization : bool, optional
        Whether to use batch normalization in the convolutional layers (default is False).
    act_fct : LightningModule, optional
        Activation function to use in the output layer (default is Sigmoid()).
    warmup : int, optional
        Number of warmup steps for the learning rate scheduler (default is 15000).
    """

    def __init__(
        self,
        window: int,
        convolution_features: list[int],
        convolution_width: list[int],
        convolution_dropout: float,
        pooling: str = "avg",
        loss: str = "label",
        loss_boost_fp: float = 0.2,
        batch_normalization: bool = False,
        act_fct: LightningModule = Sigmoid(),
        warmup: int = 15000,
    ):
        super().__init__()
        self.window = window
        self.loss = loss
        self.loss_boost_fp = loss_boost_fp
        self.save_hyperparameters()
        self.convolution_features = convolution_features
        # convolutional layers to extract features
        self.convolutions = _convolutions(
            convolution_features=[1] + convolution_features,
            convolution_width=convolution_width,
            convolution_dropout=convolution_dropout,
            batch_normalization=batch_normalization,
            activation="relu",
            pad=True,
        )
        # dropout layer before dense block
        self.pooling = pooling
        # final linear block to convert each
        # feature to a single value
        self.act_fct = act_fct
        self.warmup = warmup
        self.f1_score = BinaryF1Score(multidim_average="global")
        self.accuracy = BinaryAccuracy(multidim_average="global")
        self.fbeta_score = BinaryFBetaScore(beta=0.5)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch, window).

        Returns
        -------
        Tensor
            Output tensor of shape (batch,).
        """
        x = x.unsqueeze(1)
        x = self.convolutions(x)
        # Pooling
        if self.pooling == "cls":
            x = x[:, 0, :]
        elif self.pooling == "max":
            x = x.max(1)[0]
        # Global Average Pooling
        elif self.pooling == "avg":
            x = adaptive_avg_pool2d(x, (1, 1))
        else:
            # another linear layer instead of pooling
            x = x.transpose(1, 2)
            x = Linear(self.window, 1)(x)
            x = x.squeeze()
        x = x.squeeze()
        return self.act_fct(x)

    def training_step(self, batch, _):
        """
        Training step.

        Parameters
        ----------
        batch : dict
            Dictionary containing the input data and targets.
        _ : Any
            Placeholder for unused argument.

        Returns
        -------
        Tensor
            Loss value for the training step.
        """
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss].float()
        # loss = macro_soft_f1(preds=y, labels=m, beta=0.5)
        loss = binary_cross_entropy(y, m)
        if self.loss_boost_fp > 0 and self.loss_boost_fp <= 1:
            loss_fp = binary_cross_entropy(y[m == 0], m[m == 0])
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
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train", loss.item())
        return loss

    def validation_step(self, batch, _):
        """
        Validation step.

        Parameters
        ----------
        batch : dict
            Dictionary containing the input data and targets.
        _ : Any
            Placeholder for unused argument.

        Returns
        -------
        Tensor
            Loss value for the validation step.
        """
        y = self.forward(batch["data"] + batch["artifact"])
        m = batch[self.loss].float()
        # loss = macro_soft_f1(preds=y, labels=m, beta=0.5)
        loss = binary_cross_entropy(y, m)
        if self.loss_boost_fp > 0 and self.loss_boost_fp <= 1:
            loss_fp = binary_cross_entropy(y[m == 0], m[m == 0])
            loss += self.loss_boost_fp * loss_fp
            self.log("val_fp", loss_fp.item())
        fbeta_score = self.fbeta_score(y, m)
        self.log("validation", loss.item())
        self.log("val_fbeta", fbeta_score)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate scheduler.

        Returns
        -------
        dict
            Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = Adam(self.parameters(), lr=1e-2)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=3,
            min_lr=1e-5,
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
        """
        Log gradient norms before the optimizer step.

        Parameters
        ----------
        _ : Any
            Placeholder for unused argument.
        """
        self.log_dict(grad_norm(self, norm_type=2))
