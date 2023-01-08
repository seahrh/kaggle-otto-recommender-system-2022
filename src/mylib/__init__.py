import os
from typing import Any, Dict, List, Optional, Union

import os
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import scml
import torch
from torch.utils.data import Dataset
from transformers import (
    BatchEncoding,
    EncoderDecoderConfig,
    EncoderDecoderModel,
    RobertaConfig,
)

__all__ = ["OttoDataset", "OttoLightningModel"]

log = scml.get_logger(__name__)

ParamType = Union[str, int, float, bool]


class Trainer(pl.Trainer):
    def save_checkpoint(
        self,
        filepath,
        weights_only: bool = False,
        storage_options: Optional[Any] = None,
    ) -> None:
        if self.is_global_zero:
            # model = self.lightning_module.model_to_save
            model = self.lightning_module.model
            white = ["weighted_layer_pooling", "log_vars"]
            for name, param in model.named_parameters():  # type: ignore
                for w in white:
                    if name.startswith(w):
                        log.info(f"{name}={param}")
            if isinstance(model, EncoderDecoderModel):
                dirpath = os.path.split(filepath)[0]
                model.save_pretrained(dirpath)  # type: ignore
                return
        super().save_checkpoint(filepath, weights_only, storage_options)


def training_callbacks(patience: int, monitor: str = "val_loss", verbose: bool = True):
    return [
        pl.callbacks.EarlyStopping(monitor=monitor, patience=patience, verbose=verbose),
        pl.callbacks.ModelCheckpoint(monitor=monitor, verbose=verbose, save_top_k=1),
        pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
    ]


class OttoDataset(Dataset):
    def __init__(
        self,
        encoding: Union[BatchEncoding, Dict],
        session_ids: Optional[List[int]] = None,
        labels: Optional[List[List[int]]] = None,
    ):
        self.overflow_to_sample_mapping = []
        if "overflow_to_sample_mapping" in encoding:
            self.overflow_to_sample_mapping = encoding.pop("overflow_to_sample_mapping")
        self.encoding = encoding
        if labels is not None:
            self.encoding["labels"] = labels
        self.session_ids: List[int] = []
        if session_ids is not None:
            self.session_ids = session_ids
            if len(self.overflow_to_sample_mapping) != 0:
                self.session_ids = [
                    session_ids[i] for i in self.overflow_to_sample_mapping
                ]

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encoding.items()}

    def __len__(self):
        return len(self.encoding["input_ids"])

    def seqlen(self) -> int:
        """Sequence length"""
        return len(self.encoding["input_ids"][0])

    def labels(self) -> List[int]:
        return list(self.encoding.get("labels", []))

    def stratification(self) -> List[int]:
        res = []
        if "labels" in self.encoding:
            # first aid at index 1
            res = [row[1] for row in self.encoding["labels"]]
        return res

    def groups(self) -> List[int]:
        return self.session_ids


# noinspection PyAbstractClass
class OttoLightningModel(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        decoder_start_token_id: int,
        pad_token_id: int,
        vocab_size: int,
        hidden_size: int,
    ):
        super().__init__()
        self.automatic_optimization = True
        self.lr = lr
        self.model = EncoderDecoderModel(
            config=EncoderDecoderConfig.from_encoder_decoder_configs(
                encoder_config=RobertaConfig(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    intermediate_size=1 * hidden_size,
                    num_attention_heads=4,
                    # max_position_embeddings=32, triggers cuda-side assert
                ),
                decoder_config=RobertaConfig(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    intermediate_size=1 * hidden_size,
                    num_attention_heads=4,
                    # max_position_embeddings=32, triggers cuda-side assert
                ),
            )
        )
        # config required for training
        self.model.config.decoder_start_token_id = decoder_start_token_id
        self.model.config.pad_token_id = pad_token_id
        self.model.config.vocab_size = vocab_size

    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        if not self.automatic_optimization:
            opts = self.optimizers()
            if not isinstance(opts, list):
                opts = [opts]
            for opt in opts:
                opt.zero_grad()
                self.manual_backward(loss)
                opt.step()
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def configure_optimizers(self):
        base = []
        weighted_layer_pooling = []
        log_vars = []
        for name, param in self.model.named_parameters():
            if name.startswith("weighted_layer_pooling"):
                weighted_layer_pooling.append(param)
                continue
            if name.startswith("log_vars"):
                log_vars.append(param)
                continue
            base.append(param)
        optimizers = [
            torch.optim.AdamW(
                [
                    {"params": base},
                    {"params": weighted_layer_pooling, "lr": 1e-3},
                    {"params": log_vars, "lr": 1e-3},
                ],
                lr=self.lr,
                amsgrad=False,
            )
        ]
        schedulers = []
        return optimizers, schedulers
