import gc
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import scml
import sklearn
import torch
from scml import torchx

from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    BatchEncoding,
    RobertaConfig,
    BertConfig,
    EncoderDecoderConfig,
    EncoderDecoderModel,
)

__all__ = ["OttoDataset", "OttoLightningModel", "OttoObjective"]

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
        labels: Optional[List[List[int]]] = None,
        session_ids: Optional[List[int]] = None,
    ):

        self.encoding = encoding
        if labels is not None:
            self.encoding["labels"] = labels
        self.session_ids = session_ids

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encoding.items()}

    def __len__(self):
        return len(self.encoding.input_ids)

    def seqlen(self) -> int:
        """Sequence length"""
        return len(self.encoding["input_ids"][0])

    def labels(self) -> List[int]:
        return self.encoding.get("labels", [])

    def stratification(self) -> List[int]:
        # first aid at index 1
        return [x[1] for x in self.encoding["labels"]]


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
                    intermediate_size=4 * hidden_size,
                    num_attention_heads=8,
                    # max_position_embeddings=32,
                ),
                decoder_config=RobertaConfig(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    intermediate_size=4 * hidden_size,
                    num_attention_heads=8,
                    # max_position_embeddings=32,
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


class OttoObjective:
    def __init__(
        self,
        ds: OttoDataset,
        n_splits: int,
        epochs: int,
        batch_size: int,
        patience: int,
        job_ts: str,
        job_dir: Path,
        decoder_start_token_id: int,
        pad_token_id: int,
        vocab_size: int,
        lr: Tuple[float, float],
        gpus: List[int],
    ):
        self.ds = ds
        self.splitter = StratifiedGroupKFold(n_splits=n_splits)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.job_ts = job_ts
        self.job_dir = job_dir
        self.lr = lr
        self.decoder_start_token_id = decoder_start_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.gpus = gpus
        self.history: List[Dict[str, ParamType]] = []

    def __call__(self, trial):
        hist = {
            "time": self.job_ts,
            "model_name": "",
            "remarks": "",
            "trial_id": trial.number,
            "lr": trial.suggest_loguniform("lr", self.lr[0], self.lr[1]),
        }
        scores: List[float] = []
        epochs_list: List[int] = []
        dummy = np.zeros(len(self.ds))
        y = np.array(self.ds.labels(), dtype=np.uint8)
        for fold, (ti, vi) in enumerate(
            self.splitter.split(
                dummy,
                y=self.ds.stratification(),
                groups=self.ds.session_ids,
            )
        ):
            gc.collect()
            torch.cuda.empty_cache()
            directory = self.job_dir / f"trial{trial.number:02d}" / f"fold{fold:02d}"
            directory.mkdir(parents=True, exist_ok=True)
            tra_ds = torch.utils.data.Subset(self.ds, ti)
            val_ds = torch.utils.data.Subset(self.ds, vi)
            model = OttoLightningModel(
                lr=hist["lr"],
                decoder_start_token_id=self.decoder_start_token_id,
                pad_token_id=self.pad_token_id,
                vocab_size=self.vocab_size,
                hidden_size=64,
            )
            trainer = Trainer(
                default_root_dir=str(directory),
                gpus=self.gpus,
                max_epochs=self.epochs,
                callbacks=training_callbacks(patience=self.patience),
                deterministic=False,
            )
            trainer.fit(
                model,
                train_dataloaders=DataLoader(
                    tra_ds,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0,
                ),
                val_dataloaders=DataLoader(
                    val_ds,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                ),
            )
            epochs = trainer.current_epoch
            pass
            del model, trainer, tra_ds, val_ds
        log.debug("all folds completed")
        pass
        self.history.append(hist)
        return hist["score_ef_worst"]
