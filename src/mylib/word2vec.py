from typing import Optional, List
import pytorch_lightning as pl
import scml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

__all__ = ["SkipGramDataset", "SkipGramWord2Vec"]
log = scml.get_logger(__name__)


class SkipGramDataset(Dataset):
    def __init__(
        self,
        center_words: List[int],
        center_types: List[int],
        outside_words: List[int],
        outside_types: List[int],
    ):
        self.encoding = {
            "center_words": center_words,
            "center_types": center_types,
            "outside_words": outside_words,
            "outside_types": outside_types,
        }

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encoding.items()}

    def __len__(self):
        return len(self.encoding["center_words"])

    def labels(self) -> List[int]:
        return list(self.encoding.get("outside_words", []))


# noinspection PyAbstractClass
class SkipGramWord2Vec(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        vocab_size: int,
        types_size: int,
        embedding_size: int,
        noise_dist: Optional[List[float]] = None,
        negative_samples: int = 10,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.automatic_optimization = True
        self.lr = lr
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.type_embeddings = nn.Embedding(types_size, embedding_size)
        self.vocab_size = vocab_size
        self.types_size = types_size
        self.negative_samples = negative_samples
        self.noise_dist = (
            noise_dist if noise_dist is not None else torch.ones(self.vocab_size)
        )
        # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
        # initializer_range(`float`, *optional *, defaults to 0.02):
        # The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        # self.word_embeddings.weight.data.normal_(mean=0.0, std=initializer_range)
        # self.type_embeddings.weight.data.normal_(mean=0.0, std=initializer_range)
        # Initialize both embedding tables with uniform distribution
        self.word_embeddings.weight.data.uniform_(-1, 1)
        self.type_embeddings.weight.data.uniform_(-1, 1)

    def forward(self, center_words, center_types, outside_words, outside_types):
        log.debug(
            f"center_word.size={center_words.size()}, outside_word.size={outside_words.size()}"
        )  # bs
        # bs, emb_dim
        em_center = self.word_embeddings(center_words) + self.type_embeddings(
            center_types
        )
        # bs, emb_dim
        em_outside = self.word_embeddings(outside_words) + self.type_embeddings(
            outside_types
        )
        log.debug(
            f"em_center.size={em_center.size()}, em_outside.size={em_outside.size()}"
        )
        # dot product: element-wise multiply, followed by sum
        em_dot = torch.mul(em_center, em_outside)  # bs, emb_dim
        em_dot = torch.sum(em_dot, dim=1)  # bs
        log.debug(f"em_dot.size={em_dot.size()}")
        true_pair_loss = F.logsigmoid(em_dot).neg()  # bs
        log.debug(f"true_pair_loss.size={true_pair_loss.size()}")
        loss = true_pair_loss
        if self.negative_samples > 0:
            num_samples = outside_words.size()[0] * self.negative_samples
            neg_input_ids = torch.multinomial(
                self.noise_dist,
                num_samples=num_samples,
                replacement=num_samples > self.vocab_size,
            )
            # bs, num_neg_samples
            # need to set device explicitly here, else error:
            # Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu
            neg_input_ids = neg_input_ids.view(
                outside_words.size()[0], self.negative_samples
            ).to(self.device)
            log.debug(f"neg_input_ids.size={neg_input_ids}")
            # bs, neg_samples, emb_dim
            em_neg = self.word_embeddings(neg_input_ids) + self.type_embeddings(
                torch.zeros(size=neg_input_ids.size(), dtype=torch.int).to(
                    neg_input_ids
                )
            )
            log.debug(f"em_neg.size={em_neg.size()}")
            # batch matrix multiply
            # (B, K, D) * (B, D, 1) = (B, K, 1)
            # Negated dot product of noise pair
            # Large +dot, large -dot, sigmoid 0, logsigmoid -Inf
            # Large -dot, large +dot, sigmoid 1, logsigmoid zero
            em_dot_neg = torch.bmm(em_neg, em_center.unsqueeze(2)).neg()
            em_dot_neg = em_dot_neg.squeeze(2)
            log.debug(f"em_dot_neg.size={em_dot_neg.size()}")
            noise_pair_loss = F.logsigmoid(em_dot_neg).sum(1).neg()  # bs
            log.debug(f"noise_pair_loss.size={noise_pair_loss.size()}")
            loss += noise_pair_loss
        return loss.mean()

    def training_step(self, batch, batch_idx):
        loss = self(**batch)
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
        loss = self(**batch)
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
        for name, param in self.named_parameters():
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
