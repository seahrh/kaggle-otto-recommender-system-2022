from typing import List, Optional
import pytorch_lightning as pl
import scml
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SkipGramWord2Vec", "Word2VecLightningModel"]
log = scml.get_logger(__name__)


class SkipGramWord2Vec(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        vocab_size: int,
        noise_dist: Optional[List[float]] = None,
        negative_samples: int = 10,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        self.noise_dist = noise_dist
        # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
        # initializer_range(`float`, *optional *, defaults to 0.02):
        # The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        self.embeddings.weight.data.normal_(mean=0.0, std=initializer_range)

    def forward(self, center_words, outside_words):
        log.debug(
            f"center_word.size={center_words.size()}, outside_word.size={outside_words.size()}"
        )  # bs
        em_center = self.embeddings(center_words)  # bs, emb_dim
        em_outside = self.embeddings_context(outside_words)  # bs, emb_dim
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
            noise_dist = self.noise_dist
            if self.noise_dist is None:
                noise_dist = torch.ones(self.vocab_size)
            num_samples = outside_words.size()[0] * self.negative_samples
            neg_input_ids = torch.multinomial(
                noise_dist,
                num_samples=num_samples,
                replacement=num_samples > self.vocab_size,
            )
            # bs, num_neg_samples
            neg_input_ids = neg_input_ids.view(
                outside_words.shape[0], self.negative_samples
            )
            log.debug(f"neg_input_ids.size={neg_input_ids}")
            # bs, neg_samples, emb_dim
            em_neg = self.embeddings(neg_input_ids)
            log.debug(f"em_neg.size={em_neg.size()}")
            # batch matrix multiply
            # (B, K, D) * (B, D, 1) = (B, K, 1)
            # Negation of outside word vectors
            em_dot_neg = torch.bmm(em_neg.neg(), em_center.unsqueeze(2))
            em_dot_neg = em_dot_neg.squeeze(2)
            log.debug(f"em_dot_neg.size={em_dot_neg.size()}")
            noise_pair_loss = F.logsigmoid(em_dot_neg).sum(1).neg()  # bs
            log.debug(f"noise_pair_loss.size={noise_pair_loss.size()}")
            loss += noise_pair_loss
        return loss.mean()


# noinspection PyAbstractClass
class Word2VecLightningModel(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        vocab_size: int,
        embedding_size: int,
        noise_dist: Optional[List[float]],
        negative_samples: int,
    ):
        super().__init__()
        self.automatic_optimization = True
        self.lr = lr
        self.model = SkipGramWord2Vec(
            embedding_size=embedding_size,
            vocab_size=vocab_size,
            noise_dist=noise_dist,
            negative_samples=negative_samples,
        )

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
