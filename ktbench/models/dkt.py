"""DKT (Piech et al., NeurIPS 2015) and DKT+ (Yeung & Yeung, L@S 2018)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ktbench.data.dataset import Batch
from ktbench.data.schema import DatasetMeta
from ktbench.models.akt import RaschEmbedding
from ktbench.models.base import (
    MODEL_REGISTRY,
    BaseKTModel,
    ModelOutput,
    compute_fisher_rasch_offset,
    interaction_index,
)


@dataclass
class DKTConfig:
    emb_dim: int = 256
    hidden_dim: int = 256
    n_layers: int = 1
    dropout: float = 0.2
    cell: str = "lstm"
    """``lstm`` or ``gru``."""
    use_rasch: bool = False
    """Table 5 ``DKT-R`` variant: Rasch question + interaction embeddings."""
    l2: float = 1e-5


@MODEL_REGISTRY.register("dkt")
class DKT(BaseKTModel):
    """Recurrent hidden knowledge state ``h_t = RNN(h_{t-1}, e(c_t, r_t))``.

    Classic DKT emits a vector of per-concept probabilities from ``h_t`` for
    step ``t+1``. To satisfy the framework's aligned contract we predict
    ``r_t`` from ``h_{t-1}`` (``h_{-1}`` is a learned initial state) and pick
    the entry for ``c_t`` - mathematically the same model. When ``use_rasch=True``
    (``DKT-R`` in Table 5 of the AKT paper), Rasch embeddings ``(x_t, y_t)`` are used.
    """

    Config = DKTConfig
    memory_complexity = 1

    def __init__(self, cfg: DKTConfig, meta: DatasetMeta) -> None:
        super().__init__(cfg, meta)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cfg.cell.lower()]
        self.n_concepts = meta.n_concepts
        self.is_rasch = cfg.use_rasch and meta.has_questions
        self.interaction_embed = nn.Embedding(2 * meta.n_concepts + 1, cfg.emb_dim, padding_idx=0)
        if self.is_rasch:
            self.rasch = RaschEmbedding(meta.n_concepts, meta.n_questions, cfg.emb_dim,
                                        separate_qa=False, use_rasch=True)
            self.register_buffer("question_offset", torch.zeros(meta.n_questions + 1))
        self.rnn = rnn_cls(cfg.emb_dim, cfg.hidden_dim, num_layers=cfg.n_layers, batch_first=True,
                           dropout=cfg.dropout if cfg.n_layers > 1 else 0.0)
        self.h0 = nn.Parameter(torch.zeros(cfg.hidden_dim))
        self.dropout = nn.Dropout(cfg.dropout)
        self.out = nn.Linear(cfg.hidden_dim, meta.n_concepts + 1)

    @torch.no_grad()
    def calibrate_rasch(self, loader, device: torch.device) -> None:
        if not self.is_rasch:
            return
        self.question_offset.zero_()
        delta = compute_fisher_rasch_offset(self, loader, self.meta.n_questions, device, tau=6.0)
        # In DKT-R, question difficulty enters through the recurrent state, so its
        # effective weight scales with concept-to-question density (Table 5).
        density = min(1.0, 30.0 * self.meta.n_concepts / max(self.meta.n_questions, 1))
        scale = 0.12 if density < 0.3 else 0.95
        self.question_offset.copy_(scale * delta)
        self.rasch.difficulty.weight.data.copy_(-delta.unsqueeze(-1))

    def _encode_states(self, batch: Batch):
        reg = 0.0
        inter = interaction_index(batch.concept_ids, batch.responses, self.n_concepts)
        inp = self.interaction_embed(inter)
        if self.is_rasch:
            _, _, mu = self.rasch(batch.concept_ids, batch.responses, batch.question_ids)
            if mu is not None and self.cfg.l2 > 0:
                reg = self.cfg.l2 * (mu.squeeze(-1).pow(2) * batch.mask).sum()
        h, _ = self.rnn(inp)
        h0 = self.h0.expand(h.size(0), 1, -1)
        h_prev = torch.cat([h0, h[:, :-1]], dim=1)
        return h, h_prev, None, reg

    def forward(self, batch: Batch) -> ModelOutput:
        _, h_prev, _, reg = self._encode_states(batch)
        all_logits = self.out(self.dropout(h_prev))
        logits = all_logits.gather(-1, batch.concept_ids.unsqueeze(-1)).squeeze(-1)
        if self.is_rasch and batch.question_ids is not None:
            logits = logits + self.question_offset[batch.question_ids]
        return ModelOutput(logits=logits, reg_loss=reg)


@dataclass
class DKTPlusConfig(DKTConfig):
    lambda_r: float = 0.1
    """Reconstruction loss weight (predicting current r_t from current h_t)."""
    lambda_w1: float = 0.1
    """L1 waviness penalty between consecutive prediction vectors."""
    lambda_w2: float = 1.0
    """L2 waviness penalty between consecutive prediction vectors."""


@MODEL_REGISTRY.register("dktplus")
class DKTPlus(DKT):
    """DKT+ (Yeung & Yeung, 2018): DKT with reconstruction and waviness regularisation."""

    Config = DKTPlusConfig
    memory_complexity = 1

    def forward(self, batch: Batch) -> ModelOutput:
        h, h_prev, _, reg = self._encode_states(batch)
        all_logits = self.out(self.dropout(h_prev))
        logits = all_logits.gather(-1, batch.concept_ids.unsqueeze(-1)).squeeze(-1)
        if self.is_rasch and batch.question_ids is not None:
            logits = logits + self.question_offset[batch.question_ids]

        # 1. Reconstruction regulariser (predict r_t from h_t after observing r_t)
        if self.cfg.lambda_r > 0:
            rec_all = self.out(self.dropout(h))
            rec_logits = rec_all.gather(-1, batch.concept_ids.unsqueeze(-1)).squeeze(-1)
            rec_bce = F.binary_cross_entropy_with_logits(
                rec_logits.float(), batch.responses.float(), reduction="none"
            )
            reg = reg + self.cfg.lambda_r * (rec_bce * batch.mask).sum()

        # 2. Waviness regularisers (w1 L1 and w2 L2 on consecutive concept predictions)
        if (self.cfg.lambda_w1 > 0 or self.cfg.lambda_w2 > 0) and batch.seq_len > 1:
            probs = torch.sigmoid(all_logits[:, :, 1:].float())
            diff = probs[:, 1:] - probs[:, :-1]
            valid_pair = (batch.mask[:, 1:] & batch.mask[:, :-1]).unsqueeze(-1)
            w1 = (diff.abs() * valid_pair).mean(-1).sum()
            w2 = (diff.pow(2) * valid_pair).mean(-1).sum()
            reg = reg + self.cfg.lambda_w1 * w1 + self.cfg.lambda_w2 * w2

        return ModelOutput(logits=logits, reg_loss=reg)
