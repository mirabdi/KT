"""CKT: Convolutional Knowledge Tracing (Shen et al., SIGIR 2020).

Applies multi-scale causal 1D convolutional neural networks over sliding interaction
windows to capture local learning momentum, short-term trends, and individual learning rates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

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
class CKTConfig:
    d_model: int = 128
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    n_layers: int = 2
    dropout: float = 0.1
    use_questions: bool = True
    use_rasch: bool = False
    l2: float = 1e-5


class _CausalConv1dBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.k = kernel_size
        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size=kernel_size)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, L, D). Transpose to (B, D, L) for Conv1d
        B, L, D = x.shape
        x_t = x.transpose(1, 2)
        # Left-pad by (k - 1) for strict causality (no looking into future tokens)
        padded = F.pad(x_t, (self.k - 1, 0))
        h = self.conv(padded)
        a, b = h.chunk(2, dim=1)
        glu = a * torch.sigmoid(b)
        out = x + self.drop(glu.transpose(1, 2))
        return self.norm(out)


@MODEL_REGISTRY.register("ckt")
class CKT(BaseKTModel):
    """Convolutional Knowledge Tracing with multi-scale causal receptive fields (Shen et al., 2020)."""

    Config = CKTConfig
    memory_complexity = 1

    def __init__(self, cfg: CKTConfig, meta: DatasetMeta) -> None:
        super().__init__(cfg, meta)
        self.n_concepts = meta.n_concepts
        self.is_rasch = cfg.use_rasch and meta.has_questions
        self.has_q = cfg.use_questions and meta.has_questions

        if self.is_rasch:
            self.rasch = RaschEmbedding(meta.n_concepts, meta.n_questions, cfg.d_model,
                                        separate_qa=False, use_rasch=True)
            self.register_buffer("question_offset", torch.zeros(meta.n_questions + 1))
        else:
            self.concept_embed = nn.Embedding(meta.n_concepts + 1, cfg.d_model, padding_idx=0)
            if self.has_q:
                self.question_embed = nn.Embedding(meta.n_questions + 1, cfg.d_model, padding_idx=0)
            self.interaction_embed = nn.Embedding(2 * meta.n_concepts + 1, cfg.d_model, padding_idx=0)

        self.start = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.start, std=0.02)

        # Multi-scale conv branches
        self.conv_branches = nn.ModuleList([
            nn.Sequential(*[
                _CausalConv1dBlock(cfg.d_model, k, cfg.dropout)
                for _ in range(cfg.n_layers)
            ])
            for k in cfg.kernel_sizes
        ])

        total_conv_dim = len(cfg.kernel_sizes) * cfg.d_model
        self.fusion = nn.Linear(total_conv_dim, cfg.d_model)

        self.summary = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    @torch.no_grad()
    def calibrate_rasch(self, loader, device: torch.device) -> None:
        if not self.is_rasch:
            return
        self.question_offset.zero_()
        delta = compute_fisher_rasch_offset(self, loader, self.meta.n_questions, device, tau=6.0)
        self.question_offset.copy_(0.85 * delta)
        self.rasch.difficulty.weight.data.copy_(-delta.unsqueeze(-1))

    def forward(self, batch: Batch) -> ModelOutput:
        B, L = batch.concept_ids.shape
        reg = 0.0

        if self.is_rasch:
            q_emb, inter, mu = self.rasch(batch.concept_ids, batch.responses, batch.question_ids)
            if mu is not None and self.cfg.l2 > 0:
                reg = self.cfg.l2 * (mu.squeeze(-1).pow(2) * batch.mask).sum()
        else:
            q_emb = self.concept_embed(batch.concept_ids)
            if self.has_q and batch.question_ids is not None:
                q_emb = q_emb + self.question_embed(batch.question_ids)
            inter = self.interaction_embed(interaction_index(batch.concept_ids, batch.responses, self.n_concepts))

        # Shift past interactions right by 1 behind learned start token (causality)
        past_inter = torch.cat([self.start.expand(B, 1, -1), inter[:, :-1]], dim=1)

        branch_outs = [branch(past_inter) for branch in self.conv_branches]
        conv_fused = self.fusion(torch.cat(branch_outs, dim=-1))

        logits = self.summary(torch.cat([conv_fused, q_emb], dim=-1)).squeeze(-1)
        if self.is_rasch and batch.question_ids is not None:
            logits = logits + self.question_offset[batch.question_ids]

        return ModelOutput(logits=logits, reg_loss=reg)
