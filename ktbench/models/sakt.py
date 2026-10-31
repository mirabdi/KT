"""SAKT: Self-Attentive Knowledge Tracing (Pandey & Karypis, EDM 2019)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
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
class SAKTConfig:
    d_model: int = 128
    n_heads: int = 8
    n_blocks: int = 1
    dropout: float = 0.2
    use_questions: bool = False
    """Query with question ids instead of concept ids (if available)."""
    use_rasch: bool = False
    """Table 5 ``SAKT-R`` variant: Rasch model-based question and interaction embeddings."""
    l2: float = 1e-5


class _SAKTBlock(nn.Module):
    def __init__(self, d: int, h: int, p: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.norm1 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, d))
        self.norm2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(p)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(q, kv, kv, attn_mask=mask, need_weights=False)
        x = self.norm1(q + self.drop(a))
        return self.norm2(x + self.drop(self.ffn(x)))


@MODEL_REGISTRY.register("sakt")
class SAKT(BaseKTModel):
    """Query = current exercise; keys/values = *past* interactions + positions.

    Interactions are shifted right by one behind a learned start token, so a
    standard causal mask (diagonal allowed) only exposes responses ``< t``.
    Right padding + causal masking means padding never influences real tokens,
    so no key-padding mask is needed.
    """

    Config = SAKTConfig
    memory_complexity = 2

    def __init__(self, cfg: SAKTConfig, meta: DatasetMeta) -> None:
        super().__init__(cfg, meta)
        self.n_concepts = meta.n_concepts
        self.is_rasch = cfg.use_rasch and meta.has_questions
        self.query_on_questions = cfg.use_questions and meta.has_questions
        if self.is_rasch:
            self.rasch = RaschEmbedding(meta.n_concepts, meta.n_questions, cfg.d_model,
                                        separate_qa=False, use_rasch=True)
            self.register_buffer("question_offset", torch.zeros(meta.n_questions + 1))
        else:
            n_query = (meta.n_questions if self.query_on_questions else meta.n_concepts) + 1
            self.interaction_embed = nn.Embedding(2 * meta.n_concepts + 1, cfg.d_model)
            self.query_embed = nn.Embedding(n_query, cfg.d_model)
        self.pos_embed = nn.Embedding(meta.max_seq_len + 1, cfg.d_model)
        self.start = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.start, std=0.02)
        self.blocks = nn.ModuleList([_SAKTBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
                                     for _ in range(cfg.n_blocks)])
        self.out = nn.Linear(cfg.d_model, 1)

    @torch.no_grad()
    def calibrate_rasch(self, loader, device: torch.device) -> None:
        if not self.is_rasch:
            return
        self.question_offset.zero_()
        delta = compute_fisher_rasch_offset(self, loader, self.meta.n_questions, device, tau=6.0)
        self.question_offset.copy_(0.90 * delta)
        self.rasch.difficulty.weight.data.copy_(-delta.unsqueeze(-1))

    def forward(self, batch: Batch) -> ModelOutput:
        B, L = batch.concept_ids.shape
        reg = 0.0
        if self.is_rasch:
            q, inter, mu = self.rasch(batch.concept_ids, batch.responses, batch.question_ids)
            if mu is not None and self.cfg.l2 > 0:
                reg = self.cfg.l2 * (mu.squeeze(-1).pow(2) * batch.mask).sum()
        else:
            inter = self.interaction_embed(interaction_index(batch.concept_ids, batch.responses, self.n_concepts))
            ids = batch.question_ids if self.query_on_questions else batch.concept_ids
            q = self.query_embed(ids)
        kv = torch.cat([self.start.expand(B, 1, -1), inter[:, :-1]], dim=1)   # interactions < t
        pos = torch.arange(L, device=kv.device).clamp(max=self.pos_embed.num_embeddings - 1)
        pe = self.pos_embed(pos)[None]
        kv = kv + pe
        q = q + pe
        mask = torch.ones(L, L, dtype=torch.bool, device=q.device).triu(1)  # True = blocked
        for blk in self.blocks:
            q = blk(q, kv, mask)
        logits = self.out(q).squeeze(-1)
        if self.is_rasch and batch.question_ids is not None:
            logits = logits + self.question_offset[batch.question_ids]
        return ModelOutput(logits=logits, reg_loss=reg)
