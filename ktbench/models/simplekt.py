"""simpleKT: A Simple but Tough-to-Beat Baseline for Knowledge Tracing (Liu et al., NeurIPS 2023).

Challenging complex multi-stream decoders, simpleKT uses standard scaled dot-product
attention directly between target exercise queries and past interaction representations,
incorporating Rasch-inspired item difficulty vectors.
"""

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
class SimpleKTConfig:
    d_model: int = 128
    n_heads: int = 4
    n_blocks: int = 1
    d_ff: int = 256
    dropout: float = 0.1
    use_questions: bool = True
    use_rasch: bool = True
    l2: float = 1e-5


class _SimpleKTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(q, kv, kv, attn_mask=mask, need_weights=False)
        x = self.norm1(q + self.drop(a))
        return self.norm2(x + self.drop(self.ffn(x)))


@MODEL_REGISTRY.register("simplekt")
class SimpleKT(BaseKTModel):
    """Minimalist Knowledge Tracing with Rasch-modeled dot-product attention (Liu et al., NeurIPS 2023)."""

    Config = SimpleKTConfig
    memory_complexity = 2

    def __init__(self, cfg: SimpleKTConfig, meta: DatasetMeta) -> None:
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

        self.pos_embed = nn.Embedding(meta.max_seq_len + 1, cfg.d_model)
        self.start = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.start, std=0.02)

        self.blocks = nn.ModuleList([
            _SimpleKTBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_blocks)
        ])

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
        self.question_offset.copy_(0.90 * delta)
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

        # Strictly past interactions shifted right by 1
        kv = torch.cat([self.start.expand(B, 1, -1), inter[:, :-1]], dim=1)

        pos = torch.arange(L, device=q_emb.device).clamp(max=self.pos_embed.num_embeddings - 1)
        pe = self.pos_embed(pos)[None]
        q_emb = q_emb + pe
        kv = kv + pe

        mask = torch.ones(L, L, dtype=torch.bool, device=q_emb.device).triu(1)

        h = q_emb
        for blk in self.blocks:
            h = blk(h, kv, mask)

        logits = self.summary(torch.cat([h, q_emb], dim=-1)).squeeze(-1)
        if self.is_rasch and batch.question_ids is not None:
            logits = logits + self.question_offset[batch.question_ids]

        return ModelOutput(logits=logits, reg_loss=reg)
