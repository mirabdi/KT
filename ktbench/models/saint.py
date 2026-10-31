"""SAINT: Separated Self-Attentive Neural Knowledge Tracing (Choi et al., EDM 2020).

SAINT decouples the curriculum path (exercise/question concepts) from the
performance path (past responses/interactions) into separate Transformer
Encoder and Decoder stacks with strictly causal cross-attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
class SAINTConfig:
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 1
    n_decoder_layers: int = 1
    d_ff: int = 256
    dropout: float = 0.1
    use_questions: bool = True
    use_rasch: bool = False
    l2: float = 1e-5


class _TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a, _ = self.self_attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + self.drop(a))
        return self.norm2(x + self.drop(self.ffn(x)))


class _TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Self-attention over past response sequence
        a1, _ = self.self_attn(tgt, tgt, tgt, attn_mask=mask, need_weights=False)
        tgt = self.norm1(tgt + self.drop(a1))
        # Cross-attention: queries from decoder, keys/values from exercise encoder
        a2, _ = self.cross_attn(tgt, memory, memory, attn_mask=mask, need_weights=False)
        tgt = self.norm2(tgt + self.drop(a2))
        return self.norm3(tgt + self.drop(self.ffn(tgt)))


@MODEL_REGISTRY.register("saint")
class SAINT(BaseKTModel):
    """Separated Self-Attentive Neural KT (Choi et al., EDM 2020).

    Encoder processes exercises: c_1:t (and optional q_1:t).
    Decoder processes past interactions: (c_0:t-1, r_0:t-1) with causal self-attention
    and cross-attention to the encoder's exercise states.
    """

    Config = SAINTConfig
    memory_complexity = 2

    def __init__(self, cfg: SAINTConfig, meta: DatasetMeta) -> None:
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
        self.start_resp = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.start_resp, std=0.02)

        self.encoder_layers = nn.ModuleList([
            _TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            _TransformerDecoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_decoder_layers)
        ])

        self.out = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
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
            enc_inp, inter, mu = self.rasch(batch.concept_ids, batch.responses, batch.question_ids)
            if mu is not None and self.cfg.l2 > 0:
                reg = self.cfg.l2 * (mu.squeeze(-1).pow(2) * batch.mask).sum()
        else:
            enc_inp = self.concept_embed(batch.concept_ids)
            if self.has_q and batch.question_ids is not None:
                enc_inp = enc_inp + self.question_embed(batch.question_ids)
            inter = self.interaction_embed(interaction_index(batch.concept_ids, batch.responses, self.n_concepts))

        # Shift past responses right by 1 behind learned start token (causality)
        dec_inp = torch.cat([self.start_resp.expand(B, 1, -1), inter[:, :-1]], dim=1)

        pos = torch.arange(L, device=enc_inp.device).clamp(max=self.pos_embed.num_embeddings - 1)
        pe = self.pos_embed(pos)[None]
        enc_inp = enc_inp + pe
        dec_inp = dec_inp + pe

        mask = torch.ones(L, L, dtype=torch.bool, device=enc_inp.device).triu(1)

        # 1. Exercise Encoder
        enc_state = enc_inp
        for layer in self.encoder_layers:
            enc_state = layer(enc_state, mask)

        # 2. Response Decoder (cross-attention to exercise states)
        dec_state = dec_inp
        for layer in self.decoder_layers:
            dec_state = layer(dec_state, enc_state, mask)

        logits = self.out(dec_state).squeeze(-1)
        if self.is_rasch and batch.question_ids is not None:
            logits = logits + self.question_offset[batch.question_ids]

        return ModelOutput(logits=logits, reg_loss=reg)
