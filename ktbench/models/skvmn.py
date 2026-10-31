"""SKVMN: Sequential Key-Value Memory Networks (Abdelrahman & Wang, SIGIR 2019).

Evolving DKVMN, SKVMN introduces a sequential Hop-LSTM mechanism that chains
knowledge states across conceptually similar exercises over time, capturing
longitudinal learning trajectories beyond single-step Markovian updates.
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
class SKVMNConfig:
    memory_size: int = 30
    key_dim: int = 32
    value_dim: int = 64
    hop_lstm_dim: int = 64
    final_fc_dim: int = 32
    dropout: float = 0.1
    init_std: float = 0.1
    use_rasch: bool = False
    l2: float = 1e-5


@MODEL_REGISTRY.register("skvmn")
class SKVMN(BaseKTModel):
    """Sequential Key-Value Memory Network with Hop-LSTM (Abdelrahman & Wang, SIGIR 2019)."""

    Config = SKVMNConfig
    memory_complexity = 1

    def __init__(self, cfg: SKVMNConfig, meta: DatasetMeta) -> None:
        super().__init__(cfg, meta)
        self.n_concepts = meta.n_concepts
        self.is_rasch = cfg.use_rasch and meta.has_questions

        self.q_embed = nn.Embedding(meta.n_concepts + 1, cfg.key_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * meta.n_concepts + 1, cfg.value_dim, padding_idx=0)
        nn.init.normal_(self.q_embed.weight, std=cfg.init_std)
        nn.init.normal_(self.qa_embed.weight, std=cfg.init_std)
        with torch.no_grad():
            self.q_embed.weight[0].zero_()
            self.qa_embed.weight[0].zero_()

        if self.is_rasch:
            self.rasch = RaschEmbedding(meta.n_concepts, meta.n_questions, cfg.value_dim,
                                        separate_qa=False, use_rasch=True)
            self.register_buffer("question_offset", torch.zeros(meta.n_questions + 1))

        self.Mk = nn.Parameter(torch.empty(cfg.memory_size, cfg.key_dim))
        self.Mv0 = nn.Parameter(torch.empty(cfg.memory_size, cfg.value_dim))
        nn.init.normal_(self.Mk, std=0.5)
        nn.init.normal_(self.Mv0, std=cfg.init_std)

        self.erase_linear = nn.Linear(cfg.value_dim, cfg.value_dim)
        self.add_linear = nn.Linear(cfg.value_dim, cfg.value_dim)

        # Hop-LSTM capturing longitudinal concept trajectories
        self.hop_lstm = nn.LSTM(cfg.value_dim + cfg.key_dim, cfg.hop_lstm_dim, batch_first=True)
        self.h0 = nn.Parameter(torch.zeros(1, 1, cfg.hop_lstm_dim))

        self.summary = nn.Sequential(
            nn.Linear(cfg.value_dim + cfg.hop_lstm_dim + cfg.key_dim, cfg.final_fc_dim),
            nn.Tanh(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.final_fc_dim, 1),
        )

    @torch.no_grad()
    def calibrate_rasch(self, loader, device: torch.device) -> None:
        if not self.is_rasch:
            return
        self.question_offset.zero_()
        delta = compute_fisher_rasch_offset(self, loader, self.meta.n_questions, device, tau=6.0)
        self.question_offset.copy_(0.80 * delta)
        self.rasch.difficulty.weight.data.copy_(-delta.unsqueeze(-1))

    def forward(self, batch: Batch) -> ModelOutput:
        B, L = batch.concept_ids.shape
        reg = 0.0
        k = self.q_embed(batch.concept_ids)
        v = self.qa_embed(interaction_index(batch.concept_ids, batch.responses, self.n_concepts))

        if self.is_rasch:
            _, _, mu = self.rasch(batch.concept_ids, batch.responses, batch.question_ids)
            if mu is not None and self.cfg.l2 > 0:
                reg = self.cfg.l2 * (mu.squeeze(-1).pow(2) * batch.mask).sum()

        w = F.softmax(torch.matmul(k, self.Mk.t()), dim=-1)  # (B, L, N)
        e = torch.sigmoid(self.erase_linear(v))
        a = torch.tanh(self.add_linear(v))

        Mv = self.Mv0.unsqueeze(0).expand(B, -1, -1).contiguous()
        reads = []
        for t in range(L):
            wt = w[:, t, :]                                    # (B, N)
            read_t = torch.bmm(wt.unsqueeze(1), Mv).squeeze(1) # (B, D_v)
            reads.append(read_t)
            # Write update for next step
            et = e[:, t, :].unsqueeze(1)                       # (B, 1, D_v)
            at = a[:, t, :].unsqueeze(1)                       # (B, 1, D_v)
            wt_u = wt.unsqueeze(-1)                            # (B, N, 1)
            Mv = Mv * (1.0 - wt_u * et) + wt_u * at

        read_all = torch.stack(reads, dim=1)                   # (B, L, D_v)

        # Hop-LSTM processes sequential [read_t, k_t]
        hop_inp = torch.cat([read_all, k], dim=-1)
        hop_h, _ = self.hop_lstm(hop_inp)
        # Shift hop_h right by 1 for causal prediction
        hop_prev = torch.cat([self.h0.expand(B, 1, -1), hop_h[:, :-1]], dim=1)

        logits = self.summary(torch.cat([read_all, hop_prev, k], dim=-1)).squeeze(-1)
        if self.is_rasch and batch.question_ids is not None:
            logits = logits + self.question_offset[batch.question_ids]

        return ModelOutput(logits=logits, reg_loss=reg)
