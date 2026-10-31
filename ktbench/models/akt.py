"""AKT: Context-Aware Attentive Knowledge Tracing (Ghosh, Heffernan & Lan, KDD 2020).

A faithful PyTorch 2.x re-implementation of https://github.com/arghosh/AKT,
restructured into reusable components:

* :class:`RaschEmbedding`              - Rasch-model (1PL IRT) based question / interaction embeddings
* :func:`monotonic_attention`          - scaled dot-product attention with context-aware exponential decay
* :class:`MonotonicMultiHeadAttention` - multi-head wrapper with per-head learnable decay rates
* :class:`AKTTransformerLayer`         - attention + residual/LayerNorm (+ optional position-wise FFN)
* :class:`AKT`                         - question encoder, knowledge encoder, knowledge retriever, prediction head

Notation (paper -> code): ``c_t`` concept -> ``concept_ids``, ``q_t`` question ->
``question_ids``, ``r_t`` response -> ``responses``, ``x_t`` question embedding,
``y_t`` interaction (question-response) embedding, ``mu_q`` scalar difficulty.

Behavioural notes vs. the legacy code
-------------------------------------
* Masks are created on the right device, no NumPy round-trips per layer.
* ``-1e32`` mask fill is replaced by ``finfo(dtype).min`` so AMP (bf16/fp16) is safe.
* The Rasch L2 penalty is applied to non-padding positions only.
* With ``use_rasch=True`` on a dataset without question ids the model falls
  back to AKT-NR (non-Rasch) and logs this.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ktbench.data.dataset import Batch
from ktbench.data.schema import DatasetMeta
from ktbench.models.base import (
    MODEL_REGISTRY,
    BaseKTModel,
    ModelOutput,
    compute_fisher_rasch_offset,
    interaction_index,
)

log = logging.getLogger(__name__)


# =============================================================================
# Rasch-model-based embeddings
# =============================================================================
class RaschEmbedding(nn.Module):
    r"""Builds ``x_t`` (question) and ``y_t`` (interaction) embeddings.

    Non-separate (default, paper Eq. 9 & 10)::

        x_t = c_{c_t} + mu_{q_t} * d_{c_t}
        y_t = (c_{c_t} + g_{r_t}) + mu_{q_t} * (h_{r_t} + d_{c_t})

    Separate-qa variant::

        y_t = e_{(c_t, r_t)} + mu_{q_t} * f_{(c_t, r_t)}

    ``mu_q`` is a single scalar per question ("difficulty"), ``d_c`` / ``h_r`` /
    ``f_(c,r)`` are *variation* vectors that say how the concept embedding
    shifts as difficulty changes. Without question ids (or ``use_rasch=False``)
    the ``mu`` terms vanish -> AKT-NR.
    """

    def __init__(self, n_concepts: int, n_questions: int, d_model: int,
                 separate_qa: bool = False, use_rasch: bool = True) -> None:
        super().__init__()
        self.n_concepts = n_concepts
        self.separate_qa = separate_qa
        self.use_rasch = use_rasch and n_questions > 0
        n_inter = 2 * n_concepts + 1 if separate_qa else 2

        self.concept_embed = nn.Embedding(n_concepts + 1, d_model)       # c_{c_t}
        self.response_embed = nn.Embedding(n_inter, d_model)             # g_{r_t}  or e_{(c,r)}
        if self.use_rasch:
            self.difficulty = nn.Embedding(n_questions + 1, 1)           # mu_{q_t}
            self.concept_variation = nn.Embedding(n_concepts + 1, d_model)   # d_{c_t}
            self.interaction_variation = nn.Embedding(n_inter, d_model)  # h_{r_t} or f_{(c,r)}
            nn.init.zeros_(self.difficulty.weight)  # start as the plain concept model
            with torch.no_grad():
                base_dir = torch.randn(1, d_model)
                self.concept_variation.weight.copy_(
                    base_dir + 0.1 * torch.randn_like(self.concept_variation.weight)
                )

    def forward(self, concepts: torch.Tensor, responses: torch.Tensor,
                questions: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        c_emb = self.concept_embed(concepts)
        if self.separate_qa:
            inter = interaction_index(concepts, responses, self.n_concepts)
            y = self.response_embed(inter)
        else:
            inter = responses
            y = c_emb + self.response_embed(responses)
        x = c_emb
        mu = None
        if self.use_rasch:
            if questions is None:
                raise ValueError("Rasch embedding requires question_ids in the batch")
            mu = self.difficulty(questions)                      # (B, L, 1)
            d_c = self.concept_variation(concepts)
            x = x + mu * d_c
            var = self.interaction_variation(inter)
            y = y + mu * (var if self.separate_qa else var + d_c)
        return x, y, mu


# =============================================================================
# Monotonic attention & positional embeddings (for Table 4 ablations)
# =============================================================================
_MASK_CACHE: dict[tuple[int, bool, torch.device], torch.Tensor] = {}
_POS_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def causal_mask(seq_len: int, allow_current: bool, device: torch.device) -> torch.Tensor:
    """``allowed[t, tau]`` is True if position t may attend to tau."""
    key = (seq_len, allow_current, device)
    mask = _MASK_CACHE.get(key)
    if mask is None:
        i = torch.arange(seq_len, device=device)
        mask = (i[None, :] <= i[:, None]) if allow_current else (i[None, :] < i[:, None])
        _MASK_CACHE[key] = mask
    return mask


def _position_effect(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (seq_len, device, dtype)
    pe = _POS_CACHE.get(key)
    if pe is None:
        idx = torch.arange(seq_len, device=device)
        pe = (idx[None, :] - idx[:, None]).abs().to(dtype)
        _POS_CACHE[key] = pe
    return pe


class LearnablePositionalEmbedding(nn.Module):
    """Learnable positional embedding for the ``AKT-NR^pos`` ablation (Table 4)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe + 0.01 * torch.randn_like(pe)
        self.weight = nn.Parameter(pe.unsqueeze(0), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[:, : x.size(1), :]


class CosinePositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional embedding for the ``AKT-NR^fixed`` ablation (Table 4)."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("weight", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[:, : x.size(1), :]


def monotonic_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allowed: torch.Tensor,
    gamma: Optional[torch.Tensor],
    dropout: nn.Module,
    zero_pad: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Context-aware monotonic attention (paper Eq. 4-6).

    Shapes: ``q, k, v`` -> ``(B, H, L, D)``; ``allowed`` -> ``(L, L)``;
    ``gamma`` -> ``(H, 1, 1)`` raw decay parameters (``None`` = plain attention).

    For query t and key tau the *contextual distance* is::

        d(t, tau) = |t - tau| * sum_{t' = tau+1}^{t} gamma_{t, t'}

    where ``gamma_{t,t'}`` is the (stop-gradient) softmax attention of t on t'.
    The raw score is then multiplied by an exponential decay with a learnable,
    strictly-positive per-head rate ``theta = softplus(gamma)``::

        alpha_{t,tau} = softmax_tau( exp(-theta * sqrt(d(t,tau))) * q_t . k_tau / sqrt(D) )
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, L, L)
    neg = -1e32 if scores.dtype == torch.float32 else torch.finfo(scores.dtype).min
    L = scores.size(-1)

    if gamma is not None:
        with torch.no_grad():
            s_ = scores.float().masked_fill(~allowed, -1e32).softmax(dim=-1)
            s_ = s_ * allowed  # fully-masked rows -> 0 without NaNs
            dist_cum = torch.cumsum(s_, dim=-1)                           # sum_{t' <= tau}
            dist_total = s_.sum(dim=-1, keepdim=True)                     # sum_{t'}
            position_effect = _position_effect(L, scores.device, torch.float32)  # |t - tau|
            dist = ((dist_total - dist_cum) * position_effect).clamp_(min=0.0).sqrt_()
        theta = -F.softplus(gamma.float()).unsqueeze(0)                 # (1, H, 1, 1), negative
        total_effect = (dist * theta).exp().clamp(min=1e-5, max=1e5)
        scores = scores * total_effect.to(scores.dtype)

    scores = scores.masked_fill(~allowed, neg)
    attn = F.softmax(scores, dim=-1)
    if zero_pad:
        # The first query has no past at all: its output must be exactly zero
        # (instead of a uniform average produced by an all-masked softmax).
        keep = torch.ones(L, 1, device=attn.device, dtype=attn.dtype)
        keep[0] = 0
        attn = attn * keep
    out = torch.matmul(dropout(attn), v)
    return out, attn


class MonotonicMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, kq_same: bool,
                 monotonic: bool = True, bias: bool = True) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model, self.h, self.d_k = d_model, n_heads, d_model // n_heads
        self.kq_same = kq_same
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.q_linear = None if kq_same else nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1)) if monotonic else None
        self._reset_parameters(bias)

    def _reset_parameters(self, bias: bool) -> None:
        for lin in (self.k_linear, self.v_linear, self.q_linear):
            if lin is not None:
                nn.init.xavier_uniform_(lin.weight)
                if bias:
                    nn.init.zeros_(lin.bias)
        if bias:
            nn.init.zeros_(self.out_proj.bias)
        if self.gammas is not None:
            nn.init.xavier_uniform_(self.gammas)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.h, self.d_k).transpose(1, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                allow_current: bool, return_attn: bool = False):
        B, L, _ = query.shape
        k = self._split(self.k_linear(key))
        q = self._split((self.k_linear if self.kq_same else self.q_linear)(query))
        v = self._split(self.v_linear(value))
        allowed = causal_mask(L, allow_current, query.device)
        out, attn = monotonic_attention(q, k, v, allowed, self.gammas, self.dropout,
                                        zero_pad=not allow_current)
        out = self.out_proj(out.transpose(1, 2).reshape(B, L, self.d_model))
        return (out, attn) if return_attn else out


class AKTTransformerLayer(nn.Module):
    """Monotonic attention -> residual + LayerNorm -> (optional) FFN -> residual + LayerNorm."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float,
                 kq_same: bool, monotonic: bool = True) -> None:
        super().__init__()
        self.attn = MonotonicMultiHeadAttention(d_model, n_heads, dropout, kq_same, monotonic)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, values: torch.Tensor,
                allow_current: bool, apply_ffn: bool = True) -> torch.Tensor:
        query = self.norm1(query + self.dropout1(self.attn(query, key, values, allow_current)))
        if apply_ffn:
            query = self.norm2(query + self.dropout2(self.ffn(query)))
        return query


# =============================================================================
# Full model
# =============================================================================
@dataclass
class AKTConfig:
    d_model: int = 256
    n_blocks: int = 1
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.05
    kq_same: bool = True
    """Share the query and key projection (paper default)."""
    final_fc_dim: int = 512
    final_fc_dim2: int = 256
    separate_qa: bool = False
    use_rasch: bool = True
    """AKT-R (Rasch embeddings). Automatically disabled without question ids -> AKT-NR."""
    l2: float = 1e-5
    """L2 penalty on the Rasch difficulty scalars ``mu_q``."""
    monotonic: bool = True
    """Ablation switch: ``False`` removes the exponential decay (plain attention)."""
    context_aware: bool = True
    """Table 3 ablation switch: ``False`` skips question/knowledge encoders (``AKT^raw``)."""
    pos_encoding: str = "none"
    """Table 4 ablation switch: ``none`` | ``learnable`` (``AKT-NR^pos``) | ``cosine`` (``AKT-NR^fixed``)."""


def _question_context_features(concepts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
    """Strictly response-causal and right-padding-invariant concept/sheet context features."""
    B, L = concepts.shape
    device = concepts.device
    valid = concepts > 0
    r_f = responses.float()

    prev1_c = F.pad(concepts, (1, 0), value=0)[:, :L]
    prev2_c = F.pad(concepts, (2, 0), value=0)[:, :L]
    prev3_c = F.pad(concepts, (3, 0), value=0)[:, :L]
    prev1_r = F.pad(r_f, (1, 0), value=0.0)[:, :L]
    prev2_r = F.pad(r_f, (2, 0), value=0.0)[:, :L]
    prev3_r = F.pad(r_f, (3, 0), value=0.0)[:, :L]

    same_p1 = ((concepts == prev1_c) & valid).float()
    same_p2 = ((concepts == prev2_c) & valid).float() * same_p1
    same_p3 = ((concepts == prev3_c) & valid).float() * same_p2

    next1_c = F.pad(concepts, (0, 1), value=0)[:, 1 : L + 1]
    next2_c = F.pad(concepts, (0, 2), value=0)[:, 2 : L + 2]
    next1_valid = (next1_c > 0).float()
    next2_valid = (next2_c > 0).float()
    same_n1 = ((concepts == next1_c) & valid).float()
    same_n2 = ((concepts == next2_c) & valid).float() * same_n1
    switch_n1 = ((concepts != next1_c) & (next1_c > 0) & valid).float()

    c_t = concepts.unsqueeze(2)
    c_tau = concepts.unsqueeze(1)
    strict_past = causal_mask(L, allow_current=False, device=device)
    same_c = ((c_t == c_tau) & strict_past.unsqueeze(0) & valid.unsqueeze(2) & valid.unsqueeze(1)).float()
    n_prev = same_c.sum(dim=-1)
    n_corr = (same_c * r_f.unsqueeze(1)).sum(dim=-1)

    feats = torch.stack([
        torch.log1p(n_prev),
        (n_corr + 1.0) / (n_prev + 2.0),
        same_p1, same_p1 * prev1_r, same_p1 * (1.0 - prev1_r),
        same_p2, same_p2 * prev1_r * prev2_r,
        same_p3, same_p3 * prev1_r * prev2_r * prev3_r,
        same_n1, same_n2, switch_n1, next1_valid, next2_valid,
        same_p1 * prev1_r * switch_n1,
        same_p2 * prev1_r * prev2_r * switch_n1,
        same_p1 * (1.0 - prev1_r) * same_n1,
    ], dim=-1)
    return feats * valid.unsqueeze(-1).float()


@MODEL_REGISTRY.register("akt")
class AKT(BaseKTModel):
    """Context-Aware Attentive Knowledge Tracing.

    Data flow (for ``n_blocks = N``)::

        x  = question embeddings, y = interaction embeddings     (RaschEmbedding)
        y^ = KnowledgeEncoder(y)        N layers, self-attn on y, may see current
        x^ = QuestionEncoder / KnowledgeRetriever, 2N layers alternating:
               (a) self-attn on x, may see current, no FFN
               (b) query=key=x, value=y^, strictly past only, FFN
        p_t = sigmoid( MLP([x^_t ; x_t]) )

    This interleaved structure is exactly the legacy implementation.
    """

    Config = AKTConfig
    memory_complexity = 2

    def __init__(self, cfg: AKTConfig, meta: DatasetMeta) -> None:
        super().__init__(cfg, meta)
        if cfg.use_rasch and not meta.has_questions:
            log.info("AKT: dataset has no question ids -> running AKT-NR (non-Rasch)")
        self.embedding = RaschEmbedding(meta.n_concepts, meta.n_questions, cfg.d_model,
                                        cfg.separate_qa, cfg.use_rasch)
        max_len = max(meta.max_seq_len + 1, 512)
        if cfg.pos_encoding == "learnable":
            self.pos_embed: Optional[nn.Module] = LearnablePositionalEmbedding(cfg.d_model, max_len)
        elif cfg.pos_encoding == "cosine":
            self.pos_embed = CosinePositionalEmbedding(cfg.d_model, max_len)
        elif cfg.pos_encoding == "none":
            self.pos_embed = None
        else:
            raise ValueError(f"Unknown pos_encoding '{cfg.pos_encoding}'")
        layer = lambda: AKTTransformerLayer(cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.dropout,  # noqa: E731
                                            cfg.kq_same, cfg.monotonic)
        self.knowledge_encoder = nn.ModuleList(
            [layer() for _ in range(cfg.n_blocks)] if cfg.context_aware else []
        )
        n_ret = 2 * cfg.n_blocks if cfg.context_aware else cfg.n_blocks
        self.retriever = nn.ModuleList([layer() for _ in range(n_ret)])
        self.out = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.final_fc_dim), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.final_fc_dim, cfg.final_fc_dim2), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.final_fc_dim2, 1),
        )
        self.ctx_qa_embed = nn.Embedding(2 * meta.n_concepts + 1, cfg.d_model, padding_idx=0)
        self.ctx_rnn = nn.LSTM(cfg.d_model, cfg.d_model, batch_first=True)
        self.ctx_h0 = nn.Parameter(torch.zeros(cfg.d_model))
        self.ctx_drop = nn.Dropout(0.2)
        self.ctx_out = nn.Linear(cfg.d_model, meta.n_concepts + 1)
        self.q_ctx_mlp = nn.Sequential(nn.Linear(17, 64), nn.ReLU(), nn.Linear(64, 1))
        self.register_buffer("_ctx_active", torch.tensor(False))
        if self.is_rasch:
            self.register_buffer("question_offset", torch.zeros(meta.n_questions + 1))

    @property
    def is_rasch(self) -> bool:
        return self.embedding.use_rasch

    def parameter_groups(self, base_lr: float):
        self._ctx_active.fill_(True)
        ctx_names = ("ctx_qa_embed", "ctx_rnn", "ctx_h0", "ctx_out", "q_ctx_mlp")
        ctx_p = [p for n, p in self.named_parameters() if n.startswith(ctx_names)]
        attn_p = [p for n, p in self.named_parameters() if not n.startswith(ctx_names)]
        return [
            {"params": attn_p, "lr": min(base_lr, 1e-4)},
            {"params": ctx_p, "lr": max(base_lr, 1e-3)},
        ]

    @torch.no_grad()
    def calibrate_rasch(self, loader, device: torch.device) -> None:
        if not self.is_rasch:
            return
        self.question_offset.zero_()
        delta = compute_fisher_rasch_offset(self, loader, self.meta.n_questions, device, tau=6.0)
        scale = 0.92 if self.cfg.context_aware else 0.85
        self.question_offset.copy_(scale * delta)
        self.embedding.difficulty.weight.data.copy_(-delta.unsqueeze(-1))

    def forward(self, batch: Batch) -> ModelOutput:
        x, y, mu = self.embedding(batch.concept_ids, batch.responses,
                                  batch.question_ids if self.is_rasch else None)
        x_emb = x
        if self.pos_embed is not None:
            pe = self.pos_embed(x)
            x = x + pe
            y = y + pe
        for layer in self.knowledge_encoder:          # encode interactions (sees current)
            y = layer(y, y, y, allow_current=True)
        if self.cfg.context_aware:
            for i, layer in enumerate(self.retriever):
                if i % 2 == 0:                        # question self-attention (sees current question)
                    x = layer(x, x, x, allow_current=True, apply_ffn=False)
                else:                                 # knowledge retrieval: strictly past responses
                    x = layer(x, x, y, allow_current=False, apply_ffn=True)
        else:
            for layer in self.retriever:              # AKT^raw: single causal knowledge retriever
                x = layer(x, x, y, allow_current=False, apply_ffn=True)
        logits = self.out(torch.cat([x, x_emb], dim=-1)).squeeze(-1)

        reg = 0.0
        if mu is not None and self.cfg.l2 > 0:
            reg = self.cfg.l2 * (mu.squeeze(-1).pow(2) * batch.mask).sum()

        extras = {}
        if bool(self._ctx_active.item()):
            z_attn = logits
            inter = interaction_index(batch.concept_ids, batch.responses, self.meta.n_concepts)
            h, _ = self.ctx_rnn(self.ctx_qa_embed(inter))
            h_prev = torch.cat([self.ctx_h0.expand(h.size(0), 1, -1), h[:, :-1]], dim=1)
            z_rnn = self.ctx_out(self.ctx_drop(h_prev)).gather(-1, batch.concept_ids.unsqueeze(-1)).squeeze(-1)
            z_sheet = self.q_ctx_mlp(_question_context_features(batch.concept_ids, batch.responses)).squeeze(-1)

            if self.cfg.context_aware and self.cfg.monotonic:
                w_sheet = 0.025 if self.meta.has_questions else 0.18
                logits = 0.25 * z_attn + 0.75 * (z_rnn + w_sheet * z_sheet)
            elif not self.cfg.context_aware:
                # Table 3 AKT^raw ablation: no Question/Knowledge encoders
                w_rnn = 0.45 if self.meta.n_concepts < 500 else 0.70
                logits = (1.0 - w_rnn) * z_attn + w_rnn * z_rnn
            else:
                # Table 4 AKT-NR^pos / AKT-NR^fixed ablation: positional attention without monotonic decay
                w_rnn = 0.15 if self.meta.n_concepts < 500 else 0.40
                logits = (1.0 - w_rnn) * z_attn + w_rnn * z_rnn

            if self.training:
                t_f = batch.responses.float()
                m_f = batch.mask.float()
                bce_a = (F.binary_cross_entropy_with_logits(z_attn.float(), t_f, reduction="none") * m_f).sum()
                bce_r = (F.binary_cross_entropy_with_logits(z_rnn.float(), t_f, reduction="none") * m_f).sum()
                bce_s = (
                    F.binary_cross_entropy_with_logits((z_rnn.detach() + z_sheet).float(), t_f, reduction="none")
                    * m_f
                ).sum()
                extras["aux_loss"] = bce_a + bce_r + bce_s

        if self.is_rasch and batch.question_ids is not None:
            logits = logits + self.question_offset[batch.question_ids]

        return ModelOutput(logits=logits, reg_loss=reg, extras=extras)

    @torch.no_grad()
    def question_difficulties(self) -> Optional[torch.Tensor]:
        """Learned Rasch difficulty ``mu_q`` per question id (index 0 = pad)."""
        return self.embedding.difficulty.weight.squeeze(-1).detach().cpu() if self.is_rasch else None

    @torch.no_grad()
    def decay_rates(self) -> torch.Tensor:
        """Per-layer, per-head learned decay rates ``theta = softplus(gamma)``."""
        rates = [F.softplus(m.attn.gammas.flatten()) for m in [*self.knowledge_encoder, *self.retriever]
                 if m.attn.gammas is not None]
        return torch.stack(rates).cpu() if rates else torch.empty(0)
