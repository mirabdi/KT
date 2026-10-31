"""CogKT: Cognitive Dual-Process & 2PL-Psychometric Knowledge Tracing.

A novel Knowledge Tracing architecture synthesizing:
1. Psychometric 2PL-Rasch Item Response Theory (discrimination alpha_q + difficulty mu_q
   with hierarchical empirical Bayes shrinkage).
2. Cognitive Dual-Process Architecture:
   - Stream A (Working Memory): Multi-scale causal convolutions tracking immediate
     cognitive momentum, recent errors, and trial-and-error trajectory.
   - Stream B (Long-Term Associative Store): Context-aware monotonic multi-head attention
     with adaptive Ebbinghaus consolidation decay.
3. Cognitive Gating Fusion: Dynamic learned psychometric gate balancing short-term momentum
   against long-term concept retrieval.
4. Strictly causal and padding-invariant by construction.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ktbench.data.dataset import Batch
from ktbench.data.schema import DatasetMeta
from ktbench.models.akt import (
    CosinePositionalEmbedding,
    LearnablePositionalEmbedding,
    causal_mask,
    monotonic_attention,
)
from ktbench.models.base import (
    MODEL_REGISTRY,
    BaseKTModel,
    ModelOutput,
    compute_fisher_rasch_offset,
    interaction_index,
)

log = logging.getLogger(__name__)


# =============================================================================
# 1. 2PL-Psychometric Item Embeddings (Discrimination + Difficulty)
# =============================================================================
class Psychometric2PLEmbedding(nn.Module):
    r"""2PL-Rasch Item Response Theory embedding.

    Questions are parameterized along two psychometric dimensions:
    - Scalar Difficulty: \mu_q \in \mathbb{R} (initialized to 0, shrunk via L2).
    - Scalar Discrimination: \alpha_q = 2 * \sigma(\nu_q) \in (0, 2), centered at 1.0.

    Embeddings:
        Question:    x_t = \alpha_{q_t} * (c_{c_t} + \mu_{q_t} * d_{c_t})
        Interaction: y_t = (c_{c_t} + g_{r_t}) + \mu_{q_t} * (d_{c_t} + h_{r_t})
    """

    def __init__(
        self,
        n_concepts: int,
        n_questions: int,
        d_model: int,
        use_rasch: bool = True,
        use_2pl: bool = True,
    ) -> None:
        super().__init__()
        self.n_concepts = n_concepts
        self.n_questions = n_questions
        self.use_rasch = use_rasch and n_questions > 0
        self.use_2pl = use_2pl and self.use_rasch

        self.concept_embed = nn.Embedding(n_concepts + 1, d_model, padding_idx=0)
        self.response_embed = nn.Embedding(2, d_model)  # binary response embedding g_r

        if self.use_rasch:
            self.difficulty = nn.Embedding(n_questions + 1, 1, padding_idx=0)
            self.concept_variation = nn.Embedding(n_concepts + 1, d_model, padding_idx=0)
            self.response_variation = nn.Embedding(2, d_model)
            nn.init.zeros_(self.difficulty.weight)
            with torch.no_grad():
                base_dir = torch.randn(1, d_model)
                self.concept_variation.weight.copy_(
                    base_dir + 0.1 * torch.randn_like(self.concept_variation.weight)
                )
                self.concept_variation.weight[0].zero_()

            if self.use_2pl:
                # Discrimination parameterization: alpha_q = 2 * sigmoid(nu_q)
                self.log_discrimination = nn.Embedding(n_questions + 1, 1, padding_idx=0)
                nn.init.zeros_(self.log_discrimination.weight)

    def forward(
        self,
        concepts: torch.Tensor,
        responses: torch.Tensor,
        questions: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        c_emb = self.concept_embed(concepts)
        r_emb = self.response_embed(responses)
        y = c_emb + r_emb
        x = c_emb

        mu = None
        alpha = None

        if self.use_rasch and questions is not None:
            mu = self.difficulty(questions)  # (B, L, 1)
            d_c = self.concept_variation(concepts)
            h_r = self.response_variation(responses)
            y = y + mu * (d_c + h_r)
            x = x + mu * d_c

            if self.use_2pl:
                nu = self.log_discrimination(questions)
                alpha = 2.0 * torch.sigmoid(nu)  # in (0, 2), centered at 1.0
                x = alpha * x

        return x, y, mu, alpha


# =============================================================================
# 2. Adaptive Monotonic Multi-Head Attention (with Spacing Effect)
# =============================================================================
class AdaptiveMonotonicAttention(nn.Module):
    r"""Context-aware monotonic multi-head attention with optional concept-repetition
    decay modulation (Ebbinghaus Spacing Effect).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.05,
        monotonic: bool = True,
        adaptive_decay: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.monotonic = monotonic
        self.adaptive_decay = adaptive_decay

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if monotonic:
            self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        else:
            self.gammas = None

    def forward(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        allow_current: bool,
        zero_pad: bool = False,
        concept_reps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = q_in.shape
        H, D_k = self.n_heads, self.d_k

        q = self.q_proj(q_in).view(B, L, H, D_k).transpose(1, 2)
        k = self.k_proj(k_in).view(B, L, H, D_k).transpose(1, 2)
        v = self.v_proj(v_in).view(B, L, H, D_k).transpose(1, 2)

        allowed = causal_mask(L, allow_current=allow_current, device=q_in.device)

        gamma = self.gammas
        if gamma is not None and self.adaptive_decay and concept_reps is not None:
            # concept_reps: (B, L) prior counts -> dampens decay rate for well-practiced items
            damping = (1.0 / (1.0 + torch.log1p(concept_reps.float()))).unsqueeze(1).unsqueeze(-1)
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            neg = -1e32 if scores.dtype == torch.float32 else torch.finfo(scores.dtype).min
            with torch.no_grad():
                s_ = scores.float().masked_fill(~allowed, -1e32).softmax(dim=-1) * allowed
                dist_cum = torch.cumsum(s_, dim=-1)
                dist_total = s_.sum(dim=-1, keepdim=True)
                idx = torch.arange(L, device=scores.device)
                pe = (idx[None, :] - idx[:, None]).abs().to(torch.float32)
                dist = ((dist_total - dist_cum) * pe).clamp_(min=0.0).sqrt_()
            theta = -F.softplus(gamma.float()).unsqueeze(0) * damping
            total_effect = (dist * theta).exp().clamp(min=1e-5, max=1e5)
            scores = (scores * total_effect.to(scores.dtype)).masked_fill(~allowed, neg)
            attn = F.softmax(scores, dim=-1)
            if zero_pad:
                keep = torch.ones(L, 1, device=attn.device, dtype=attn.dtype)
                keep[0] = 0
                attn = attn * keep
            out = torch.matmul(self.dropout(attn), v)
        else:
            out, attn = monotonic_attention(q, k, v, allowed, gamma, self.dropout, zero_pad)

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out), attn


class CogKTTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        dropout: float = 0.05,
        monotonic: bool = True,
        adaptive_decay: bool = False,
    ) -> None:
        super().__init__()
        self.attn = AdaptiveMonotonicAttention(
            d_model, n_heads, dropout=dropout, monotonic=monotonic, adaptive_decay=adaptive_decay
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        allow_current: bool,
        apply_ffn: bool = True,
        zero_pad: bool = False,
        concept_reps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        a, _ = self.attn(q, k, v, allow_current=allow_current, zero_pad=zero_pad, concept_reps=concept_reps)
        x = self.norm1(q + self.drop(a))
        if apply_ffn:
            x = self.norm2(x + self.drop(self.ffn(x)))
        return x


# =============================================================================
# 3. Working Memory Stream (Multi-Scale Causal Convolutions)
# =============================================================================
class CausalConvBlock(nn.Module):
    """Causal depthwise-separable 1D convolution with GLU gating."""

    def __init__(self, channels: int, kernel_size: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.k = kernel_size
        self.conv = nn.Conv1d(channels, 2 * channels, kernel_size=kernel_size)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D) -> transpose to (B, D, L)
        x_t = x.transpose(1, 2)
        # Left-pad by (k - 1) for strict causality (past tokens only)
        padded = F.pad(x_t, (self.k - 1, 0))
        h = self.conv(padded)
        a, b = h.chunk(2, dim=1)
        glu = a * torch.sigmoid(b)
        out = x + self.drop(glu.transpose(1, 2))
        return self.norm(out)


class WorkingMemoryStream(nn.Module):
    """Multi-scale causal working memory tracking recent momentum and trajectory."""

    def __init__(
        self,
        d_model: int,
        kernel_sizes: List[int] = (3, 5),
        n_layers: int = 1,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(*[
                CausalConvBlock(d_model, k, dropout=dropout)
                for _ in range(n_layers)
            ])
            for k in kernel_sizes
        ])
        self.fusion = nn.Linear(len(kernel_sizes) * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, L, D) past interactions
        branch_outs = [b(y) for b in self.branches]
        concat = torch.cat(branch_outs, dim=-1)
        return self.norm(self.fusion(concat))


# =============================================================================
# 4. Cognitive Confidence Gating
# =============================================================================
class CognitiveGate(nn.Module):
    """Learned psychometric gate fusing Working Memory (momentum) and Long-Term Memory (retrieval)."""

    def __init__(self, d_model: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, working_mem: torch.Tensor, long_term_mem: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        # Concatenate working memory, retrieved long-term memory, and target question
        cat = torch.cat([working_mem, long_term_mem, target_q], dim=-1)
        g = self.gate_proj(cat)
        # Fused cognitive state
        return g * working_mem + (1.0 - g) * long_term_mem


# =============================================================================
# 5. Full CogKT Model
# =============================================================================
@dataclass
class CogKTConfig:
    d_model: int = 128
    d_ff: int = 512
    n_heads: int = 8
    n_blocks: int = 1
    dropout: float = 0.05
    # Psychometric IRT toggles
    use_rasch: bool = True
    use_2pl: bool = True
    l2_difficulty: float = 1e-5
    l2_discrimination: float = 1e-5
    # Cognitive Dual-Process toggles
    use_working_memory: bool = True
    conv_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5])
    conv_layers: int = 1
    # Monotonic Attention toggles
    monotonic: bool = True
    adaptive_decay: bool = True
    context_aware: bool = True
    pos_encoding: str = "none"
    # Output Head
    final_fc_dim: int = 512
    final_fc_dim2: int = 256
    use_slip_guess: bool = False


@MODEL_REGISTRY.register("cogkt")
class CogKT(BaseKTModel):
    """Cognitive Dual-Process & 2PL-Psychometric Knowledge Tracing."""

    Config = CogKTConfig
    memory_complexity = 2

    def __init__(self, cfg: CogKTConfig, meta: DatasetMeta) -> None:
        super().__init__(cfg, meta)
        self.n_concepts = meta.n_concepts
        self.has_q = meta.has_questions
        self.is_rasch = cfg.use_rasch and meta.has_questions
        self.is_2pl = cfg.use_2pl and self.is_rasch

        # 1. Psychometric Embeddings
        self.embedding = Psychometric2PLEmbedding(
            meta.n_concepts,
            meta.n_questions,
            cfg.d_model,
            use_rasch=cfg.use_rasch,
            use_2pl=cfg.use_2pl,
        )

        # Positional encodings if requested
        max_len = max(meta.max_seq_len + 1, 512)
        if cfg.pos_encoding == "learnable":
            self.pos_embed: Optional[nn.Module] = LearnablePositionalEmbedding(cfg.d_model, max_len)
        elif cfg.pos_encoding == "cosine":
            self.pos_embed = CosinePositionalEmbedding(cfg.d_model, max_len)
        elif cfg.pos_encoding == "none":
            self.pos_embed = None
        else:
            raise ValueError(f"Unknown pos_encoding '{cfg.pos_encoding}'")

        # 2. Long-Term Memory Stream: Monotonic Encoders + Causal Retriever
        make_layer = lambda: CogKTTransformerLayer(  # noqa: E731
            cfg.d_model,
            cfg.d_ff,
            cfg.n_heads,
            dropout=cfg.dropout,
            monotonic=cfg.monotonic,
            adaptive_decay=cfg.adaptive_decay,
        )
        self.knowledge_encoder = nn.ModuleList(
            [make_layer() for _ in range(cfg.n_blocks)] if cfg.context_aware else []
        )
        n_ret = 2 * cfg.n_blocks if cfg.context_aware else cfg.n_blocks
        self.retriever = nn.ModuleList([make_layer() for _ in range(n_ret)])

        # 3. Working Memory Stream: Multi-Scale Causal Convolutions
        self.use_working_memory = cfg.use_working_memory
        if cfg.use_working_memory:
            self.working_memory = WorkingMemoryStream(
                cfg.d_model,
                kernel_sizes=cfg.conv_kernel_sizes,
                n_layers=cfg.conv_layers,
                dropout=cfg.dropout,
            )
            self.cognitive_gate = CognitiveGate(cfg.d_model, dropout=cfg.dropout)
            self.start_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            nn.init.normal_(self.start_token, std=0.02)

        # 4. Output Head
        self.out = nn.Sequential(
            nn.Linear(2 * cfg.d_model, cfg.final_fc_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.final_fc_dim, cfg.final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.final_fc_dim2, 1),
        )

        # Optional 3PL Slip & Guess bounds
        self.use_slip_guess = cfg.use_slip_guess
        if cfg.use_slip_guess:
            self.slip_param = nn.Parameter(torch.tensor(-2.5))  # sigmoid(-2.5) ~ 0.076 slip
            self.guess_param = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2.0) ~ 0.119 guess

        # Optional Fisher-scoring Rasch calibration buffer
        if self.is_rasch:
            self.register_buffer("question_offset", torch.zeros(meta.n_questions + 1))

    @torch.no_grad()
    def calibrate_rasch(self, loader, device: torch.device) -> None:
        if not self.is_rasch:
            return
        self.question_offset.zero_()
        delta = compute_fisher_rasch_offset(self, loader, self.meta.n_questions, device, tau=6.0)
        scale = 0.90 if self.cfg.context_aware else 0.85
        self.question_offset.copy_(scale * delta)
        self.embedding.difficulty.weight.data.copy_(-delta.unsqueeze(-1))

    def _compute_concept_repetitions(self, concepts: torch.Tensor) -> torch.Tensor:
        """Compute cumulative strictly prior count of each concept in the sequence."""
        B, L = concepts.shape
        device = concepts.device
        valid = (concepts > 0)
        c_t = concepts.unsqueeze(2)
        c_tau = concepts.unsqueeze(1)
        strict_past = causal_mask(L, allow_current=False, device=device).unsqueeze(0)
        match = (c_t == c_tau) & strict_past & valid.unsqueeze(2) & valid.unsqueeze(1)
        return match.sum(dim=-1)  # (B, L)

    def forward(self, batch: Batch) -> ModelOutput:
        B, L = batch.concept_ids.shape
        x, y, mu, alpha = self.embedding(
            batch.concept_ids,
            batch.responses,
            batch.question_ids if self.is_rasch else None,
        )
        x_raw = x

        if self.pos_embed is not None:
            pe = self.pos_embed(x)
            x = x + pe
            y = y + pe

        concept_reps = None
        if self.cfg.adaptive_decay:
            concept_reps = self._compute_concept_repetitions(batch.concept_ids)

        # Stream B: Long-Term Knowledge Retrieval
        for layer in self.knowledge_encoder:
            y = layer(y, y, y, allow_current=True, concept_reps=concept_reps)

        x_ret = x
        if self.cfg.context_aware:
            for i, layer in enumerate(self.retriever):
                if i % 2 == 0:
                    # Question self-attention (may see current question)
                    x_ret = layer(x_ret, x_ret, x_ret, allow_current=True, apply_ffn=False, concept_reps=concept_reps)
                else:
                    # Knowledge retrieval: strictly past interactions
                    x_ret = layer(x_ret, x_ret, y, allow_current=False, apply_ffn=True, zero_pad=True, concept_reps=concept_reps)
        else:
            for layer in self.retriever:
                x_ret = layer(x_ret, x_ret, y, allow_current=False, apply_ffn=True, zero_pad=True, concept_reps=concept_reps)

        # Stream A: Working Memory (Causal Convolutions over strictly past interactions)
        if self.use_working_memory:
            # Shift interactions right by 1 so step t only sees responses < t
            y_past = torch.cat([self.start_token.expand(B, 1, -1), y[:, :-1]], dim=1)
            working_state = self.working_memory(y_past)
            # Cognitive Gating Fusion
            fused_state = self.cognitive_gate(working_state, x_ret, x_raw)
        else:
            fused_state = x_ret

        # Prediction Head
        logits = self.out(torch.cat([fused_state, x_raw], dim=-1)).squeeze(-1)

        # Psychometric 2PL discrimination scaling on logits
        if self.is_2pl and alpha is not None:
            logits = logits * alpha.squeeze(-1)

        # Rasch post-calibration offset if calibrated
        if self.is_rasch and batch.question_ids is not None:
            logits = logits + self.question_offset[batch.question_ids]

        # Regularization losses (empirical Bayes shrinkage)
        reg = 0.0
        if mu is not None and self.cfg.l2_difficulty > 0:
            reg = reg + self.cfg.l2_difficulty * (mu.squeeze(-1).pow(2) * batch.mask).sum()
        if self.is_2pl and hasattr(self.embedding, "log_discrimination") and self.cfg.l2_discrimination > 0:
            nu = self.embedding.log_discrimination(batch.question_ids)
            reg = reg + self.cfg.l2_discrimination * (nu.squeeze(-1).pow(2) * batch.mask).sum()

        extras = {}
        if self.use_slip_guess:
            s = torch.sigmoid(self.slip_param) * 0.25   # max slip 25%
            c = torch.sigmoid(self.guess_param) * 0.35  # max guess 35%
            probs = c + (1.0 - s - c) * torch.sigmoid(logits)
            logits = torch.logit(probs.clamp(1e-6, 1.0 - 1e-6))
            extras["slip"] = s.item()
            extras["guess"] = c.item()

        return ModelOutput(logits=logits, reg_loss=reg, extras=extras)
