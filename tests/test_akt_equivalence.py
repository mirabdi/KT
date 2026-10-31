"""Numerical equivalence against the original KDD 2020 AKT implementation
(https://github.com/arghosh/AKT) plus unit tests for AKT's three core
innovations:

1. Rasch-model (1PL IRT) question & interaction embeddings
2. Context-aware monotonic attention (exponential decay with contextual distance)
3. Dual-encoder / causal knowledge-retriever architecture
"""

import pytest
import torch

from ktbench.data.dataset import Batch
from ktbench.data.schema import DatasetMeta
from ktbench.models.akt import AKT, AKTConfig, RaschEmbedding, causal_mask, monotonic_attention
from ktbench.training.losses import kt_loss_sum
from tests.reference.akt_legacy import AKT as LegacyAKT
from tests.reference.akt_legacy import attention as legacy_attention


def _copy_weights_from_legacy(modern: AKT, legacy: LegacyAKT) -> None:
    """Map every parameter of ``LegacyAKT`` onto ``modern`` 1-for-1."""
    with torch.no_grad():
        modern.embedding.concept_embed.weight.copy_(legacy.q_embed.weight)
        modern.embedding.response_embed.weight.copy_(legacy.qa_embed.weight)
        if modern.is_rasch:
            modern.embedding.difficulty.weight.copy_(legacy.difficult_param.weight)
            modern.embedding.concept_variation.weight.copy_(legacy.q_embed_diff.weight)
            n_rows = modern.embedding.interaction_variation.num_embeddings
            modern.embedding.interaction_variation.weight.copy_(legacy.qa_embed_diff.weight[:n_rows])

        def copy_block(dst_blk, src_blk):
            dst_blk.attn.v_linear.load_state_dict(src_blk.masked_attn_head.v_linear.state_dict())
            dst_blk.attn.k_linear.load_state_dict(src_blk.masked_attn_head.k_linear.state_dict())
            if not dst_blk.attn.kq_same:
                dst_blk.attn.q_linear.load_state_dict(src_blk.masked_attn_head.q_linear.state_dict())
            dst_blk.attn.out_proj.load_state_dict(src_blk.masked_attn_head.out_proj.state_dict())
            dst_blk.attn.gammas.copy_(src_blk.masked_attn_head.gammas)
            dst_blk.norm1.load_state_dict(src_blk.layer_norm1.state_dict())
            dst_blk.ffn[0].load_state_dict(src_blk.linear1.state_dict())
            dst_blk.ffn[3].load_state_dict(src_blk.linear2.state_dict())
            dst_blk.norm2.load_state_dict(src_blk.layer_norm2.state_dict())

        for d, s in zip(modern.knowledge_encoder, legacy.model.blocks_1):
            copy_block(d, s)
        for d, s in zip(modern.retriever, legacy.model.blocks_2):
            copy_block(d, s)

        for i in (0, 3, 6):
            modern.out[i].load_state_dict(legacy.out[i].state_dict())


@pytest.mark.parametrize("use_rasch", [True, False])
@pytest.mark.parametrize("separate_qa", [False, True])
@pytest.mark.parametrize("kq_same", [True, False])
def test_forward_and_backward_match_legacy_akt(use_rasch: bool, separate_qa: bool, kq_same: bool):
    """Given identical weights and unpadded inputs, modern AKT produces
    the exact same probabilities, loss, and parameter gradients as the
    original KDD 2020 codebase."""
    torch.manual_seed(7)
    n_c, n_q, B, L = 15, (40 if use_rasch else 0), 4, 16
    d_model, n_blocks, n_heads, d_ff = 32, 1, 4, 64

    legacy = LegacyAKT(
        n_question=n_c,
        n_pid=n_q if use_rasch else -1,
        d_model=d_model,
        n_blocks=n_blocks,
        kq_same=int(kq_same),
        dropout=0.0,
        model_type="akt",
        final_fc_dim=32,
        n_heads=n_heads,
        d_ff=d_ff,
        l2=1e-5,
        separate_qa=separate_qa,
    )
    # Randomize difficulty_param too so the Rasch path & L2 loss are non-trivial
    if use_rasch:
        torch.nn.init.normal_(legacy.difficult_param.weight, std=0.5)
    legacy.eval()

    meta = DatasetMeta(
        name="oracle", n_concepts=n_c, n_questions=n_q, has_timestamps=False,
        n_students=B, n_interactions=B * L, max_seq_len=L,
    )
    cfg = AKTConfig(
        d_model=d_model, n_blocks=n_blocks, n_heads=n_heads, d_ff=d_ff,
        dropout=0.0, kq_same=kq_same, final_fc_dim=32, final_fc_dim2=256,
        separate_qa=separate_qa, use_rasch=use_rasch, l2=1e-5,
    )
    modern = AKT(cfg, meta)
    _copy_weights_from_legacy(modern, legacy)
    modern.eval()

    concepts = torch.randint(1, n_c + 1, (B, L))
    responses = torch.randint(0, 2, (B, L))
    questions = torch.randint(1, n_q + 1, (B, L)) if use_rasch else None
    qa_legacy = concepts + responses * n_c

    leg_loss, leg_preds, _ = legacy(concepts, qa_legacy, responses.float(), questions)
    leg_loss.backward()

    batch = Batch(
        concept_ids=concepts,
        responses=responses,
        mask=torch.ones(B, L, dtype=torch.bool),
        question_ids=questions,
    )
    out = modern(batch)
    mod_loss, _ = kt_loss_sum(out, batch)
    mod_loss.backward()

    mod_preds = torch.sigmoid(out.logits).reshape(-1)
    torch.testing.assert_close(mod_preds, leg_preds, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(mod_loss, leg_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        modern.embedding.concept_embed.weight.grad,
        legacy.q_embed.weight.grad,
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("allow_current", [True, False])
def test_monotonic_attention_matches_legacy_kernel(allow_current: bool):
    torch.manual_seed(11)
    B, H, L, D = 3, 4, 12, 8
    q = torch.randn(B, H, L, D)
    k = torch.randn(B, H, L, D)
    v = torch.randn(B, H, L, D)
    gamma = torch.randn(H, 1, 1)

    allowed = causal_mask(L, allow_current=allow_current, device=torch.device("cpu"))
    out_mod, attn = monotonic_attention(
        q, k, v, allowed, gamma, torch.nn.Identity(), zero_pad=not allow_current
    )

    import numpy as np

    nopeek = np.triu(np.ones((1, 1, L, L)), k=int(allow_current)).astype("uint8")
    src_mask = torch.from_numpy(nopeek) == 0
    out_leg = legacy_attention(
        q, k, v, D, src_mask, torch.nn.Identity(), zero_pad=not allow_current, gamma=gamma
    )
    torch.testing.assert_close(out_mod, out_leg, rtol=1e-5, atol=1e-5)

    if not allow_current:
        # First query has no past; zero_pad must zero out row 0 completely.
        assert torch.all(attn[:, :, 0, :] == 0)
        assert torch.all(out_mod[:, :, 0, :] == 0)


def test_rasch_embedding_equations():
    """Verify x_t = c_{c_t} + mu_{q_t} * d_{c_t} and difficulty init = 0."""
    emb = RaschEmbedding(n_concepts=5, n_questions=10, d_model=8, separate_qa=False, use_rasch=True)
    c = torch.tensor([[1, 2]])
    r = torch.tensor([[0, 1]])
    q = torch.tensor([[3, 7]])

    # At initialization mu_q == 0, so Rasch embedding equals concept embedding
    x0, y0, mu0 = emb(c, r, q)
    assert torch.all(mu0 == 0)
    torch.testing.assert_close(x0, emb.concept_embed(c))
    torch.testing.assert_close(y0, emb.concept_embed(c) + emb.response_embed(r))

    # With non-zero mu_q, verify exact linear perturbation along variation vectors
    with torch.no_grad():
        emb.difficulty.weight.fill_(2.5)
    x1, y1, _ = emb(c, r, q)
    torch.testing.assert_close(x1, emb.concept_embed(c) + 2.5 * emb.concept_variation(c))
    torch.testing.assert_close(
        y1,
        emb.concept_embed(c) + emb.response_embed(r) + 2.5 * (emb.interaction_variation(r) + emb.concept_variation(c)),
    )
