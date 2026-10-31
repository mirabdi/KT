"""Model contract, causality (no future response leakage), padding invariance,
and plug-in extensibility tests across all registered KT models."""

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from ktbench.data.dataset import Batch
from ktbench.data.schema import DatasetMeta
from ktbench.models import MODEL_REGISTRY, BaseKTModel, ModelOutput, build_model
from tests.conftest import SMALL_PARAMS


@pytest.mark.parametrize("model_name", MODEL_REGISTRY.names())
def test_forward_backward_and_causality(model_name: str, dm, device):
    """Every model must:
    1. return logits of shape (B, L),
    2. strictly respect causality: changing responses[:, t:] must NOT alter logits[:, :t+1],
    3. propagate gradients to parameters.
    """
    model = build_model(model_name, SMALL_PARAMS[model_name], dm.meta).to(device)
    model.eval()

    batch = next(iter(dm.train_dataloader(4))).to(device)
    out1 = model(batch)
    assert out1.logits.shape == (batch.batch_size, batch.seq_len)

    # Corrupt responses from step t_cut onwards (including t_cut itself!)
    t_cut = batch.seq_len // 2
    flipped_r = batch.responses.clone()
    flipped_r[:, t_cut:] = 1 - flipped_r[:, t_cut:]
    batch2 = Batch(
        concept_ids=batch.concept_ids,
        responses=flipped_r,
        mask=batch.mask,
        question_ids=batch.question_ids,
        timestamps=batch.timestamps,
    )
    out2 = model(batch2)
    # Prediction at t_cut uses responses[:t_cut] only, so logits[:, :t_cut + 1] must match!
    torch.testing.assert_close(
        out1.logits[:, : t_cut + 1], out2.logits[:, : t_cut + 1], rtol=1e-4, atol=1e-5
    )

    # Gradient check
    model.train()
    out = model(batch)
    loss = (out.logits * batch.mask).sum() + out.reg_loss
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("model_name", MODEL_REGISTRY.names())
def test_right_padding_invariance(model_name: str, dm, device):
    """Appending right-padded columns must not change logits on valid positions."""
    model = build_model(model_name, SMALL_PARAMS[model_name], dm.meta).to(device)
    model.eval()

    b_raw = next(iter(dm.val_dataloader(4)))
    # Trim to 40 tokens first so adding pad_cols stays within max_seq_len=60
    b = b_raw._map(lambda t: t[:, :40]).to(device)
    logits_unpadded = model(b).logits

    pad_cols = 7
    B = b.batch_size
    z_long = torch.zeros(B, pad_cols, dtype=torch.long, device=device)
    z_bool = torch.zeros(B, pad_cols, dtype=torch.bool, device=device)
    b_padded = Batch(
        concept_ids=torch.cat([b.concept_ids, z_long], dim=1),
        responses=torch.cat([b.responses, z_long], dim=1),
        mask=torch.cat([b.mask, z_bool], dim=1),
        question_ids=torch.cat([b.question_ids, z_long], dim=1) if b.question_ids is not None else None,
    )
    logits_padded = model(b_padded).logits[:, : b.seq_len]
    torch.testing.assert_close(logits_unpadded, logits_padded, rtol=1e-4, atol=1e-5)


def test_akt_falls_back_to_non_rasch_without_questions(dm_no_q, device):
    model = build_model("akt", SMALL_PARAMS["akt"], dm_no_q.meta).to(device)
    assert not model.is_rasch
    assert model.question_difficulties() is None
    b = next(iter(dm_no_q.train_dataloader(4))).to(device)
    out = model(b)
    assert out.logits.shape == (b.batch_size, b.seq_len)
    assert out.reg_loss == 0.0


def test_custom_model_plugin_registration(dm, device):
    """Verify a researcher can register and run a brand-new model in <25 lines
    without touching any framework internals."""

    @dataclass
    class IRTBaselineConfig:
        d: int = 16

    @MODEL_REGISTRY.register("irt_test_plugin")
    class IRTBaseline(BaseKTModel):
        Config = IRTBaselineConfig

        def __init__(self, cfg: IRTBaselineConfig, meta: DatasetMeta) -> None:
            super().__init__(cfg, meta)
            self.item_ease = nn.Embedding(meta.n_concepts + 1, 1)

        def forward(self, batch: Batch) -> ModelOutput:
            # Predict from running historical accuracy minus concept difficulty
            r_float = batch.responses.float()
            cum_correct = torch.cumsum(r_float, dim=1) - r_float
            steps = torch.arange(1, batch.seq_len + 1, device=r_float.device).float()
            running_rate = cum_correct / steps
            logits = running_rate + self.item_ease(batch.concept_ids).squeeze(-1)
            return ModelOutput(logits=logits)

    try:
        m = build_model("irt_test_plugin", {"d": 8}, dm.meta).to(device)
        b = next(iter(dm.train_dataloader(4))).to(device)
        out = m(b)
        assert out.logits.shape == (b.batch_size, b.seq_len)
    finally:
        MODEL_REGISTRY._items.pop("irt_test_plugin", None)
