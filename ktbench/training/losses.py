"""Loss functions. Everything is a *sum* over valid tokens; the trainer divides
by the token count of the full logical batch, which makes gradient
accumulation over arbitrary micro-batch splits exactly equivalent to a single
large batch."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ktbench.data.dataset import Batch
from ktbench.models.base import ModelOutput


def masked_bce_sum(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    return (loss * mask).sum()


def kt_loss_sum(output: ModelOutput, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(total_sum, bce_sum)`` where total includes model regularisers."""
    bce = masked_bce_sum(output.logits, batch.responses, batch.mask)
    aux = output.extras.get("aux_loss", 0.0) if output.extras else 0.0
    return bce + output.reg_loss + aux, bce
