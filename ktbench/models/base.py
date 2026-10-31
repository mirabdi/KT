"""The single contract every Knowledge Tracing model implements."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Union

import torch
from torch import nn

from ktbench.config import build_dataclass
from ktbench.data.dataset import Batch
from ktbench.data.schema import DatasetMeta
from ktbench.registry import Registry

MODEL_REGISTRY: Registry = Registry("model")


@dataclass
class ModelOutput:
    logits: torch.Tensor
    """``(B, L)`` pre-sigmoid scores; ``logits[:, t]`` predicts ``responses[:, t]``."""
    reg_loss: Union[torch.Tensor, float] = 0.0
    """Model-specific regulariser, expressed as a *sum* (same units as summed
    BCE), so the trainer can normalise everything by the token count."""
    extras: Dict[str, Any] = field(default_factory=dict)
    """Optional diagnostics (attention maps, hidden states...)."""


@dataclass
class EmptyConfig:
    pass


class BaseKTModel(nn.Module, ABC):
    """Abstract base for all KT models.

    Subclasses must:

    1. define a nested ``Config`` dataclass with their hyper-parameters
       (validated automatically - unknown YAML keys raise errors);
    2. implement :meth:`forward` returning a :class:`ModelOutput` that honours
       the causal prediction contract documented on :class:`Batch`;
    3. register themselves with ``@MODEL_REGISTRY.register("name")``.

    Optional class attributes inform the pipeline:

    * ``memory_complexity`` - exponent ``p`` in the activation-cost model
      ``cost = B * L**p`` used by the memory manager (1 = RNN, 2 = attention).
    * ``requires_questions`` - refuse datasets without question ids.
    """

    Config: ClassVar[type] = EmptyConfig
    memory_complexity: ClassVar[int] = 1
    requires_questions: ClassVar[bool] = False

    def __init__(self, cfg: Any, meta: DatasetMeta) -> None:
        super().__init__()
        if self.requires_questions and not meta.has_questions:
            raise ValueError(f"{type(self).__name__} requires question ids, dataset '{meta.name}' has none")
        self.cfg = cfg
        self.meta = meta

    # ------------------------------------------------------------ factories
    @classmethod
    def from_params(cls, params: Optional[Dict[str, Any]], meta: DatasetMeta) -> "BaseKTModel":
        cfg = build_dataclass(cls.Config, params or {}, owner=cls.__name__)
        return cls(cfg, meta)

    # ---------------------------------------------------------------- api
    @abstractmethod
    def forward(self, batch: Batch) -> ModelOutput:  # pragma: no cover - abstract
        ...

    @torch.no_grad()
    def predict_proba(self, batch: Batch) -> torch.Tensor:
        return torch.sigmoid(self.forward(batch).logits)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def parameter_groups(self, base_lr: float) -> Any:
        return self.parameters()

    @torch.no_grad()
    def calibrate_rasch(self, loader: Any, device: torch.device) -> None:
        """Optional post-epoch 1PL IRT / Rasch question-difficulty calibration."""
        return None

    def hparams(self) -> Dict[str, Any]:
        return dataclasses.asdict(self.cfg) if dataclasses.is_dataclass(self.cfg) else {}


@torch.no_grad()
def compute_fisher_rasch_offset(
    model: BaseKTModel,
    loader: Any,
    n_questions: int,
    device: torch.device,
    tau: float = 6.0,
) -> torch.Tensor:
    """One-step conditional 1PL Rasch / Fisher-scoring difficulty offset per question."""
    was_training = model.training
    model.eval()
    num = torch.zeros(n_questions + 1, device=device)
    den = torch.zeros(n_questions + 1, device=device)
    for b in loader:
        b = b.to(device)
        if b.question_ids is None:
            continue
        logits = model(b).logits
        p = torch.sigmoid(logits)
        m = b.mask & (b.question_ids > 0)
        q = b.question_ids[m]
        r = b.responses[m].float()
        pm = p[m]
        num.scatter_add_(0, q, r - pm)
        den.scatter_add_(0, q, pm * (1.0 - pm))
    if was_training:
        model.train()
    delta = num / (den + tau)
    delta[0] = 0.0
    return delta


def build_model(name: str, params: Optional[Dict[str, Any]], meta: DatasetMeta) -> BaseKTModel:
    cls = MODEL_REGISTRY.get(name)
    return cls.from_params(params, meta)


def interaction_index(concepts: torch.Tensor, responses: torch.Tensor, n_concepts: int) -> torch.Tensor:
    """Joint (concept, response) index ``c + r * n_concepts``; padding (c=0) -> 0
    when r = 0. Mirrors the ``qa`` encoding of the AKT/DKVMN code bases."""
    return concepts + responses * n_concepts * (concepts > 0).long()
