"""Torch ``Dataset`` / ``Batch`` abstractions.

Sequences are stored *unpadded*; padding happens per batch in
:func:`collate_sequences`, to the longest sequence in that batch only. Combined
with :class:`LengthBucketBatchSampler` this removes most padding waste, which
directly translates into lower peak memory for ``O(L^2)`` attention models.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


@dataclass
class Batch:
    """Right-padded mini-batch; every model consumes exactly this object.

    Shapes are ``(B, L)``. Padding positions have ``mask == False`` and id 0.

    Prediction contract: a model's ``logits[:, t]`` is the prediction of
    ``responses[:, t]`` and may use ``concept_ids/question_ids[:, :t+1]`` but
    only ``responses[:, :t]`` (strictly past responses).
    """

    concept_ids: torch.Tensor
    responses: torch.Tensor
    mask: torch.Tensor
    question_ids: Optional[torch.Tensor] = None
    timestamps: Optional[torch.Tensor] = None

    # ---------------------------------------------------------------- helpers
    @property
    def batch_size(self) -> int:
        return int(self.concept_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.concept_ids.shape[1])

    @property
    def num_tokens(self) -> int:
        return int(self.mask.sum().item())

    def _map(self, fn) -> "Batch":
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kw[f.name] = fn(v) if isinstance(v, torch.Tensor) else v
        return Batch(**kw)

    def to(self, device: torch.device) -> "Batch":
        non_blocking = device.type == "cuda"
        return self._map(lambda t: t.to(device, non_blocking=non_blocking))

    def slice(self, start: int, end: int) -> "Batch":
        """Rows ``start:end``, trimmed to their own max length."""
        sub = self._map(lambda t: t[start:end])
        max_len = int(sub.mask.sum(1).max().item()) if sub.batch_size else 0
        return sub._map(lambda t: t[:, :max_len])

    def sort_by_length(self) -> "Batch":
        order = torch.argsort(self.mask.sum(1), descending=True)
        return self._map(lambda t: t[order])


class KTSequenceDataset(Dataset):
    def __init__(self, sequences: List[Dict[str, np.ndarray]]) -> None:
        self.sequences = sequences
        self.lengths = np.array([len(s["concept_ids"]) for s in sequences], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self.sequences[idx]

    @property
    def num_interactions(self) -> int:
        return int(self.lengths.sum())


def collate_sequences(items: List[Dict[str, np.ndarray]]) -> Batch:
    B = len(items)
    L = max(len(it["concept_ids"]) for it in items)

    def pad(key: str, dtype) -> torch.Tensor:
        out = np.zeros((B, L), dtype=dtype)
        for i, it in enumerate(items):
            v = it[key]
            out[i, : len(v)] = v
        return torch.from_numpy(out)

    mask = np.zeros((B, L), dtype=bool)
    for i, it in enumerate(items):
        mask[i, : len(it["concept_ids"])] = True
    has_q = "question_ids" in items[0]
    has_t = "timestamps" in items[0]
    return Batch(
        concept_ids=pad("concept_ids", np.int64),
        responses=pad("responses", np.int64),
        mask=torch.from_numpy(mask),
        question_ids=pad("question_ids", np.int64) if has_q else None,
        timestamps=pad("timestamps", np.float32) if has_t else None,
    )


class LengthBucketBatchSampler(Sampler[List[int]]):
    """Shuffle -> sort inside large buckets -> chunk -> shuffle chunks.

    Keeps randomness across epochs while ensuring sequences in a batch have
    similar lengths (little padding).
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        bucket_factor: int = 50,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = batch_size * bucket_factor
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        idx = rng.permutation(len(self.lengths)) if self.shuffle else np.arange(len(self.lengths))
        batches: List[List[int]] = []
        for b0 in range(0, len(idx), self.bucket_size):
            bucket = idx[b0 : b0 + self.bucket_size]
            bucket = bucket[np.argsort(-self.lengths[bucket], kind="stable")]
            for i in range(0, len(bucket), self.batch_size):
                chunk = bucket[i : i + self.batch_size].tolist()
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append(chunk)
        if self.shuffle:
            order = rng.permutation(len(batches))
            batches = [batches[i] for i in order]
        return iter(batches)

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size  # approximate upper bound per bucket
        full, rem = divmod(n, self.bucket_size)
        return full * -(-self.bucket_size // self.batch_size) + -(-rem // self.batch_size)
