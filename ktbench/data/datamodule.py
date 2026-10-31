"""``KTDataModule``: the single entry point from config to DataLoaders."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ktbench.config import DataConfig
from ktbench.data import schema as S
from ktbench.data.adapters import build_adapter
from ktbench.data.dataset import Batch, KTSequenceDataset, LengthBucketBatchSampler, collate_sequences
from ktbench.data.preprocessing import Vocab, assign_splits, build_sequences, encode_ids

log = logging.getLogger(__name__)


class KTDataModule:
    """Orchestrates adapter -> encoding -> split -> windowing -> loaders.

    The module is model-agnostic: it always produces the full canonical
    :class:`Batch`; models simply ignore fields they do not need.
    """

    def __init__(self, cfg: DataConfig, seed: int = 42, interactions: Optional[pd.DataFrame] = None) -> None:
        self.cfg = cfg
        self.seed = seed
        self._interactions = interactions
        self.meta: Optional[S.DatasetMeta] = None
        self.vocabs: Dict[str, Vocab] = {}
        self.datasets: Dict[str, KTSequenceDataset] = {}

    # ------------------------------------------------------------------ setup
    def setup(self) -> "KTDataModule":
        adapter = None
        if self._interactions is None:
            adapter = build_adapter(self.cfg.adapter, self.cfg.adapter_params)
            df = adapter.load()
        else:
            df = S.validate_interactions(self._interactions, source="dataframe")
        df, self.vocabs = encode_ids(df)
        df = assign_splits(df, self.cfg.split, self.seed)
        seqs = build_sequences(df, self.cfg.max_seq_len, self.cfg.window_stride, self.cfg.min_seq_len)
        self.datasets = {k: KTSequenceDataset(v) for k, v in seqs.items()}
        if len(self.datasets["train"]) == 0:
            raise ValueError("Training split is empty")
        self.meta = S.DatasetMeta(
            name=adapter.name if adapter else "dataframe",
            n_concepts=len(self.vocabs[S.CONCEPT]),
            n_questions=len(self.vocabs[S.QUESTION]) if S.QUESTION in self.vocabs else 0,
            has_timestamps=S.TIMESTAMP in df.columns,
            n_students=int(df[S.USER].nunique()),
            n_interactions=int(len(df)),
            max_seq_len=self.cfg.max_seq_len,
        )
        log.info(
            "Data '%s': %d students, %d interactions, %d concepts, %d questions | sequences train/valid/test = %s",
            self.meta.name, self.meta.n_students, self.meta.n_interactions, self.meta.n_concepts,
            self.meta.n_questions, {k: len(v) for k, v in self.datasets.items()},
        )
        return self

    # ---------------------------------------------------------------- loaders
    def _loader(self, split: str, batch_size: int, shuffle: bool) -> DataLoader:
        ds = self.datasets[split]
        common = dict(collate_fn=collate_sequences, num_workers=self.cfg.num_workers,
                      persistent_workers=self.cfg.num_workers > 0)
        if self.cfg.length_bucketing:
            sampler = LengthBucketBatchSampler(ds.lengths, batch_size, shuffle=shuffle, seed=self.seed)
            return DataLoader(ds, batch_sampler=sampler, **common)
        gen = torch.Generator().manual_seed(self.seed)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen, **common)

    def train_dataloader(self, batch_size: int) -> DataLoader:
        return self._loader("train", batch_size, shuffle=True)

    def val_dataloader(self, batch_size: int) -> DataLoader:
        return self._loader("valid", batch_size, shuffle=False)

    def test_dataloader(self, batch_size: int) -> DataLoader:
        return self._loader("test", batch_size, shuffle=False)

    # --------------------------------------------------------------- probing
    def dummy_batch(self, batch_size: int, seq_len: int) -> Batch:
        """A worst-case (fully dense, max-length) random batch for memory probing."""
        assert self.meta is not None, "call setup() first"
        g = torch.Generator().manual_seed(0)
        shape = (batch_size, seq_len)
        has_t = self.meta.has_timestamps
        return Batch(
            concept_ids=torch.randint(1, self.meta.n_concepts + 1, shape, generator=g),
            responses=torch.randint(0, 2, shape, generator=g),
            mask=torch.ones(shape, dtype=torch.bool),
            question_ids=torch.randint(1, self.meta.n_questions + 1, shape, generator=g)
            if self.meta.has_questions else None,
            timestamps=torch.arange(seq_len, dtype=torch.float32).expand(shape).clone() if has_t else None,
        )

    def summary(self) -> Dict[str, object]:
        assert self.meta is not None
        out: Dict[str, object] = self.meta.to_dict()
        for k, ds in self.datasets.items():
            out[f"{k}_sequences"] = len(ds)
            out[f"{k}_interactions"] = ds.num_interactions
        return out
