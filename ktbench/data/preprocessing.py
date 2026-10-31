"""Model-agnostic preprocessing shared by all datasets.

Pipeline: canonical table -> contiguous id encoding -> student-level split ->
per-student chronological sequences -> length-capped windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ktbench.data import schema as S


@dataclass
class Vocab:
    """Maps raw ids <-> contiguous indices ``1..n`` (``0`` = padding)."""

    to_index: Dict = field(default_factory=dict)

    @classmethod
    def fit(cls, values: pd.Series) -> "Vocab":
        uniques = pd.unique(values)
        try:
            uniques = np.sort(uniques)
        except TypeError:  # mixed types: keep first-seen order
            pass
        return cls({v: i + 1 for i, v in enumerate(uniques)})

    def __len__(self) -> int:
        return len(self.to_index)

    def transform(self, values: pd.Series) -> np.ndarray:
        out = values.map(self.to_index)
        if out.isna().any():
            raise KeyError("Unseen ids encountered during transform")
        return out.to_numpy(dtype=np.int64)


def encode_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Vocab]]:
    """Encode concepts / questions over the *whole* dataset.

    Fitting on all splits mirrors the AKT reference (item ids are known
    up-front in an item bank); embeddings of items that only appear in the test
    split simply remain untrained.
    """
    vocabs = {S.CONCEPT: Vocab.fit(df[S.CONCEPT])}
    df = df.copy()
    df[S.CONCEPT] = vocabs[S.CONCEPT].transform(df[S.CONCEPT])
    if S.QUESTION in df.columns:
        vocabs[S.QUESTION] = Vocab.fit(df[S.QUESTION])
        df[S.QUESTION] = vocabs[S.QUESTION].transform(df[S.QUESTION])
    return df, vocabs


def assign_splits(df: pd.DataFrame, fractions: List[float], seed: int) -> pd.DataFrame:
    """Student-level random split (no student appears in two splits).

    If the adapter already provided a ``split`` column it is respected.
    """
    if S.SPLIT in df.columns:
        return df
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"split fractions must be 3 numbers summing to 1, got {fractions}")
    users = pd.unique(df[S.USER])
    rng = np.random.default_rng(seed)
    users = users[rng.permutation(len(users))]
    n_train = int(round(fractions[0] * len(users)))
    n_valid = int(round(fractions[1] * len(users)))
    labels = {}
    for i, u in enumerate(users):
        labels[u] = "train" if i < n_train else ("valid" if i < n_train + n_valid else "test")
    df = df.copy()
    df[S.SPLIT] = df[S.USER].map(labels)
    return df


def build_sequences(
    df: pd.DataFrame,
    max_seq_len: int,
    stride: Optional[int] = None,
    min_seq_len: int = 2,
) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """Group interactions per student (chronologically) and window them.

    Windowing caps the sequence length - the single biggest lever on memory
    for attention models, whose activations grow as ``O(L^2)``. With the
    default ``stride = max_seq_len`` this reproduces the AKT reference split of
    long histories into consecutive chunks.
    """
    stride = stride or max_seq_len
    if stride > max_seq_len:
        raise ValueError("window_stride must be <= max_seq_len")
    sort_cols = [S.USER] + ([S.TIMESTAMP] if S.TIMESTAMP in df.columns else [])
    df = df.sort_values(sort_cols, kind="stable")
    has_q = S.QUESTION in df.columns
    has_t = S.TIMESTAMP in df.columns

    out: Dict[str, List[Dict[str, np.ndarray]]] = {s: [] for s in S.SPLITS}
    for (split, _uid), g in df.groupby([S.SPLIT, S.USER], sort=False):
        c = g[S.CONCEPT].to_numpy(np.int64)
        r = g[S.CORRECT].to_numpy(np.int64)
        q = g[S.QUESTION].to_numpy(np.int64) if has_q else None
        t = g[S.TIMESTAMP].to_numpy(np.float64) if has_t else None
        n = len(c)
        # Overlapping windows are a train-time augmentation only: evaluation
        # must score every interaction exactly once.
        step = stride if split == "train" else max_seq_len
        start = 0
        while start < n:
            end = min(start + max_seq_len, n)
            if end - start >= min_seq_len:
                seq = {"concept_ids": c[start:end], "responses": r[start:end]}
                if q is not None:
                    seq["question_ids"] = q[start:end]
                if t is not None:
                    seq["timestamps"] = t[start:end]
                out[split].append(seq)
            if end == n:
                break
            start += step
    return out
