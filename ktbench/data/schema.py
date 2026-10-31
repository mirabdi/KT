"""Canonical interaction schema shared by every dataset adapter and model.

Every dataset - whatever its on-disk format - is standardised by an adapter
into a long-format :class:`pandas.DataFrame` with these columns:

==============  ========  ========================================================
column          required  meaning
==============  ========  ========================================================
``user_id``     yes       student identifier (any hashable)
``concept_id``  yes       knowledge component / skill (AKT's ``c_t``)
``correct``     yes       binary response ``r_t`` in {0, 1}
``question_id`` no        individual item / problem (AKT's ``q_t``; enables Rasch)
``timestamp``   no        ordering key (seconds, or any monotonic number)
``split``       no        pre-defined split: ``train`` / ``valid`` / ``test``
==============  ========  ========================================================

Multi-concept items are represented as one row per (interaction, concept), the
same convention used by the AKT / DKVMN preprocessed ASSISTments files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

USER = "user_id"
CONCEPT = "concept_id"
QUESTION = "question_id"
CORRECT = "correct"
TIMESTAMP = "timestamp"
SPLIT = "split"

REQUIRED_COLUMNS = (USER, CONCEPT, CORRECT)
OPTIONAL_COLUMNS = (QUESTION, TIMESTAMP, SPLIT)
SPLITS = ("train", "valid", "test")

PAD_INDEX = 0
"""Index 0 is reserved for padding in every id vocabulary."""


@dataclass
class DatasetMeta:
    """Static facts about a standardised dataset, handed to every model."""

    name: str
    n_concepts: int
    """Number of distinct concepts. Valid ids are ``1..n_concepts`` (0 = pad)."""
    n_questions: int
    """Number of distinct questions (``0`` if the dataset has no question ids)."""
    has_timestamps: bool
    n_students: int
    n_interactions: int
    max_seq_len: int

    @property
    def has_questions(self) -> bool:
        return self.n_questions > 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["has_questions"] = self.has_questions
        return d


def validate_interactions(df: pd.DataFrame, source: str = "dataset") -> pd.DataFrame:
    """Check the canonical contract and coerce dtypes. Raises ``ValueError``."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: missing required columns {missing}; got {list(df.columns)}")
    keep = [c for c in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS) if c in df.columns]
    df = df[keep].copy()
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    correct = pd.to_numeric(df[CORRECT], errors="coerce")
    if correct.isna().any():
        raise ValueError(f"{source}: '{CORRECT}' contains non-numeric values")
    # Partial credit (e.g. 0.5) is binarised at 0.5, the common KT convention.
    df[CORRECT] = (correct >= 0.5).astype("int64")
    if SPLIT in df.columns:
        bad = set(df[SPLIT].unique()) - set(SPLITS)
        if bad:
            raise ValueError(f"{source}: unknown split labels {bad}; expected {SPLITS}")
    if QUESTION in df.columns:
        df = df.dropna(subset=[QUESTION])
    if len(df) == 0:
        raise ValueError(f"{source}: no interactions left after validation")
    return df.reset_index(drop=True)
