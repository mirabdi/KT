"""Dataset adapters: raw source  ->  canonical interaction table.

An adapter's *only* job is ingestion + column standardisation. Everything
downstream (id encoding, splitting, windowing, batching) is shared and lives
in :mod:`ktbench.data.preprocessing` / :mod:`ktbench.data.datamodule`, which
guarantees every model sees identically processed data.

To add a dataset, subclass :class:`BaseDatasetAdapter`, implement
:meth:`load_raw`, and decorate it with ``@ADAPTER_REGISTRY.register("name")``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ktbench.data import schema as S
from ktbench.data.synthetic import SyntheticConfig, generate_interactions
from ktbench.registry import Registry

ADAPTER_REGISTRY: Registry = Registry("dataset adapter")


class BaseDatasetAdapter(ABC):
    """Template-method base: subclasses implement :meth:`load_raw`; :meth:`load`
    validates the output against the canonical schema."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or getattr(self, "registry_name", type(self).__name__)

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """Return a DataFrame that already uses the canonical column names."""

    def load(self) -> pd.DataFrame:
        return S.validate_interactions(self.load_raw(), source=self.name)


# ---------------------------------------------------------------------------
@ADAPTER_REGISTRY.register("csv")
class CSVInteractionAdapter(BaseDatasetAdapter):
    """Generic long-format CSV/TSV/Parquet adapter with a column mapping.

    Example (ASSISTments 2009 skill-builder raw export)::

        adapter: csv
        adapter_params:
          path: data/assist2009/skill_builder_data.csv
          columns: {user_id: user_id, question_id: problem_id,
                    concept_id: skill_id, correct: correct, timestamp: order_id}
          read_kwargs: {encoding: latin-1}
    """

    def __init__(
        self,
        path: str,
        columns: Optional[Dict[str, str]] = None,
        sep: str = ",",
        read_kwargs: Optional[dict] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or Path(path).stem)
        self.path = Path(path)
        self.columns = columns or {}
        self.sep = sep
        self.read_kwargs = read_kwargs or {}

    def load_raw(self) -> pd.DataFrame:
        if self.path.suffix == ".parquet":
            df = pd.read_parquet(self.path)
        else:
            df = pd.read_csv(self.path, sep=self.sep, **self.read_kwargs)
        # mapping is canonical -> source; invert for rename
        rename = {src: canon for canon, src in self.columns.items()}
        return df.rename(columns=rename)


# ---------------------------------------------------------------------------
@ADAPTER_REGISTRY.register("akt_legacy")
class AKTLegacyAdapter(BaseDatasetAdapter):
    """Reads the line-based files shipped with the original AKT/DKVMN repos.

    ``files`` maps split name -> path, e.g.::

        adapter: akt_legacy
        adapter_params:
          files: {train: data/assist2009_pid/assist2009_pid_train1.csv,
                  valid: data/assist2009_pid/assist2009_pid_valid1.csv,
                  test:  data/assist2009_pid/assist2009_pid_test1.csv}
          with_questions: true    # 4-line "_pid" format, else 3-line

    The pre-defined split is preserved through the ``split`` column. Line
    order is used as time order (no timestamps in these files).
    """

    def __init__(
        self,
        files: Optional[Dict[str, str]] = None,
        with_questions: Optional[bool] = None,
        name: Optional[str] = None,
        data_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        fold: int = 1,
    ) -> None:
        if files is None:
            if data_dir is None:
                raise ValueError("AKTLegacyAdapter requires either 'files' or 'data_dir'")
            dpath = Path(data_dir)
            ds = dataset_name or dpath.name
            files = {
                "train": str(dpath / f"{ds}_train{fold}.csv"),
                "valid": str(dpath / f"{ds}_valid{fold}.csv"),
                "test": str(dpath / f"{ds}_test{fold}.csv"),
            }
            if with_questions is None:
                with_questions = "pid" in ds
        super().__init__(name or dataset_name or "akt_legacy")
        self.files = files
        self.with_questions = True if with_questions is None else with_questions

    def _parse(self, path: str, split: str) -> pd.DataFrame:
        stride = 4 if self.with_questions else 3
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if len(lines) % stride:
            raise ValueError(f"{path}: line count {len(lines)} not divisible by {stride}")

        def ints(s: str) -> list[int]:
            return [int(x) if x.lstrip("-").isdigit() else int(float(x)) for x in s.split(",") if x != ""]

        users: list[str] = []
        concepts: list[int] = []
        responses: list[int] = []
        timestamps: list[int] = []
        questions: list[int] = []

        for i in range(0, len(lines), stride):
            block = lines[i : i + stride]
            header = block[0].split(",")
            uid = f"{split}_{i // stride:06d}_{header[1] if len(header) > 1 else ''}"
            if self.with_questions:
                q, c, r = ints(block[1]), ints(block[2]), ints(block[3])
            else:
                c, r = ints(block[1]), ints(block[2])
                q = None
            n = len(c)
            users.extend([uid] * n)
            concepts.extend(c)
            responses.extend(r)
            timestamps.extend(range(n))
            if q is not None:
                questions.extend(q)

        data = {
            S.USER: users,
            S.CONCEPT: np.asarray(concepts, dtype=np.int64),
            S.CORRECT: np.asarray(responses, dtype=np.int64),
            S.TIMESTAMP: np.asarray(timestamps, dtype=np.int64),
            S.SPLIT: split,
        }
        if self.with_questions:
            data[S.QUESTION] = np.asarray(questions, dtype=np.int64)
        return pd.DataFrame(data)

    def load_raw(self) -> pd.DataFrame:
        frames = [self._parse(path, split) for split, path in self.files.items()]
        return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
@ADAPTER_REGISTRY.register("pykt")
class PyKTAdapter(BaseDatasetAdapter):
    """Adapter for pyKT sequence-format CSV files (train_valid.csv and test.csv).

    Expected columns in both files:
      - uid: student identifier
      - questions: comma-separated question IDs (if with_questions)
      - concepts: comma-separated concept IDs
      - responses: comma-separated binary responses (0 or 1)
      - timestamps: comma-separated timestamps (optional)
      - fold: cross-validation fold (0..4 for train_valid, -1 for test)

    Parameters:
      data_dir: Path to directory containing train_valid.csv and test.csv.
      dataset_name: Optional name for the dataset.
      fold: Fold index used for validation. Accepts 0..4 or 1..5.
      with_questions: Whether to parse question IDs (problem IDs).
      files: Optional dict mapping 'train_valid' and 'test' to explicit file paths.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        fold: int = 1,
        with_questions: bool = True,
        files: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or dataset_name or (Path(data_dir).name if data_dir else "pykt"))
        if files is None:
            if data_dir is None:
                raise ValueError("PyKTAdapter requires either 'files' or 'data_dir'")
            dpath = Path(data_dir)
            files = {
                "train_valid": str(dpath / "train_valid.csv"),
                "test": str(dpath / "test.csv"),
            }
        self.files = files
        self.fold = fold
        self.with_questions = with_questions

    def load_raw(self) -> pd.DataFrame:
        tr = pd.read_csv(self.files["train_valid"])
        te = pd.read_csv(self.files["test"])

        val_fold_idx = self.fold - 1 if (self.fold in (1, 2, 3, 4, 5) and (tr["fold"] == 0).any()) else self.fold

        users: list[str] = []
        concepts: list[int] = []
        questions: list[int] = []
        responses: list[int] = []
        timestamps: list[int] = []
        splits: list[str] = []

        has_ts = "timestamps" in tr.columns and "timestamps" in te.columns

        for row in tr.itertuples(index=False):
            u = f"tr_{row.uid}"
            sp = "valid" if int(row.fold) == val_fold_idx else "train"
            c_list = [int(x) for x in str(row.concepts).split(",") if x != ""]
            r_list = [int(x) for x in str(row.responses).split(",") if x != ""]
            n = len(r_list)
            users.extend([u] * n)
            concepts.extend(c_list)
            responses.extend(r_list)
            splits.extend([sp] * n)
            if has_ts and pd.notna(getattr(row, "timestamps", None)):
                t_list = [int(x) for x in str(row.timestamps).split(",") if x != ""]
                timestamps.extend(t_list)
            else:
                timestamps.extend(range(n))
            if self.with_questions:
                q_list = [int(x) for x in str(row.questions).split(",") if x != ""]
                questions.extend(q_list)

        for row in te.itertuples(index=False):
            u = f"te_{row.uid}"
            c_list = [int(x) for x in str(row.concepts).split(",") if x != ""]
            r_list = [int(x) for x in str(row.responses).split(",") if x != ""]
            n = len(r_list)
            users.extend([u] * n)
            concepts.extend(c_list)
            responses.extend(r_list)
            splits.extend(["test"] * n)
            if has_ts and pd.notna(getattr(row, "timestamps", None)):
                t_list = [int(x) for x in str(row.timestamps).split(",") if x != ""]
                timestamps.extend(t_list)
            else:
                timestamps.extend(range(n))
            if self.with_questions:
                q_list = [int(x) for x in str(row.questions).split(",") if x != ""]
                questions.extend(q_list)

        data = {
            S.USER: users,
            S.CONCEPT: np.asarray(concepts, dtype=np.int64),
            S.CORRECT: np.asarray(responses, dtype=np.int64),
            S.TIMESTAMP: np.asarray(timestamps, dtype=np.int64),
            S.SPLIT: splits,
        }
        if self.with_questions:
            data[S.QUESTION] = np.asarray(questions, dtype=np.int64)
        return pd.DataFrame(data)


# ---------------------------------------------------------------------------
@ADAPTER_REGISTRY.register("synthetic")
class SyntheticAdapter(BaseDatasetAdapter):
    """In-memory simulated dataset (see :mod:`ktbench.data.synthetic`).

    ``use_questions: false`` drops question ids to emulate concept-only
    datasets (e.g. ASSISTments 2015), exercising the non-Rasch code paths.
    """

    def __init__(self, use_questions: bool = True, name: Optional[str] = None, **params) -> None:
        super().__init__(name or "synthetic")
        self.cfg = SyntheticConfig(**params)
        self.use_questions = use_questions

    def load_raw(self) -> pd.DataFrame:
        df = generate_interactions(self.cfg)
        if not self.use_questions:
            df = df.drop(columns=[S.QUESTION])
        return df


def build_adapter(name: str, params: Optional[dict] = None) -> BaseDatasetAdapter:
    return ADAPTER_REGISTRY.get(name)(**(params or {}))
