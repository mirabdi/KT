"""Data layer: schema, synthetic generator, adapters, preprocessing, batching."""

import numpy as np
import pandas as pd
import pytest
import torch

from ktbench.data import schema as S
from ktbench.data.adapters import AKTLegacyAdapter, CSVInteractionAdapter, build_adapter
from ktbench.data.dataset import LengthBucketBatchSampler, collate_sequences
from ktbench.data.preprocessing import assign_splits, build_sequences, encode_ids
from ktbench.data.synthetic import SyntheticConfig, generate_interactions, write_akt_legacy

from tests.conftest import make_dm


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return generate_interactions(SyntheticConfig(n_students=60, n_concepts=8, n_questions=40, seed=3))


def test_synthetic_schema_and_determinism(raw):
    assert list(raw.columns) == [S.USER, S.QUESTION, S.CONCEPT, S.CORRECT, S.TIMESTAMP]
    assert set(raw[S.CORRECT].unique()) <= {0, 1}
    assert raw[S.USER].nunique() == 60
    # every question belongs to exactly one concept (Rasch item bank)
    assert (raw.groupby(S.QUESTION)[S.CONCEPT].nunique() == 1).all()
    # timestamps strictly increase within each student
    assert raw.groupby(S.USER)[S.TIMESTAMP].apply(lambda t: t.is_monotonic_increasing).all()
    again = generate_interactions(SyntheticConfig(n_students=60, n_concepts=8, n_questions=40, seed=3))
    pd.testing.assert_frame_equal(raw, again)
    # non-degenerate labels
    assert 0.2 < raw[S.CORRECT].mean() < 0.9


def test_validate_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required"):
        S.validate_interactions(pd.DataFrame({"user_id": [1], "correct": [1]}))


def test_encode_ids_reserve_zero(raw):
    enc, vocabs = encode_ids(raw)
    for col in (S.CONCEPT, S.QUESTION):
        ids = np.sort(enc[col].unique())
        assert ids.min() == 1 and ids.max() == len(vocabs[col])
        np.testing.assert_array_equal(ids, np.arange(1, len(vocabs[col]) + 1))


def test_student_level_split_has_no_leakage(raw):
    df = assign_splits(raw, [0.6, 0.2, 0.2], seed=0)
    users = {s: set(df[df[S.SPLIT] == s][S.USER]) for s in S.SPLITS}
    assert not (users["train"] & users["valid"]) and not (users["train"] & users["test"])
    assert not (users["valid"] & users["test"])
    assert sum(map(len, users.values())) == raw[S.USER].nunique()


def test_windowing_caps_length_and_preserves_interactions(raw):
    enc, _ = encode_ids(raw)
    df = assign_splits(enc, [0.6, 0.2, 0.2], seed=0)
    seqs = build_sequences(df, max_seq_len=50, min_seq_len=1)
    total = sum(len(s["concept_ids"]) for split in seqs.values() for s in split)
    assert total == len(df)
    assert max(len(s["concept_ids"]) for split in seqs.values() for s in split) <= 50
    # overlapping stride is applied to train only
    seqs_ov = build_sequences(df, max_seq_len=50, stride=25, min_seq_len=1)
    assert len(seqs_ov["train"]) > len(seqs["train"])
    assert len(seqs_ov["test"]) == len(seqs["test"])


def test_collate_pads_right_with_mask():
    items = [
        {"concept_ids": np.array([1, 2, 3]), "responses": np.array([1, 0, 1]), "question_ids": np.array([4, 5, 6])},
        {"concept_ids": np.array([7]), "responses": np.array([1]), "question_ids": np.array([9])},
    ]
    b = collate_sequences(items)
    assert b.concept_ids.shape == (2, 3)
    assert b.mask.tolist() == [[True, True, True], [True, False, False]]
    assert b.concept_ids[1].tolist() == [7, 0, 0] and b.num_tokens == 4
    sub = b.slice(1, 2)
    assert sub.seq_len == 1 and sub.concept_ids.tolist() == [[7]]


def test_bucket_sampler_covers_each_index_once():
    lengths = np.random.default_rng(0).integers(1, 200, 503)
    s = LengthBucketBatchSampler(lengths, batch_size=16, bucket_factor=4)
    seen = [i for batch in s for i in batch]
    assert sorted(seen) == list(range(503))
    assert len(list(iter(s))) == len(s)


def test_legacy_adapter_roundtrip(tmp_path, raw):
    path = write_akt_legacy(raw, tmp_path / "legacy.csv", with_questions=True)
    df = AKTLegacyAdapter(files={"train": str(path)}, with_questions=True).load()
    assert len(df) == len(raw)
    assert df[S.USER].nunique() == raw[S.USER].nunique()
    assert df[S.CORRECT].sum() == raw[S.CORRECT].sum()
    assert set(df[S.SPLIT]) == {"train"}


def test_csv_adapter_column_mapping(tmp_path, raw):
    src = raw.rename(columns={S.USER: "student", S.CONCEPT: "skill", S.QUESTION: "problem", S.CORRECT: "ok"})
    p = tmp_path / "log.csv"
    src.to_csv(p, index=False)
    ad = build_adapter("csv", {"path": str(p), "columns": {S.USER: "student", S.CONCEPT: "skill",
                                                          S.QUESTION: "problem", S.CORRECT: "ok"}})
    assert isinstance(ad, CSVInteractionAdapter)
    df = ad.load()
    assert {S.USER, S.CONCEPT, S.QUESTION, S.CORRECT, S.TIMESTAMP} <= set(df.columns)
    assert len(df) == len(raw)


def test_pykt_adapter_parsing(tmp_path):
    tr_content = (
        "fold,uid,questions,concepts,responses,timestamps\n"
        "0,101,\"1,2,3\",\"10,20,30\",\"1,0,1\",\"100,200,300\"\n"
        "1,102,\"4,5\",\"40,50\",\"0,1\",\"400,500\"\n"
    )
    te_content = (
        "fold,uid,questions,concepts,responses,timestamps\n"
        "-1,201,\"6,7\",\"60,70\",\"1,1\",\"600,700\"\n"
    )
    (tmp_path / "train_valid.csv").write_text(tr_content)
    (tmp_path / "test.csv").write_text(te_content)

    ad = build_adapter("pykt", {"data_dir": str(tmp_path), "fold": 1, "with_questions": True})
    df = ad.load()
    assert len(df) == 7
    assert set(df[S.SPLIT].unique()) == {"valid", "train", "test"}
    assert df[df[S.SPLIT] == "valid"][S.USER].iloc[0] == "tr_101"
    assert df[df[S.SPLIT] == "train"][S.USER].iloc[0] == "tr_102"
    assert df[df[S.SPLIT] == "test"][S.USER].iloc[0] == "te_201"


def test_datamodule_meta_and_loaders(dm):
    m = dm.meta
    assert m.n_concepts == 12 and m.has_questions and m.has_timestamps
    b = next(iter(dm.train_dataloader(8)))
    assert b.batch_size <= 8 and b.seq_len <= 60
    assert int(b.concept_ids.max()) <= m.n_concepts and int(b.question_ids.max()) <= m.n_questions
    assert torch.all(b.concept_ids[~b.mask] == 0)


def test_datamodule_without_questions(dm_no_q):
    assert not dm_no_q.meta.has_questions
    assert next(iter(dm_no_q.train_dataloader(4))).question_ids is None


def test_datamodule_respects_predefined_split(tmp_path, raw):
    p = write_akt_legacy(raw, tmp_path / "a.csv")
    from ktbench.config import DataConfig
    from ktbench.data import KTDataModule
    cfg = DataConfig(adapter="akt_legacy", adapter_params={"files": {"train": str(p), "valid": str(p), "test": str(p)}})
    d = KTDataModule(cfg).setup()
    assert len(d.datasets["train"]) == len(d.datasets["valid"]) == len(d.datasets["test"])


def test_make_dm_helper_smoke():
    assert make_dm(n_students=20).meta.n_students == 20
