"""Hardware constraints, memory management, gradient accumulation equivalence,
OOM recovery, and full end-to-end training & evaluation proof-of-work."""

import copy

import pytest
import torch

from ktbench.config import DataConfig, ExperimentConfig, MemoryConfig, ModelConfig, TrainConfig
from ktbench.device import resolve_device
from ktbench.experiment import BenchmarkConfig, run_benchmark, run_experiment
from ktbench.models import build_model
from ktbench.training.losses import kt_loss_sum
from ktbench.training.memory import MemoryManager
from ktbench.training.trainer import Trainer
from tests.conftest import SMALL_PARAMS


def test_device_resolves_mps_on_apple_silicon():
    dev = resolve_device("auto")
    if torch.backends.mps.is_available():
        assert dev.type == "mps"
    else:
        assert dev.type in {"cuda", "cpu"}


def test_memory_manager_microbatch_planning_and_calibration(dm, device):
    model = build_model("akt", SMALL_PARAMS["akt"], dm.meta).to(device)
    mem = MemoryManager(MemoryConfig(memory_cap_gb=11.0, auto_budget=True), device, complexity=2)
    mem.install_cap()
    budget = mem.calibrate(
        model,
        dm.dummy_batch,
        lambda m, b: kt_loss_sum(m(b), b)[0] / max(b.num_tokens, 1),
        seq_len=dm.meta.max_seq_len,
    )
    assert budget > 0

    # Force a tight budget that allows at most 3 sequences of length 60: 3 * 60^2 = 10800
    mem.budget = 3 * (60**2)
    full_batch = dm.dummy_batch(batch_size=10, seq_len=60)
    micro_batches = mem.plan(full_batch)
    assert len(micro_batches) == 4  # 3 + 3 + 3 + 1
    assert sum(mb.batch_size for mb in micro_batches) == 10
    assert all(mb.batch_size <= 3 for mb in micro_batches)


def test_gradient_accumulation_is_exact_across_microbatches(dm, device):
    """Splitting a logical batch into arbitrary micro-batches must yield the
    exact same parameter gradients as a single unsplit batch."""
    torch.manual_seed(99)
    m_full = build_model("akt", SMALL_PARAMS["akt"], dm.meta).to(device)
    m_split = copy.deepcopy(m_full)
    m_full.eval()
    m_split.eval()

    batch = next(iter(dm.train_dataloader(8))).to(device)
    n_tokens = batch.num_tokens

    # 1) Single unsplit step
    out_full = m_full(batch)
    loss_full, _ = kt_loss_sum(out_full, batch)
    (loss_full / n_tokens).backward()

    # 2) Micro-batched step with max 2 rows per micro-batch
    mem = MemoryManager(MemoryConfig(cost_budget=2 * (batch.seq_len**2)), device, complexity=2)
    chunks = mem.plan(batch.to(torch.device("cpu")))
    assert len(chunks) >= 4
    for mb in chunks:
        mb = mb.to(device)
        out_mb = m_split(mb)
        loss_mb, _ = kt_loss_sum(out_mb, mb)
        (loss_mb / n_tokens).backward()

    for (name, p1), (_, p2) in zip(m_full.named_parameters(), m_split.named_parameters()):
        if p1.grad is not None:
            torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-4, atol=1e-5, msg=name)


def test_oom_recovery_halves_budget_and_completes_step(dm, device):
    """Simulate an OutOfMemoryError on the first attempt and verify that
    Trainer catches it, halves the micro-batch budget, and finishes the step."""
    model = build_model("dkt", SMALL_PARAMS["dkt"], dm.meta)
    trainer = Trainer(
        model,
        TrainConfig(epochs=1, batch_size=8, lr=1e-3),
        MemoryConfig(cost_budget=8 * 60, oom_retry=True, max_oom_retries=3),
        device=device,
    )
    real_forward = trainer.forward_model
    calls = {"n": 0}

    def flaky_forward(b):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.OutOfMemoryError("Simulated unified memory OOM")
        return real_forward(b)

    trainer.forward_model = flaky_forward
    batch = next(iter(dm.train_dataloader(8)))
    from ktbench.training.metrics import MetricAccumulator

    acc = MetricAccumulator()
    loss = trainer._train_step(batch, acc)
    assert loss > 0
    assert trainer.memory.oom_events == 1
    assert trainer.memory.budget == 0.5 * min(8 * 60, batch.batch_size * batch.seq_len)


def test_end_to_end_akt_training_and_evaluation(tmp_path):
    """Full proof-of-work: synthetic data -> AKT-R -> training -> checkpointing
    -> evaluation metrics (AUC, RMSE, ACC) on Apple Silicon MPS."""
    cfg = ExperimentConfig(
        name="pytest_e2e",
        seed=42,
        device="auto",
        output_dir=str(tmp_path),
        data=DataConfig(
            adapter="synthetic",
            adapter_params={"n_students": 250, "n_concepts": 15, "n_questions": 150,
                            "min_len": 25, "max_len": 100, "seed": 7},
            max_seq_len=100,
        ),
        model=ModelConfig(
            name="akt",
            params={"d_model": 64, "d_ff": 128, "n_heads": 4, "final_fc_dim": 64,
                    "final_fc_dim2": 32, "dropout": 0.05, "use_rasch": True},
        ),
        train=TrainConfig(epochs=4, batch_size=24, lr=1e-3, patience=4),
        memory=MemoryConfig(memory_cap_gb=11.0, enforce_cap=True, auto_budget=True),
    )
    res = run_experiment(cfg, save=True)
    # Loss must decrease and test AUC/RMSE must beat chance comfortably
    assert res.history[-1]["train_loss"] < res.history[0]["train_loss"]
    assert res.test_metrics["auc"] > 0.70
    assert 0.0 < res.test_metrics["rmse"] < 0.48
    assert 0.5 < res.test_metrics["acc"] <= 1.0
    assert res.peak_alloc_gb < 11.0


def test_head_to_head_benchmark_runner(tmp_path):
    """Compare AKT, DKT, and SAKT head-to-head on identical data splits."""
    bcfg = BenchmarkConfig(
        name="pytest_bench",
        seeds=[42],
        base={
            "device": "auto",
            "output_dir": str(tmp_path),
            "train": {"epochs": 2, "batch_size": 32, "lr": 2e-3, "patience": 2},
            "memory": {"memory_cap_gb": 11.0},
        },
        datasets={
            "synth_tiny": {
                "adapter": "synthetic",
                "adapter_params": {"n_students": 120, "n_concepts": 10, "n_questions": 60,
                                   "min_len": 20, "max_len": 80, "seed": 11},
                "max_seq_len": 80,
            }
        },
        models={
            "akt": {"name": "akt", "params": SMALL_PARAMS["akt"]},
            "dkt": {"name": "dkt", "params": SMALL_PARAMS["dkt"]},
            "sakt": {"name": "sakt", "params": SMALL_PARAMS["sakt"]},
        },
    )
    df = run_benchmark(bcfg, save=True)
    assert list(df["model"]) == ["akt", "dkt", "sakt"]
    assert (df["test_auc"] > 0.60).all()
    assert (df["test_rmse"] < 0.50).all()
