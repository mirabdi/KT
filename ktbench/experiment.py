"""Experiment orchestration: single runs and model x dataset x seed benchmarks."""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from ktbench.config import ExperimentConfig, ModelConfig, apply_overrides
from ktbench.data.datamodule import KTDataModule
from ktbench.device import resolve_device, seed_everything
from ktbench.models import build_model
from ktbench.training.trainer import Trainer

log = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    name: str
    model: str
    dataset: str
    seed: int
    best_epoch: int
    test_metrics: Dict[str, float]
    valid_metrics: Dict[str, float]
    n_params: int
    train_time_s: float
    peak_alloc_gb: float
    output_dir: Optional[str]
    history: List[Dict[str, Any]] = field(default_factory=list)

    def row(self) -> Dict[str, Any]:
        return {
            "model": self.model, "dataset": self.dataset, "seed": self.seed,
            **{f"test_{k}": v for k, v in self.test_metrics.items() if k != "n"},
            "valid_auc": self.valid_metrics.get("auc"),
            "best_epoch": self.best_epoch, "params": self.n_params,
            "train_time_s": round(self.train_time_s, 1), "peak_alloc_gb": self.peak_alloc_gb,
        }


def run_experiment(cfg: ExperimentConfig, datamodule: Optional[KTDataModule] = None,
                   save: bool = True) -> ExperimentResult:
    """Data -> model -> memory calibration -> fit -> test. Fully config-driven."""
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    dm = datamodule or KTDataModule(cfg.data, seed=cfg.seed).setup()
    assert dm.meta is not None

    out_dir = None
    if save:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = Path(cfg.output_dir) / cfg.name / f"{stamp}_{cfg.model.name}_{dm.meta.name}_s{cfg.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg.save_yaml(out_dir / "config.yaml")

    seed_everything(cfg.seed)  # identical init regardless of data-loading RNG use
    model = build_model(cfg.model.name, cfg.model.params, dm.meta)
    log.info("Model %s: %s params on %s", cfg.model.name, f"{model.num_parameters():,}", device)
    trainer = Trainer(model, cfg.train, cfg.memory, device, out_dir)
    trainer.calibrate_memory(dm.dummy_batch, seq_len=dm.meta.max_seq_len)

    bs = cfg.train.batch_size
    t0 = time.perf_counter()
    history = trainer.fit(dm.train_dataloader(bs), dm.val_dataloader(bs))
    train_time = time.perf_counter() - t0
    valid_m = trainer.evaluate(dm.val_dataloader(bs))
    test_m = trainer.evaluate(dm.test_dataloader(bs))
    log.info("[%s | %s] TEST auc %.4f rmse %.4f acc %.4f", cfg.model.name, dm.meta.name,
             test_m.get("auc", float("nan")), test_m.get("rmse", float("nan")), test_m.get("acc", float("nan")))

    result = ExperimentResult(
        name=cfg.name, model=cfg.model.name, dataset=dm.meta.name, seed=cfg.seed,
        best_epoch=trainer.best_epoch, test_metrics=test_m, valid_metrics=valid_m,
        n_params=model.num_parameters(), train_time_s=train_time,
        peak_alloc_gb=round(trainer.memory.peak_allocated / 2**30, 3),
        output_dir=str(out_dir) if out_dir else None, history=history,
    )
    if out_dir:
        with open(out_dir / "metrics.json", "w") as f:
            json.dump({"test": test_m, "valid": valid_m, "best_epoch": trainer.best_epoch,
                       "data": dm.summary(), "oom_events": trainer.memory.oom_events}, f, indent=2)
    return result


# ============================================================================
# Benchmarks
# ============================================================================
@dataclass
class BenchmarkConfig:
    """``base`` is a partial ExperimentConfig shared by all runs; ``models`` and
    ``datasets`` are named variants; every (dataset, model, seed) combination
    is executed on *identical* preprocessed data splits."""

    name: str = "benchmark"
    base: Dict[str, Any] = field(default_factory=dict)
    datasets: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    seeds: List[int] = field(default_factory=lambda: [42])
    folds: List[int] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: Optional[List[str]] = None) -> "BenchmarkConfig":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        raw = apply_overrides(raw, overrides or [])
        return cls(**raw)


def run_benchmark(bcfg: BenchmarkConfig, save: bool = True) -> pd.DataFrame:
    rows, results = [], []
    fold_list: List[Optional[int]] = list(bcfg.folds) if bcfg.folds else [None]
    for ds_name, ds_params in bcfg.datasets.items():
        for fold in fold_list:
            for seed in bcfg.seeds:
                base = ExperimentConfig.from_dict(copy.deepcopy(bcfg.base) | {"name": bcfg.name, "seed": seed})
                data_cfg = ExperimentConfig.from_dict({"data": {**base.to_dict()["data"], **ds_params}}).data
                data_cfg.adapter_params = {**data_cfg.adapter_params, "name": ds_name}
                if fold is not None and data_cfg.adapter in ("akt_legacy", "pykt"):
                    data_cfg.adapter_params["fold"] = fold
                dm = KTDataModule(data_cfg, seed=seed).setup()   # built ONCE, shared by all models
                for label, mparams in bcfg.models.items():
                    cfg = copy.deepcopy(base)
                    cfg.data = data_cfg
                    mp = copy.deepcopy(mparams)
                    overrides = mp.pop("train", {})
                    cfg.model = ModelConfig(name=mp.pop("name", label), params=mp.pop("params", {}))
                    if mp:
                        raise ValueError(f"Unknown keys in model entry '{label}': {list(mp)}")
                    for k, v in overrides.items():
                        setattr(cfg.train, k, v)
                    log.info("=== benchmark: dataset=%s fold=%s model=%s seed=%d ===", ds_name, fold, label, seed)
                    res = run_experiment(cfg, datamodule=dm, save=save)
                    row = res.row()
                    row["model"] = label
                    if fold is not None:
                        row["fold"] = fold
                    rows.append(row)
                    results.append(res)
    df = pd.DataFrame(rows)
    if save and len(df):
        out = Path(bcfg.base.get("output_dir", "runs")) / bcfg.name
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "results.csv", index=False)
        (out / "results.md").write_text(summarise(df))
        log.info("Benchmark results written to %s", out)
    return df


def summarise(df: pd.DataFrame) -> str:
    """Markdown table of mean +- std test metrics per (dataset, model)."""
    metrics = [c for c in ("test_auc", "test_rmse", "test_acc") if c in df.columns]
    g = df.groupby(["dataset", "model"])[metrics]
    mean, std = g.mean(), g.std().fillna(0.0)
    lines = ["| dataset | model | " + " | ".join(m.replace("test_", "").upper() for m in metrics) + " |",
             "|---|---|" + "---|" * len(metrics)]
    for (ds, m), r in mean.iterrows():
        cells = [f"{r[c]:.4f} ± {std.loc[(ds, m), c]:.4f}" for c in metrics]
        lines.append(f"| {ds} | {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


__all__ = ["run_experiment", "run_benchmark", "BenchmarkConfig", "ExperimentResult", "summarise"]
