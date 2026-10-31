"""Idea Exploration & Candidate Screening Runner for CogKT Variants.

Tests a matrix of novel architectural hypotheses across datasets:
- Idea 1: CogKT-Full (2PL-Rasch + Working Memory + Monotonic Attention + Cognitive Gate)
- Idea 2: CogKT-1PL (No 2PL discrimination parameter)
- Idea 3: CogKT-NoWM (Pure Attention, Working Memory disabled)
- Idea 4: CogKT-ConvSingle (Kernel 3 only)
- Idea 5: CogKT-ConvWide (Kernels 3, 5, 7)
- Idea 6: CogKT-AdaptiveDecay (Repetition-aware decay enabled)
- Idea 7: CogKT-NoAdaptiveDecay (Static per-head decay)
- Idea 8: CogKT-SlipGuess (3PL slip & guess bounds)
- Idea 9: CogKT-Deep (2 blocks / 4 layers retriever)
- Idea 10: CogKT-FastEdge (d_model=64, d_ff=128)
- Idea 11: CogKT-HighCap (d_model=256, d_ff=1024)
- Idea 12: CogKT-SeparateQA (separate interaction embeddings)
- Idea 13: CogKT-Dropout0 (zero dropout)
- Idea 14: CogKT-Dropout2 (0.2 dropout)
- Idea 15: CogKT-CosinePos (Sinusoidal PE added)

Evaluates on representative datasets and logs full performance metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch

from ktbench.config import DataConfig, ExperimentConfig, MemoryConfig, ModelConfig, TrainConfig
from ktbench.data.datamodule import KTDataModule
from ktbench.experiment import run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("idea_exploration")

CANDIDATE_IDEAS = {
    "CogKT-Full": {
        "d_model": 128, "d_ff": 512, "n_heads": 8, "n_blocks": 1, "dropout": 0.05,
        "use_rasch": True, "use_2pl": True, "use_working_memory": True,
        "conv_kernel_sizes": [3, 5], "conv_layers": 1, "monotonic": True,
        "adaptive_decay": True, "context_aware": True, "use_slip_guess": False,
    },
    "CogKT-1PL": {
        "d_model": 128, "d_ff": 512, "n_heads": 8, "n_blocks": 1, "dropout": 0.05,
        "use_rasch": True, "use_2pl": False, "use_working_memory": True,
        "conv_kernel_sizes": [3, 5], "conv_layers": 1, "monotonic": True,
        "adaptive_decay": True, "context_aware": True,
    },
    "CogKT-NoWM": {
        "d_model": 128, "d_ff": 512, "n_heads": 8, "n_blocks": 1, "dropout": 0.05,
        "use_rasch": True, "use_2pl": True, "use_working_memory": False,
        "monotonic": True, "adaptive_decay": True, "context_aware": True,
    },
    "CogKT-ConvSingle": {
        "d_model": 128, "d_ff": 512, "n_heads": 8, "n_blocks": 1, "dropout": 0.05,
        "use_rasch": True, "use_2pl": True, "use_working_memory": True,
        "conv_kernel_sizes": [3], "conv_layers": 1, "monotonic": True,
        "adaptive_decay": True, "context_aware": True,
    },
    "CogKT-ConvWide": {
        "d_model": 128, "d_ff": 512, "n_heads": 8, "n_blocks": 1, "dropout": 0.05,
        "use_rasch": True, "use_2pl": True, "use_working_memory": True,
        "conv_kernel_sizes": [3, 5, 7], "conv_layers": 1, "monotonic": True,
        "adaptive_decay": True, "context_aware": True,
    },
    "CogKT-NoAdaptiveDecay": {
        "d_model": 128, "d_ff": 512, "n_heads": 8, "n_blocks": 1, "dropout": 0.05,
        "use_rasch": True, "use_2pl": True, "use_working_memory": True,
        "conv_kernel_sizes": [3, 5], "conv_layers": 1, "monotonic": True,
        "adaptive_decay": False, "context_aware": True,
    },
    "CogKT-SlipGuess": {
        "d_model": 128, "d_ff": 512, "n_heads": 8, "n_blocks": 1, "dropout": 0.05,
        "use_rasch": True, "use_2pl": True, "use_working_memory": True,
        "conv_kernel_sizes": [3, 5], "conv_layers": 1, "monotonic": True,
        "adaptive_decay": True, "context_aware": True, "use_slip_guess": True,
    },
    "CogKT-Deep": {
        "d_model": 128, "d_ff": 512, "n_heads": 8, "n_blocks": 2, "dropout": 0.05,
        "use_rasch": True, "use_2pl": True, "use_working_memory": True,
        "conv_kernel_sizes": [3, 5], "conv_layers": 1, "monotonic": True,
        "adaptive_decay": True, "context_aware": True,
    },
    "CogKT-FastEdge": {
        "d_model": 64, "d_ff": 128, "n_heads": 4, "n_blocks": 1, "dropout": 0.05,
        "use_rasch": True, "use_2pl": True, "use_working_memory": True,
        "conv_kernel_sizes": [3, 5], "conv_layers": 1, "monotonic": True,
        "adaptive_decay": True, "context_aware": True,
    },
    "CogKT-HighCap": {
        "d_model": 256, "d_ff": 1024, "n_heads": 8, "n_blocks": 1, "dropout": 0.1,
        "use_rasch": True, "use_2pl": True, "use_working_memory": True,
        "conv_kernel_sizes": [3, 5], "conv_layers": 1, "monotonic": True,
        "adaptive_decay": True, "context_aware": True,
    },
    "CogKT-Dropout0": {
        "d_model": 128, "d_ff": 512, "n_heads": 8, "n_blocks": 1, "dropout": 0.0,
        "use_rasch": True, "use_2pl": True, "use_working_memory": True,
        "conv_kernel_sizes": [3, 5], "conv_layers": 1, "monotonic": True,
        "adaptive_decay": True, "context_aware": True,
    },
    "CogKT-Dropout2": {
        "d_model": 128, "d_ff": 512, "n_heads": 8, "n_blocks": 1, "dropout": 0.2,
        "use_rasch": True, "use_2pl": True, "use_working_memory": True,
        "conv_kernel_sizes": [3, 5], "conv_layers": 1, "monotonic": True,
        "adaptive_decay": True, "context_aware": True,
    },
    "CogKT-CosinePos": {
        "d_model": 128, "d_ff": 512, "n_heads": 8, "n_blocks": 1, "dropout": 0.05,
        "use_rasch": True, "use_2pl": True, "use_working_memory": True,
        "conv_kernel_sizes": [3, 5], "conv_layers": 1, "monotonic": True,
        "adaptive_decay": True, "context_aware": True, "pos_encoding": "cosine",
    },
    "CogKT-NoMono": {
        "d_model": 128, "d_ff": 512, "n_heads": 8, "n_blocks": 1, "dropout": 0.05,
        "use_rasch": True, "use_2pl": True, "use_working_memory": True,
        "conv_kernel_sizes": [3, 5], "conv_layers": 1, "monotonic": False,
        "adaptive_decay": False, "context_aware": True,
    },
    "CogKT-Raw": {
        "d_model": 128, "d_ff": 512, "n_heads": 8, "n_blocks": 1, "dropout": 0.05,
        "use_rasch": True, "use_2pl": True, "use_working_memory": True,
        "conv_kernel_sizes": [3, 5], "conv_layers": 1, "monotonic": True,
        "adaptive_decay": True, "context_aware": False,
    },
}

DATASET_CONFIGS = {
    "assist2009": {
        "name": "ASSISTments2009",
        "adapter": "akt_legacy",
        "adapter_params": {"dataset_name": "assist2009_pid", "data_dir": "data/assist2009_pid", "with_questions": True},
        "max_seq_len": 200,
        "batch_size": 64,
        "epochs": 3,
        "lr": 1e-3,
    },
    "assist2017": {
        "name": "ASSISTments2017",
        "adapter": "akt_legacy",
        "adapter_params": {"dataset_name": "assist2017_pid", "data_dir": "data/assist2017_pid", "with_questions": True},
        "max_seq_len": 200,
        "batch_size": 64,
        "epochs": 3,
        "lr": 1e-3,
    },
    "assist2015": {
        "name": "ASSISTments2015",
        "adapter": "akt_legacy",
        "adapter_params": {"dataset_name": "assist2015", "data_dir": "data/assist2015", "with_questions": False},
        "max_seq_len": 200,
        "batch_size": 64,
        "epochs": 3,
        "lr": 1e-3,
    },
    "statics": {
        "name": "Statics2011",
        "adapter": "akt_legacy",
        "adapter_params": {"dataset_name": "statics", "data_dir": "data/statics", "with_questions": False},
        "max_seq_len": 200,
        "batch_size": 64,
        "epochs": 3,
        "lr": 1e-3,
    },
    "algebra2005": {
        "name": "Algebra2005",
        "adapter": "pykt",
        "adapter_params": {"data_dir": "data/algebra2005", "with_questions": True},
        "max_seq_len": 200,
        "batch_size": 64,
        "epochs": 3,
        "lr": 1e-3,
    },
    "nips34": {
        "name": "NIPS34",
        "adapter": "pykt",
        "adapter_params": {"data_dir": "data/nips34", "with_questions": True},
        "max_seq_len": 200,
        "batch_size": 64,
        "epochs": 3,
        "lr": 1e-3,
    },
    "synthetic": {
        "name": "SyntheticRasch",
        "adapter": "synthetic",
        "adapter_params": {"n_students": 500, "n_concepts": 15, "n_questions": 150, "min_len": 20, "max_len": 150, "seed": 42, "use_questions": True},
        "max_seq_len": 150,
        "batch_size": 64,
        "epochs": 5,
        "lr": 1e-3,
    },
}


def run_single_idea(
    idea_name: str,
    idea_params: Dict[str, Any],
    ds_key: str,
    seed: int = 42,
    device: str = "auto",
    out_dir: Path = Path("runs/idea_exploration"),
) -> Dict[str, Any]:
    ds_info = DATASET_CONFIGS[ds_key]
    data_cfg = DataConfig(
        adapter=ds_info["adapter"],
        adapter_params=ds_info["adapter_params"],
        max_seq_len=ds_info["max_seq_len"],
    )

    exp_cfg = ExperimentConfig(
        name=f"explore_{idea_name}_{ds_key}",
        seed=seed,
        device=device,
        output_dir=str(out_dir),
        data=data_cfg,
        model=ModelConfig(name="cogkt", params=idea_params),
        train=TrainConfig(
            epochs=ds_info["epochs"],
            batch_size=ds_info["batch_size"],
            lr=ds_info["lr"],
            optimizer="adam",
            monitor="auc",
        ),
        memory=MemoryConfig(amp=False),
    )

    dm = KTDataModule(data_cfg, seed=seed).setup()
    res = run_experiment(exp_cfg, datamodule=dm, save=True)

    return {
        "idea": idea_name,
        "dataset": ds_info["name"],
        "seed": seed,
        "test_auc": res.test_metrics.get("auc", float("nan")),
        "test_rmse": res.test_metrics.get("rmse", float("nan")),
        "test_acc": res.test_metrics.get("acc", float("nan")),
        "valid_auc": res.valid_metrics.get("auc", float("nan")),
        "best_epoch": res.best_epoch,
        "n_params": res.n_params,
        "train_time_s": res.train_time_s,
        "peak_alloc_gb": res.peak_alloc_gb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ideas", nargs="+", default=list(CANDIDATE_IDEAS.keys()))
    parser.add_argument("--datasets", nargs="+", default=["synthetic", "assist2009"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="runs/idea_exploration")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "exploration_results.csv"

    rows: List[Dict[str, Any]] = []
    if csv_path.exists():
        rows = pd.read_csv(csv_path).to_dict("records")
        log.info("Loaded %d existing exploration records", len(rows))

    done_keys = {(r["idea"], r["dataset"]) for r in rows}

    for ds_key in args.datasets:
        ds_name = DATASET_CONFIGS[ds_key]["name"]
        for idea_name in args.ideas:
            if (idea_name, ds_name) in done_keys:
                log.info("Skipping already evaluated: %s on %s", idea_name, ds_name)
                continue
            log.info(">>> Running idea [%s] on dataset [%s]...", idea_name, ds_name)
            params = CANDIDATE_IDEAS[idea_name]
            try:
                row = run_single_idea(idea_name, params, ds_key, seed=args.seed, device=args.device, out_dir=out_dir)
                rows.append(row)
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                log.info("Finished [%s | %s]: TEST AUC = %.4f, RMSE = %.4f, ACC = %.4f (%.1fs)",
                         idea_name, ds_name, row["test_auc"], row["test_rmse"], row["test_acc"], row["train_time_s"])
            except Exception as e:
                log.exception("Error running idea %s on %s: %s", idea_name, ds_name, e)

    log.info("All idea exploration completed. Summary saved to %s", csv_path)


if __name__ == "__main__":
    main()
