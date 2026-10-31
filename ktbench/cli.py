"""Command line interface.

    python -m ktbench train      -c configs/akt_synthetic.yaml [key=value ...]
    python -m ktbench benchmark  -c configs/benchmark_synthetic.yaml [key=value ...]
    python -m ktbench synth      --out data/synthetic --students 1000
    python -m ktbench list
"""

from __future__ import annotations

import argparse
import logging
import sys

from ktbench.config import ExperimentConfig


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ktbench", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train + evaluate one model on one dataset")
    t.add_argument("-c", "--config", required=True)
    t.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.lr=1e-3")

    b = sub.add_parser("benchmark", help="run a model x dataset x seed grid")
    b.add_argument("-c", "--config", required=True)
    b.add_argument("overrides", nargs="*")

    s = sub.add_parser("synth", help="generate a synthetic dataset (canonical CSV + AKT legacy format)")
    s.add_argument("--out", default="data/synthetic")
    s.add_argument("--students", type=int, default=1000)
    s.add_argument("--concepts", type=int, default=20)
    s.add_argument("--questions", type=int, default=400)
    s.add_argument("--max-len", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)

    sub.add_parser("list", help="list registered models and dataset adapters")

    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    if args.cmd == "train":
        from ktbench.experiment import run_experiment

        cfg = ExperimentConfig.from_yaml(args.config, args.overrides)
        res = run_experiment(cfg)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in res.test_metrics.items()})
    elif args.cmd == "benchmark":
        from ktbench.experiment import BenchmarkConfig, run_benchmark, summarise

        df = run_benchmark(BenchmarkConfig.from_yaml(args.config, args.overrides))
        print(summarise(df))
    elif args.cmd == "synth":
        from pathlib import Path

        from ktbench.data.synthetic import SyntheticConfig, generate_interactions, write_akt_legacy, write_canonical_csv

        cfg = SyntheticConfig(n_students=args.students, n_concepts=args.concepts,
                              n_questions=args.questions, max_len=args.max_len, seed=args.seed)
        df = generate_interactions(cfg)
        out = Path(args.out)
        print("wrote", write_canonical_csv(df, out / "interactions.csv"))
        print("wrote", write_akt_legacy(df, out / "synthetic_pid_all.csv", with_questions=True))
        print(f"{len(df)} interactions, {df.user_id.nunique()} students, correct rate {df.correct.mean():.3f}")
    elif args.cmd == "list":
        from ktbench.data.adapters import ADAPTER_REGISTRY
        from ktbench.models import MODEL_REGISTRY

        print("models:  ", ", ".join(MODEL_REGISTRY.names()))
        print("adapters:", ", ".join(ADAPTER_REGISTRY.names()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
