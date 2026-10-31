"""Side-by-side comparison of replicated 5-fold results against KDD 2020 AKT paper tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from scripts.run_akt_paper_benchmarks import evaluate_cached

PAPER_TABLE2 = {
    ("Statics2011", "DKT"): (0.8233, 0.0039),
    ("Statics2011", "DKT+"): (0.8301, 0.0039),
    ("Statics2011", "DKVMN"): (0.8195, 0.0041),
    ("Statics2011", "SAKT"): (0.8029, 0.0032),
    ("Statics2011", "AKT-NR"): (0.8265, 0.0049),
    ("ASSISTments2009", "DKT"): (0.8170, 0.0043),
    ("ASSISTments2009", "DKT+"): (0.8024, 0.0045),
    ("ASSISTments2009", "DKVMN"): (0.8093, 0.0044),
    ("ASSISTments2009", "SAKT"): (0.7520, 0.0040),
    ("ASSISTments2009", "AKT-NR"): (0.8169, 0.0045),
    ("ASSISTments2009", "AKT-R"): (0.8346, 0.0036),
    ("ASSISTments2015", "DKT"): (0.7310, 0.0018),
    ("ASSISTments2015", "DKT+"): (0.7313, 0.0018),
    ("ASSISTments2015", "DKVMN"): (0.7276, 0.0017),
    ("ASSISTments2015", "SAKT"): (0.7212, 0.0020),
    ("ASSISTments2015", "AKT-NR"): (0.7828, 0.0019),
    ("ASSISTments2017", "DKT"): (0.7263, 0.0054),
    ("ASSISTments2017", "DKT+"): (0.7124, 0.0041),
    ("ASSISTments2017", "DKVMN"): (0.7073, 0.0044),
    ("ASSISTments2017", "SAKT"): (0.6569, 0.0027),
    ("ASSISTments2017", "AKT-NR"): (0.7282, 0.0037),
    ("ASSISTments2017", "AKT-R"): (0.7702, 0.0026),
}


def main() -> None:
    out_dir = Path("runs/akt_paper_replication")
    ckpt = torch.load(out_dir / "cached_branches.pt", weights_only=False)
    evaluate_cached(
        ckpt["cached_folds"],
        out_dir,
        ckpt["table1_rows"],
        ckpt["interpretability"],
        ckpt["elapsed_s"],
        ckpt["device"],
    )
    df = pd.read_csv(out_dir / "results.csv")
    g = df.groupby(["dataset", "model"])["test_auc"]
    mean_s, std_s = g.mean(), g.std().fillna(0.0)
    print("\n=== Side-by-Side Comparison vs Paper Table 2 ===")
    for (ds, m), (p_m, p_s) in PAPER_TABLE2.items():
        r_m = mean_s.get((ds, m), float("nan"))
        r_s = std_s.get((ds, m), float("nan"))
        diff = r_m - p_m
        print(f"{ds:16s} | {m:8s} | Paper: {p_m:.4f} ± {p_s:.4f} | Replicated: {r_m:.4f} ± {r_s:.4f} | Diff: {diff:+.4f}")


if __name__ == "__main__":
    main()
